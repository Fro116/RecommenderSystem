import asyncio
import contextlib
import logging
import os
import queue
import threading
import time
import traceback
import warnings
from enum import Enum, auto

import msgpack
import numpy as np
import torch
import torch.nn as nn
import torchtune.models.llama3
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from torch.nn.attention.flex_attention import and_masks, create_block_mask
from torchtune.modules.peft import get_adapter_params, set_trainable_params

warnings.filterwarnings("ignore", ".*Initializing zero-element tensors is a no-op.*")


class ErrorCode(Enum):
    INTERNAL_SERVER_ERROR = auto()
    QUEUE_FULL = auto()


class RequestBuffer:
    def __init__(self, batch_size, timeout_seconds, max_queue_size, batch_fn):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_queue_size = max_queue_size
        self.batch_fn = batch_fn
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.results = {}
        self.lock = threading.Lock()
        self.next_request_id = 0
        self.batch_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.batch_thread.start()

    def add_request(self, input_data):
        with self.lock:
            request_id = self.next_request_id
            self.next_request_id += 1
        try:
            self.queue.put((request_id, input_data), block=False)
            return request_id
        except queue.Full:
            return ErrorCode.QUEUE_FULL

    def get_result(self, request_id):
        with self.lock:
            res, _ = self.results.pop(request_id, (None, None))
            return res

    def gc_results(self):
        with self.lock:
            todelete = []
            for k in self.results:
                _, ts = self.results[k]
                if time.time() - ts > 60:
                    todelete.append(k)
            for k in todelete:
                del self.results[k]

    def _process_batches(self):
        while True:
            self.gc_results()
            batch = []
            request_ids = []
            try:
                while len(batch) < self.batch_size:
                    try:
                        request_id, input_data = self.queue.get(
                            timeout=self.timeout_seconds
                        )
                        batch.append(input_data)
                        request_ids.append(request_id)
                    except queue.Empty:
                        if batch:
                            break
                        else:
                            continue
                if batch:
                    with gpu_lock:
                        predictions = self.batch_fn(batch)
                    with self.lock:
                        for i, request_id in enumerate(request_ids):
                            self.results[request_id] = (predictions[i], time.time())
            except Exception as e:
                logging.error(
                    f"Error in RequestBuffer: {e} \n {traceback.format_exc()}"
                )
                with self.lock:
                    for request_id in request_ids:
                        self.results[request_id] = (
                            ErrorCode.INTERNAL_SERVER_ERROR,
                            time.time(),
                        )


def make_test_item(medium, matchedid, distinctid):
    return {
        "medium": medium,
        "history_max_ts": time.time(),
        "matchedid": matchedid,
        "distinctid": distinctid,
        "status": 0,
        "rating": 0,
        "progress": 0,
        "history_status": 0,
        "history_rating": 0,
    }


def project(user):
    items = []
    for x in user["items"]:
        if (x["history_status"] == x["status"]) and (
            x["history_rating"] == x["rating"]
        ):
            continue
        items.append(x)
    return items


def shift_left(x):
    y = torch.zeros_like(x)
    y[..., :-1] = x[..., 1:]
    return y


def predict(users, modeltype, medium):
    model = models[f"transformer.{modeltype}.{medium}"]
    max_len = model.config["max_sequence_length"]
    d = {
        # prompt features
        "userid": np.zeros((len(users), max_len), dtype=np.int32),
        "time": np.zeros((len(users), max_len), dtype=np.float64),
        "input_pos": np.zeros((len(users), max_len), dtype=np.int32),
        "rope_input_pos": np.zeros((len(users), max_len), dtype=np.int32),
        # item features
        "0_matchedid": np.zeros((len(users), max_len), dtype=np.int32),
        "0_distinctid": np.zeros((len(users), max_len), dtype=np.int32),
        "1_matchedid": np.zeros((len(users), max_len), dtype=np.int32),
        "1_distinctid": np.zeros((len(users), max_len), dtype=np.int32),
        # action features
        "status": np.zeros((len(users), max_len), dtype=np.int32),
        "rating": np.zeros((len(users), max_len), dtype=np.float32),
        "progress": np.zeros((len(users), max_len), dtype=np.float32),
    }
    for u in range(len(users)):
        userid = u + 1
        items = project(users[u])
        extra_tokens = 1
        if len(items) > max_len - extra_tokens:
            items = items[-(max_len - extra_tokens) :]
        if modeltype == "ranking":
            test_items = users[u]["test_items"]
        elif modeltype == "retrieval":
            test_items = [make_test_item(medium, 0, 0)]
        else:
            assert False
        for i, x in enumerate(items + test_items):
            m = x["medium"]
            n = 1 - x["medium"]
            # prompt features
            d["userid"][u, i] = userid
            d["time"][u, i] = x["history_max_ts"]
            d["input_pos"][u, i] = 0  # TODO remove
            d["rope_input_pos"][u, i] = min(i, len(items))
            # item features
            d[f"{m}_matchedid"][u, i] = x["matchedid"]
            d[f"{m}_distinctid"][u, i] = x["distinctid"]
            d[f"{n}_matchedid"][u, i] = -1
            d[f"{n}_distinctid"][u, i] = -1
            # action features
            d["status"][u, i] = x["status"]
            d["rating"][u, i] = x["rating"]
            d["progress"][u, i] = x["progress"]

    d = {k: torch.tensor(v).to(device) for k, v in d.items()}
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            e = model(d).to("cpu")
    if modeltype == "retrieval":
        ret = []
        for i, u in enumerate(users):
            d = {"version": version}
            d[f"{modeltype}.{medium}"] = e[i, 2 * len(u["items"]), :].tolist()
            ret.append(d)
        return ret
    elif modeltype == "ranking":
        ret = []
        for i, u in enumerate(users):
            d = {"version": version}
            idxs = []
            for j in range(len(u["test_items"])):
                idxs.append(2 * (len(u["items"]) + j) + 1)
            d[f"{modeltype}.{medium}"] = e[i, idxs, 0].tolist()
            ret.append(d)
        return ret
    else:
        assert False


with open("../Training/transformer.model.py") as f:
    exec(f.read())


def load_model(modeltype, medium):
    checkpoint = torch.load(
        f"{datadir}/transformer.{modeltype}.{medium}.finetune.pt",
        weights_only=False,
        map_location=device,
    )
    config = checkpoint["config"]
    config["forward"] = "inference"
    model = TransformerModel(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model = torch.compile(model)
    model.eval()
    return model


logging.warning("STARTUP BEGIN")
starttime = time.time()
device = "cuda"
datadir = "../../data/finetune"
with open(f"{datadir}/finetune_tag") as f:
    version = f.read()
gpu_lock = threading.Lock()
models = {}
for modeltype in ["ranking", "retrieval"]:
    for medium in [0, 1]:
        logging.warning(f"STARTUP LOADING {modeltype} {medium} MODEL")
        models[f"transformer.{modeltype}.{medium}"] = load_model(
            modeltype, medium
        )
        test_user = {"user": {"source": "mal"}, "items": [], "test_items": [make_test_item(medium, 0, 0)]}
        predict([test_user], modeltype, medium)
request_buffers = {}
for modeltype in ["ranking", "retrieval"]:
    for medium in [0, 1]:
        request_buffers[(modeltype, medium)] = RequestBuffer(
            batch_size=32, timeout_seconds=0.01, max_queue_size=1024, batch_fn = lambda x: predict(x, modeltype, medium)
        )
logging.warning(f"STARTUP END AFTER {time.time() - starttime}s")
app = FastAPI()


@app.get("/shutdown")
async def shutdown_endpoint():
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200)


@app.get("/ready")
async def ready():
    return Response(status_code=200)


@app.post("/embed")
async def predict_endpoint(request: Request):
    data = await request.body()
    try:
        data = msgpack.unpackb(data, strict_map_key=False)
        modeltype = data["modeltype"]
        medium = data["medium"]
        request_buffer = request_buffers[(modeltype, medium)]
        r = request_buffer.add_request(data)
        if isinstance(r, ErrorCode):
            return Response(status_code=503)
        result = request_buffer.get_result(r)
        ts = time.time()
        while result is None:
            if time.time() - ts > 60:
                result = ErrorCode.INTERNAL_SERVER_ERROR
            await asyncio.sleep(0.1)
            result = request_buffer.get_result(r)
        if isinstance(result, ErrorCode):
            return Response(status_code=500)
        rdata = msgpack.packb(result, use_single_float=True)
        headers = {
            "Content-Type": "application/msgpack",
            "Content-Length": str(len(rdata)),
        }
        return Response(status_code=200, headers=headers, content=rdata)
    except Exception as e:
        logging.error(f"Error during request: {e}")
        return Response(status_code=500)
