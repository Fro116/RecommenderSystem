import asyncio
import contextlib
import functools
import logging
import os
import queue
import signal
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


def make_masked_item(ts):
    return {
        "medium": 0,
        "history_max_ts": ts,
        "matchedid": -1,
        "distinctid": -1,
        "status": -1,
        "rating": 0,
        "progress": 0,
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


def predict(users, task, medium):
    max_user_len = 1024
    if task == "retrieval":
        modeltypes = ["masked"]
        max_seq_len = 1024
    elif task == "ranking":
        modeltypes = ["causal"]
        max_seq_len = 2048
    else:
        assert False
    d = {
        modeltype: {
            # prompt features
            "userid": np.zeros((len(users), max_seq_len), dtype=np.int32),
            "time": np.zeros((len(users), max_seq_len), dtype=np.float64),
            "rope_input_pos": np.zeros((len(users), max_seq_len), dtype=np.int32),
            # item features
            "0_matchedid": np.zeros((len(users), max_seq_len), dtype=np.int32),
            "0_distinctid": np.zeros((len(users), max_seq_len), dtype=np.int32),
            "1_matchedid": np.zeros((len(users), max_seq_len), dtype=np.int32),
            "1_distinctid": np.zeros((len(users), max_seq_len), dtype=np.int32),
            # action features
            "status": np.zeros((len(users), max_seq_len), dtype=np.int32),
            "rating": np.zeros((len(users), max_seq_len), dtype=np.float32),
            "progress": np.zeros((len(users), max_seq_len), dtype=np.float32),
        }
        for modeltype in modeltypes
    }
    for modeltype in modeltypes:
        for u in range(len(users)):
            userid = u + 1
            items = project(users[u])
            extra_tokens = 1
            if len(items) > max_user_len - extra_tokens:
                items = items[-(max_user_len - extra_tokens) :]
            if modeltype == "causal" and task == "ranking":
                test_items = users[u]["test_items"]
            else:
                test_items = [make_masked_item(users[u]["timestamp"])]
            for i, x in enumerate(items + test_items):
                m = x["medium"]
                n = 1 - x["medium"]
                # prompt features
                d[modeltype]["userid"][u, i] = userid
                d[modeltype]["time"][u, i] = x["history_max_ts"]
                if modeltype == "causal":
                    d[modeltype]["rope_input_pos"][u, i] = min(i, len(items))
                # item features
                d[modeltype][f"{m}_matchedid"][u, i] = x["matchedid"]
                d[modeltype][f"{m}_distinctid"][u, i] = x["distinctid"]
                d[modeltype][f"{n}_matchedid"][u, i] = -1
                d[modeltype][f"{n}_distinctid"][u, i] = -1
                # action features
                d[modeltype]["status"][u, i] = x["status"]
                d[modeltype]["rating"][u, i] = x["rating"]
                d[modeltype]["progress"][u, i] = x["progress"]
    for k1 in d:
        for k2 in d[k1]:
            d[k1][k2] = torch.tensor(d[k1][k2]).to(device)
    e = {}
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            for modeltype in modeltypes:
                e[modeltype] = models[f"transformer.{modeltype}.{medium}"](
                    d[modeltype], task
                ).to("cpu")
    if task == "retrieval":
        ret = []
        for i, u in enumerate(users):
            d = {"version": version}
            N = min(len(project(u)), max_user_len - 1)
            d[f"masked.{medium}"] = e["masked"][i, N, :].tolist()
            ret.append(d)
        return ret
    elif task == "ranking":
        ret = []
        for i, u in enumerate(users):
            d = {"version": version}
            N = min(len(project(u)), max_user_len - 1)
            causal_idxs = []
            for j in range(len(u["test_items"])):
                causal_idxs.append(2 * (N + j) + 1)
            d[f"causal.{medium}"] = e["causal"][i, causal_idxs, 0].tolist()
            ret.append(d)
        return ret
    else:
        assert False


with open("../Training/transformer.model.py") as f:
    exec(f.read())


def get_nested_attr(obj, attr_string):
    return functools.reduce(getattr, attr_string.split("."), obj)


def set_nested_attr(obj, attr_string, value):
    parts = attr_string.split(".")
    current_obj = obj
    for part in parts[:-1]:
        current_obj = getattr(current_obj, part)
    setattr(current_obj, parts[-1], value)


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
request_buffers = {}
for modeltype in ["causal", "masked"]:
    for medium in [0, 1]:
        logging.warning(f"STARTUP LOADING {modeltype} {medium} MODEL")
        models[f"transformer.{modeltype}.{medium}"] = load_model(modeltype, medium)
    # deduplicate shared params from LoRA
    for k, _ in models[f"transformer.{modeltype}.0"].named_parameters():
        if (
            get_nested_attr(models[f"transformer.{modeltype}.0"], k)
            == get_nested_attr(models[f"transformer.{modeltype}.1"], k)
        ).all():
            set_nested_attr(
                models[f"transformer.{modeltype}.1"],
                k,
                get_nested_attr(models[f"transformer.{modeltype}.0"], k),
            )
    torch.cuda.empty_cache()
for task in ["retrieval", "ranking"]:
    for medium in [0, 1]:
        test_user = {
            "user": {"source": "mal"},
            "items": [],
            "timestamp": time.time(),
            "test_items": [make_masked_item(time.time())],
        }
        predict([test_user], task, medium)
        request_buffers[(task, medium)] = RequestBuffer(
            batch_size=32,
            timeout_seconds=0.01,
            max_queue_size=1024,
            batch_fn=lambda x, t=task, m=medium: predict(x, t, m),
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
async def predict_endpoint(request: Request, medium: int, task: str):
    data = await request.body()
    try:
        data = msgpack.unpackb(data, strict_map_key=False)
        request_buffer = request_buffers[(task, medium)]
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