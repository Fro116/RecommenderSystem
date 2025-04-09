import asyncio
import contextlib
import datetime
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import traceback
import time
from enum import Enum, auto

import msgpack
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response


class ErrorCode(Enum):
    INTERNAL_SERVER_ERROR = auto()
    QUEUE_FULL = auto()


class RequestBuffer:
    def __init__(self, batch_size, timeout_seconds, max_queue_size):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_queue_size = max_queue_size
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
                    predictions = predict(batch)
                    with self.lock:
                        for i, request_id in enumerate(request_ids):
                            self.results[request_id] = (predictions[i], time.time())
            except Exception as e:
                logging.error(f"Error in RequestBuffer: {e} \n {traceback.format_exc()}")
                with self.lock:
                    for request_id in request_ids:
                        self.results[request_id] = (
                            ErrorCode.INTERNAL_SERVER_ERROR,
                            time.time(),
                        )


def get_user_bias(data):
    biases = {}
    for m in mediums:
        _, λ_u, _, λ_wu, λ_wa = baselines[m]["params"]
        numer = 0
        denom = np.exp(λ_u)
        ucount = len([x for x in data["items"] if x["rating"] > 0 and x["medium"] == m])
        for x in data["items"]:
            if x["medium"] != m or x["rating"] == 0:
                continue
            acount = baselines[m]["item_counts"][x["matchedid"]]
            w = ucount**λ_wu * acount**λ_wa
            numer += (x["rating"] - baselines[m]["a"][x["matchedid"]]) * w
            denom += w
        biases[m] = float(numer / denom)
    return biases


def predict_baseline(users):
    ret = []
    for u in range(len(users)):
        d = {}
        biases = get_user_bias(users[u])
        for m in mediums:
            for metric in metrics:
                if metric == "rating":
                    d[f"baseline.{m}.{metric}"] = [biases[m]]
                else:
                    d[f"baseline.{m}.{metric}"] = [0]
        ret.append(d)
        for x in users[u]["items"]:
            if x["rating"] > 0:
                m = x["medium"]
                bl = baselines[m]
                rating_pred = (biases[m] + bl["a"][x["matchedid"]]) * bl["weight"]
                x["rating.resid"] = x["rating"] - rating_pred
            else:
                x["rating.resid"] = 0
    return ret


def predict_bagofwords(users):
    X = {}
    for m in mediums:
        for metric in ["rating", "watch"]:
            X[f"{m}_{metric}"] = np.zeros((len(users), num_items[m]), dtype=np.float32)
    for i, u in enumerate(users):
        for x in u["items"]:
            m = x["medium"]
            idx = x["matchedid"]
            X[f"{m}_rating"][i, idx] = x["rating.resid"]
            if x["rating"] > 0:
                X[f"{m}_rating"][i, idx] = x["rating.resid"]
            else:
                X[f"{m}_rating"][i, idx] = 0
            if x["status"] > planned_status:
                X[f"{m}_watch"][i, idx] = 1
            else:
                X[f"{m}_watch"][i, idx] = 0
    data = {k: torch.tensor(v) for k, v in X.items()}
    ret = {}
    d = {}
    for m in mediums:
        d[f"X_{m}_rating"] = data[f"{m}_rating"].to(device).to_dense()
        d[f"X_{m}_watch"] = data[f"{m}_watch"].to(device).to_dense()
    X = torch.cat(
        (d[f"X_0_rating"], d[f"X_0_watch"], d[f"X_1_rating"], d[f"X_1_watch"]), dim=1
    )
    for medium in mediums:
        for metric in bagofwords_metrics:
            k = f"bagofwords.{medium}.{metric}"
            with torch.no_grad():
                context = (
                    contextlib.nullcontext()
                    if device == "cpu"
                    else torch.amp.autocast(device, dtype=torch.bfloat16)
                )
                with context:
                    ret[k] = models[k](X, None, None, mode="embed").to("cpu")
    ret = [{k: v[i, :].tolist() for k, v in ret.items()} for i in range(len(users))]
    return ret


def predict_transformer(users):
    max_len = max(min(max_seq_len, len(x["items"]) + 2) for x in users)
    d = {
        "status": np.zeros((len(users), max_len), dtype=np.int32),
        "rating": np.zeros((len(users), max_len), dtype=np.float32),
        "progress": np.zeros((len(users), max_len), dtype=np.float32),
        "0_matchedid": np.zeros((len(users), max_len), dtype=np.int32),
        "0_distinctid": np.zeros((len(users), max_len), dtype=np.int32),
        "1_matchedid": np.zeros((len(users), max_len), dtype=np.int32),
        "1_distinctid": np.zeros((len(users), max_len), dtype=np.int32),
        "updated_at": np.zeros((len(users), max_len), dtype=np.float32),
        "source": np.zeros((len(users), max_len), dtype=np.int32),
    }
    input_fields = list(d.keys())
    d["userid"] = np.zeros((len(users), max_len), dtype=np.int32)
    d["mask_index"] = np.zeros((len(users), max_len), dtype=np.int32)
    user_biases = []
    for u in range(len(users)):
        biases = get_user_bias(users[u])
        user_biases.append(biases)
        userid = u + 1
        for k in input_fields:
            d[k][u, 0] = cls_val
        d["userid"][u, 0] = userid
        d["rating"][u, 0] = 0
        skipped = 0
        items = users[u]["items"]
        while len(items) > max_len - 2:
            items = items[1:]
            skipped += 1
        i = 0
        for x in items:
            i += 1
            m = x["medium"]
            n = 1 - x["medium"]
            d["status"][u, i] = x["status"]
            d["rating"][u, i] = x["rating.resid"]
            d["progress"][u, i] = x["progress"]
            d[f"{m}_matchedid"][u, i] = x["matchedid"]
            d[f"{m}_distinctid"][u, i] = x["distinctid"]
            d[f"{n}_matchedid"][u, i] = cls_val
            d[f"{n}_distinctid"][u, i] = cls_val
            d["updated_at"][u, i] = np.clip(
                (x["updated_at"] - min_ts) / (max_ts - min_ts), 0, 1
            )
            d["source"][u, i] = users[u]["user"]["source"]
            d["userid"][u, i] = userid
            d["mask_index"][u, i] = 0
        i += 1
        for k in input_fields:
            d[k][u, i] = mask_val
        d["userid"][u, i] = userid
        d["updated_at"][u, i] = 1
        d["mask_index"][u, i] = 1
        d["rating"][u, i] = 0
    d = {k: torch.tensor(v).to(device) for k, v in d.items()}
    userid = d["userid"]
    m, n = userid.shape
    attn_mask = userid.reshape(m, 1, n) != userid.reshape(m, n, 1)
    d["attn_mask"] = attn_mask
    del d["userid"]
    ret = {}
    for medium in mediums:
        with torch.no_grad():
            context = (
                contextlib.nullcontext()
                if device == "cpu"
                else torch.amp.autocast(device, dtype=torch.bfloat16)
            )
            with context:
                e = models[f"transformer.{medium}"](d).to("cpu")
                ret[f"transformer.{medium}"] = e
    ret = [{k: v[i, :].tolist() for k, v in ret.items()} for i in range(len(users))]
    return ret


def predict(users):
    def merge(X, Y):
        return [x | y for (x, y) in zip(X, Y)]

    ret = [{"version": version} for _ in users]
    predict_baseline(users)
    ret = merge(ret, predict_baseline(users))
    ret = merge(ret, predict_bagofwords(users))
    ret = merge(ret, predict_transformer(users))
    return ret


def get_baselines(rank):
    baselines = {}
    for m in mediums:
        with open(f"{datadir}/baseline.{m}.msgpack", "rb") as f:
            baseline = msgpack.unpackb(f.read(), strict_map_key=False)
            d = {}
            d["params"] = baseline["params"]["λ"]
            d["weight"] = baseline["weight"][0]
            d["a"] = torch.tensor(baseline["params"]["a"]).to(rank)
            item_counts = baseline["params"]["item_counts"]
            item_counts = [
                item_counts.get(x, 1) for x in range(len(baseline["params"]["a"]))
            ]
            d["item_counts"] = torch.tensor(item_counts).to(rank)
            baselines[m] = d
    return baselines


with open("../Training/bagofwords.model.py") as f:
    exec(f.read())

with open("../Training/transformer.model.py") as f:
    exec(f.read())


def load_bagofwords_model(medium, metric):
    fn = f"{datadir}/bagofwords.{medium}.{metric}.finetune.pt"
    m = BagOfWordsModel(datadir, medium, metric)
    m.load_state_dict(torch.load(fn, weights_only=True, map_location=device))
    m = m.to(device)
    m.eval()
    return m


def load_transformer_model(medium):
    # TODO serialize config
    num_items = {
        x: pd.read_csv(f"{datadir}/{y}.csv").matchedid.max() + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    config = {
        "dropout": 0.1,
        "activation": "gelu",
        "num_layers": 4,
        "embed_size": 768,
        "intermediate_size": 768 * 4,
        "max_sequence_length": max_seq_len,
        "vocab_names": [
            "0_matchedid",
            "1_matchedid",
            "rating",
            "status",
            "updated_at",
        ],
        "vocab_sizes": [
            num_items[0],
            num_items[1],
            None,
            8,
            None,
        ],
        "num_attention_heads": 12,
        "forward": "embed",
    }
    m = TransformerModel(config)
    fn = f"{datadir}/transformer.{medium}.finetune.pt"
    m.load_state_dict(torch.load(fn, weights_only=True, map_location=device))
    m = m.to(device)
    m.eval()
    return m


logging.warning("STARTUP BEGIN")
starttime = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
mediums = [0, 1]
metrics = ["rating", "watch", "plantowatch", "drop"]
bagofwords_metrics = ["rating"]
datadir = "../../data/finetune"
models = {}
for medium in mediums:
    models[f"transformer.{medium}"] = load_transformer_model(medium)
    for metric in bagofwords_metrics:
        models[f"bagofwords.{medium}.{metric}"] = load_bagofwords_model(medium, metric)
num_items = {
    m: pd.read_csv(f"{datadir}/{n}.csv").matchedid.max() + 1
    for m, n in [(0, "manga"), (1, "anime")]
}
planned_status = 3
baselines = get_baselines(device)
request_buffer = RequestBuffer(
    batch_size=256, timeout_seconds=0.01, max_queue_size=1024
)
with open(f"{datadir}/latest") as f:
    version = f.read()
min_ts = int(
    datetime.datetime.strptime("20000101", "%Y%m%d")
    .replace(tzinfo=datetime.timezone.utc)
    .timestamp()
)
max_ts = int(
    datetime.datetime.strptime(version, "%Y%m%d")
    .replace(tzinfo=datetime.timezone.utc)
    .timestamp()
)
predict([{"user": {"source": "mal"}, "items": []}])
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
