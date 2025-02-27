import asyncio
import contextlib
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
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
                logging.error(f"Error in RequestBuffer: {e}")
                with self.lock:
                    for request_id in request_ids:
                        self.results[request_id] = (
                            ErrorCode.INTERNAL_SERVER_ERROR,
                            time.time(),
                        )


def predict_bagofwords(data, baselines, rank):
    ret = {}
    d = {}
    for m in [0, 1]:
        d[f"X_{m}_rating"] = data[f"{m}_rating"].to(rank).to_dense()
        d[f"X_{m}_watch"] = data[f"{m}_watch"].to(rank).to_dense()
        r = d[f"X_{m}_rating"]
        _, λ_u, _, λ_wu, λ_wa = baselines[m]["params"]
        user_count = (r != 0).sum(dim=1)
        user_weights = user_count**λ_wu
        user_weights[user_count == 0] = 0
        item_count = (r != 0) * baselines[m]["item_counts"]
        item_weights = item_count**λ_wa
        item_weights[item_count == 0] = 0
        weights = user_weights.reshape(-1, 1) * item_weights
        user_baseline = ((r - baselines[m]["a"]) * weights).sum(dim=1) / (
            weights.sum(dim=1) + np.exp(λ_u)
        )
        for metric in metrics:
            if metric == "rating":
                ret[f"baseline.{m}.{metric}"] = user_baseline.reshape(-1, 1)
            elif metric in ["watch", "plantowatch", "drop"]:
                ret[f"baseline.{m}.{metric}"] = 0 * user_baseline.reshape(-1, 1)
            else:
                assert False
        pred = (
            user_baseline.reshape(-1, 1) + baselines[m]["a"].reshape(1, -1)
        ) * baselines[m]["weight"]
        d[f"X_{m}_rating"] = (d[f"X_{m}_rating"] != 0) * (d[f"X_{m}_rating"] - pred)
    X = torch.cat(
        (d[f"X_0_rating"], d[f"X_0_watch"], d[f"X_1_rating"], d[f"X_1_watch"]), dim=1
    )
    for medium in mediums:
        for metric in metrics:
            k = f"bagofwords.{medium}.{metric}"
            with torch.no_grad():
                ret[k] = models[k](X, None, None, mode="embed")
    return ret


def predict(users):
    X = {}
    for m in mediums:
        for metric in metrics:
            X[f"{m}_{metric}"] = np.zeros((len(users), num_items[m]), dtype=np.float32)
    for i, u in enumerate(users):
        for x in u["items"]:
            m = x["medium"]
            idx = x["matchedid"]
            if x["rating"] > 0:
                X[f"{m}_rating"][i, idx] = x["rating"]
            else:
                X[f"{m}_rating"][i, idx] = 0
            if x["status"] > planned_status:
                X[f"{m}_watch"][i, idx] = 1
            else:
                X[f"{m}_watch"][i, idx] = 0
            if x["status"] == planned_status:
                X[f"{m}_plantowatch"][i, idx] = 1
            else:
                X[f"{m}_plantowatch"][i, idx] = 0
            if x["status"] > 0 and x["status"] < planned_status:
                X[f"{m}_drop"][i, idx] = 1
            else:
                X[f"{m}_drop"][i, idx] = 0
    data = {k: torch.tensor(v) for k, v in X.items()}
    ret = predict_bagofwords(data, baselines, device)
    ret = [{k: v[i, :].tolist() for k, v in ret.items()} for i in range(len(users))]
    for x in ret:
        x["version"] = version
    return ret


def get_baselines(rank):
    baselines = {}
    for m in [0, 1]:
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


def load_model(medium, metric):
    fn = f"{datadir}/bagofwords.{medium}.{metric}.finetune.pt"
    m = BagOfWordsModel(datadir, medium, metric)
    m.load_state_dict(
        torch.load(fn, weights_only=True, map_location=device)
    )
    m.eval()
    return m


device = "cpu"
mediums = [0, 1]
metrics = ["rating", "watch", "plantowatch", "drop"]
datadir = "../../data/finetune"
models = {
    f"bagofwords.{medium}.{metric}": load_model(medium, metric)
    for medium in mediums
    for metric in metrics
}
num_items = {
    m: pd.read_csv(f"{datadir}/{n}.csv").matchedid.max() + 1
    for m, n in [(0, "manga"), (1, "anime")]
}
planned_status = 3
baselines = get_baselines(device)
request_buffer = RequestBuffer(
    batch_size=256, timeout_seconds=0.001, max_queue_size=1024
)
with open(f"{datadir}/latest") as f:
    version = f.read()
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
