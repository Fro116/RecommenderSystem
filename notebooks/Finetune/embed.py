import asyncio
import contextlib
import functools
import gzip
import logging
import os
import signal
import threading
import time
import warnings

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


def make_item(ts, medium=0, itemid=-1):
    return {
        "medium": medium,
        "history_max_ts": ts,
        "matchedid": itemid,
        "status": -1,
        "rating": 0,
        "progress": 0,
    }


def tokenize(user_items):
    def span_to_token(x):
        token = x[0].copy()
        for k in ["status", "rating", "progress"]:
            token[k] = x[-1][k]
        return token
    items = []
    last_mid = None
    span = []
    for x in user_items:
        mid = (x["medium"], x["matchedid"])
        if mid == last_mid:
            span.append(x)
        else:
            if span:
                items.append(span_to_token(span))
            span = [x]
            last_mid = mid
    if span:
        items.append(span_to_token(span))
    return items


def project(user_items):
    items = []
    for x in user_items:
        if (x["history_status"] == x["status"]) and (
            x["history_rating"] == x["rating"]
        ):
            continue
        items.append(x)
    return items


def predict(users, task, medium):
    max_user_len = 1024
    max_ranking_items = 1024
    if task == "retrieval":
        max_seq_len = max_user_len
    elif task == "ranking":
        max_seq_len = max_user_len + max_ranking_items
    else:
        assert False
    num_items_0 = models[f"{medium}.{task}"].config["vocab_sizes"]["0_matchedid"]
    d = {
        # prompt features
        "userid": np.zeros((len(users), max_seq_len), dtype=np.int32),
        "time": np.zeros((len(users), max_seq_len), dtype=np.float64),
        "rope_input_pos": np.zeros((len(users), max_seq_len), dtype=np.int32),
        "token_mask_ids": np.zeros((len(users), max_seq_len), dtype=np.int32),
        "gender": np.zeros((len(users), max_seq_len), dtype=np.int32),
        "source": np.zeros((len(users), max_seq_len), dtype=np.int32),
        # item features
        "matchedid": np.zeros((len(users), max_seq_len), dtype=np.int32),
        # action features
        "status": np.zeros((len(users), max_seq_len), dtype=np.int32),
        "rating": np.zeros((len(users), max_seq_len), dtype=np.float32),
        "progress": np.zeros((len(users), max_seq_len), dtype=np.float32),
    }
    for u in range(len(users)):
        userid = u + 1
        user = users[u]["user"]
        items = project(tokenize(users[u]["items"]))
        extra_tokens = 1
        if len(items) > max_user_len - extra_tokens:
            items = items[-(max_user_len - extra_tokens) :]
        if task == "ranking":
            test_items = [make_item(users[u]["timestamp"], medium, x) for x in users[u]["ranking_items"]]
        elif task == "retrieval":
            test_items = [make_item(users[u]["timestamp"])]
        else:
            assert False
        for i, x in enumerate(items + test_items):
            m = x["medium"]
            # prompt features
            d["userid"][u, i] = userid
            d["time"][u, i] = x["history_max_ts"]
            d["gender"][u, i] = 0 if user["gender"] is None else user["gender"] + 1
            d["source"][u, i] = user["source"]
            d["rope_input_pos"][u, i] = i if i < len(items) else len(items)
            d["token_mask_ids"][u, i] = i if task == "ranking" and i >= len(items) else 0
            # item features
            d["matchedid"][u, i] = x["matchedid"] + (num_items_0 if m == 1 else 0)
            # action features
            d["status"][u, i] = x["status"]
            d["rating"][u, i] = x["rating"]
            d["progress"][u, i] = x["progress"]
    with gpu_lock:
        for k in d:
            d[k] = torch.tensor(d[k]).to(device)
            torch._dynamo.mark_dynamic(d[k], 0)
        start_time = time.time()
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                embs = models[f"{medium}.{task}"](d, task).to("cpu")
        duration = time.time() - start_time
        logger.debug(f"batch of {len(users)} completed in {duration} seconds")
    if task == "retrieval":
        ret = []
        for i, u in enumerate(users):
            d = {}
            N = min(len(project(tokenize(u["items"]))), max_user_len - 1)
            d[f"{medium}.{task}"] = embs[i, 2 * N, :].tolist()
            ret.append(d)
        return ret
    elif task == "ranking":
        ret = []
        for i, u in enumerate(users):
            d = {}
            N = min(len(project(tokenize(u["items"]))), max_user_len - 1)
            idxs = [2 * (N + j) + 1 for j in range(len(u["ranking_items"]))]
            d[f"{medium}.{task}"] = embs[i, idxs, 0].tolist()
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


def load_model(medium, task):
    datadir = "../../data/finetune"
    mtask = {"retrieval": "watch", "ranking": "rating"}[task]
    checkpoint = torch.load(
        f"{datadir}/transformer.masked.{medium}.{mtask}.finetune.pt",
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


def get_logger():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s › %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_models():
    starttime = time.time()
    logger.info("GET MODELS BEGIN")
    models = {}
    for medium in [0, 1]:
        for task in ["retrieval", "ranking"]:
            logger.info(f"GET MODELS LOADING {medium} {task}")
            srcmodel = "0.retrieval"
            tgtmodel = f"{medium}.{task}"
            models[tgtmodel] = load_model(medium, task)
            # deduplicate shared params from LoRA
            if tgtmodel != srcmodel:
                for k, _ in models[srcmodel].named_parameters():
                    if (
                        get_nested_attr(models[srcmodel], k)
                        == get_nested_attr(models[tgtmodel], k)
                    ).all():
                        set_nested_attr(
                            models[tgtmodel],
                            k,
                            get_nested_attr(models[srcmodel], k),
                        )
                torch.cuda.empty_cache()
    logger.info(f"GET MODELS END AFTER {time.time() - starttime}s")
    return models


def warmup(models):
    starttime = time.time()
    logger.info("WARMUP BEGIN")
    for task in ["retrieval", "ranking"]:
        for medium in [0, 1]:
            test_user = {
                "user": {"source": "mal", "birthday": None, "created_at": None, "gender": None, "source": 0},
                "items": [],
                "timestamp": time.time(),
            }
            if task == "ranking":
                max_ranking_items = 1024
                test_user["ranking_items"] = list(range(max_ranking_items))
            for batch_size in range(1, 16 + 1):
                predict([test_user] * batch_size, task, medium)
    logger.info(f"WARMUP END AFTER {time.time() - starttime}s")


device = "cuda"
gpu_lock = threading.Lock()
logger = get_logger()
models = get_models()
warmup(models)
app = FastAPI()


@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    if request.url.path == "/ready":
        return await call_next(request)
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    path = request.url.path
    if request.url.query:
        path += f"?{request.url.query}"
    logger.debug(f"Request to {path} took {duration:.4f} seconds")
    return response


@app.get("/shutdown")
async def shutdown_endpoint():
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200)


@app.get("/ready")
async def ready():
    return Response(status_code=200)


@app.post("/embed")
async def embed(request: Request, medium: int, task: str):
    try:
        data = await request.body()
        data = gzip.decompress(data)
        data = msgpack.unpackb(data, strict_map_key=False)
        result = predict(data["users"], task, medium)
        rdata = msgpack.packb({"embeds": result}, use_single_float=True)
        headers = {
            "Content-Type": "application/msgpack",
            "Content-Length": str(len(rdata)),
        }
        return Response(status_code=200, headers=headers, content=rdata)
    except Exception as e:
        logger.error(f"Error during request: {e}")
        return Response(status_code=500)