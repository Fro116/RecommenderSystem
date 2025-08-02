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
    max_ranking_items = 1024
    if task == "retrieval":
        modeltype = "masked"
        max_seq_len = max_user_len
    elif task == "ranking":
        modeltype = "causal"
        max_seq_len = max_user_len + max_ranking_items
    else:
        assert False
    d = {
        # prompt features
        "userid": np.zeros((len(users), max_seq_len), dtype=np.int32),
        "time": np.zeros((len(users), max_seq_len), dtype=np.float64),
        "rope_input_pos": np.zeros((len(users), max_seq_len), dtype=np.int32),
        "token_mask_ids": np.zeros((len(users), max_seq_len), dtype=np.int32),
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
    if modeltype == "causal":
        masked_dim = models[f"transformer.masked.{medium}"].config["embed_dim"]
        masked_embs = np.zeros((len(users), masked_dim), dtype=np.float32)
        masked_pos = np.zeros((len(users), max_ranking_items), dtype=np.int32)
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
            d["userid"][u, i] = userid
            d["time"][u, i] = x["history_max_ts"]
            if modeltype == "causal":
                d["rope_input_pos"][u, i] = i if i < len(items) else len(items)
                d["token_mask_ids"][u, i] = 0 if i < len(items) else i
            # item features
            d[f"{m}_matchedid"][u, i] = x["matchedid"]
            d[f"{m}_distinctid"][u, i] = x["distinctid"]
            d[f"{n}_matchedid"][u, i] = -1
            d[f"{n}_distinctid"][u, i] = -1
            # action features
            if modeltype == "causal" and i >= len(items) and len(items) > 0:
                d["status"][u, i] = items[-1]["status"]
                d["rating"][u, i] = items[-1]["rating"]
                d["progress"][u, i] = items[-1]["progress"]
            else:
                d["status"][u, i] = x["status"]
                d["rating"][u, i] = x["rating"]
                d["progress"][u, i] = x["progress"]
        if modeltype == "causal":
            masked_embs[u, :] = users[u]["embeds"][f"masked.{medium}"]
            masked_pos[u, :] = [x["matchedid"] for x in users[u]["test_items"]]
    e = {}
    with gpu_lock:
        for k in d:
            d[k] = torch.tensor(d[k]).to(device)
            torch._dynamo.mark_dynamic(d[k], 0)
        if modeltype == "causal":
            masked_embs = torch.tensor(masked_embs).to(device)
            masked_pos = torch.tensor(masked_pos).to(device)
        start_time = time.time()
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                ret = models[f"transformer.{modeltype}.{medium}"](d, task)
                if modeltype == "causal":
                    m = models[f"transformer.masked.{medium}"]
                    X = torch.cat(
                        (
                            masked_embs.unsqueeze(1).repeat(1, max_ranking_items, 1),
                            m.item_embedding.matchedid_embeddings[medium](masked_pos),
                        ),
                        dim=-1,
                    )
                    masked_ret = m.rating_head(X).squeeze(-1)
        duration = time.time() - start_time
        logger.debug(f"batch of {len(users)} completed in {duration} seconds")
        if modeltype == "causal":
            e_ret, e_rnk = ret
            e["retrieval"] = e_ret.to("cpu")
            e["ranking"] = e_rnk.to("cpu")
            e["masked_ranking"] = masked_ret.to("cpu")
        else:
            e["retrieval"] = ret.to("cpu")
        del ret
    if modeltype == "masked":
        ret = []
        for i, u in enumerate(users):
            d = {}
            N = min(len(project(u)), max_user_len - 1)
            d[f"masked.{medium}"] = e["retrieval"][i, N, :].tolist()
            ret.append(d)
        return ret
    elif modeltype == "causal":
        ret = []
        for i, u in enumerate(users):
            d = {}
            N = min(len(project(u)), max_user_len - 1)
            causal_idxs = []
            for j in range(len(u["test_items"])):
                causal_idxs.append(2 * (N + j) + 1)
            d[f"causal.ranking.{medium}"] = e["ranking"][i, causal_idxs, 0].tolist()
            d[f"causal.retrieval.{medium}"] = e["retrieval"][i, 2 * N, :].tolist()
            d[f"masked.ranking.{medium}"] = masked_ret[i, :].tolist()
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


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s › %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
logger.info("STARTUP BEGIN")
starttime = time.time()
device = "cuda"
datadir = "../../data/finetune"
gpu_lock = threading.Lock()
models = {}
for modeltype in ["causal", "masked"]:
    for medium in [0, 1]:
        logger.info(f"STARTUP LOADING {modeltype} {medium} MODEL")
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
        }
        if task == "ranking":
            max_ranking_items = 1024
            masked_dim = models[f"transformer.masked.{medium}"].config["embed_dim"]
            test_user["test_items"] = [
                make_masked_item(time.time())
            ] * max_ranking_items
            test_user["embeds"] = {f"masked.{medium}": np.random.rand(masked_dim)}
        for batch_size in range(1, 16 + 1):
            predict([test_user] * batch_size, task, medium)
logger.info(f"STARTUP END AFTER {time.time() - starttime}s")
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