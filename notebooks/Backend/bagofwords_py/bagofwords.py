import io
import json
import os
import warnings

import msgpack
import zstandard as zstd
from flask import Flask, Response, request

exec(open("notebooks/Train/BagOfWords/bagofwords.py").read())
app = Flask(__name__)


def pack(data):
    return Response(
        zstd.ZstdCompressor().compress(msgpack.packb(data, use_single_float=True)),
        mimetype="application/msgpack",
        headers={"Content-Encoding": "zstd", "Content-Type": "application/msgpack"},
    )


def unpack(response):
    reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(response.data))
    return msgpack.unpackb(reader.read())


def compute(model, data):
    x = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        y = model(x, labels=None, weights=None, mask=False, mode="inference")
    return list(y.numpy().astype(float))


def load_model(medium, metric):
    fn = f"data/alphas/bagofwords/v1/streaming/{medium}/{metric}"
    if not os.path.exists(fn):
        return
    config = json.load(open(f"{fn}/config.json", "r"))
    model = BagOfWordsModel(
        config["input_sizes"],
        config["output_index"] - 1,
        config["metric"],
    )
    model.load_state_dict(
        torch.load(
            f"{fn}/model.pt", weights_only=True, map_location=torch.device("cpu")
        )
    )
    model = torch.compile(model)
    model.eval()
    query = pack({"inputs": [0] * sum(config["input_sizes"]) * 2})
    data = unpack(query)["inputs"]
    compute(model, data)
    return model


MODELS = {
    (x, y): load_model(x, y)
    for x in ["manga", "anime"]
    for y in ["rating", "watch", "plantowatch", "drop"]
}


@app.route("/query", methods=["POST"])
def query():
    data = unpack(request)
    medium = request.args.get("medium", type=str)
    metric = request.args.get("metric", type=str)
    output = compute(MODELS[(medium, metric)], data["inputs"])
    return pack({f"bagofwords/v1/streaming/{medium}/{metric}": output})


@app.route("/ready")
def ready():
    return Response(status=200)


if __name__ == "__main__":
    app.run()