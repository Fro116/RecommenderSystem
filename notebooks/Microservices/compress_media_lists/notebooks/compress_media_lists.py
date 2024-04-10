import os
import tempfile

import msgpack
import pandas as pd
import ProcessData.process_media_lists_helper as pml
from flask import Flask, abort, request, Response
from ImportDatasets.Sources import anilist, animeplanet, kitsu, mal

app = Flask(__name__)


def pack(data):
    return Response(
        msgpack.packb(data, use_single_float=True), mimetype="application/msgpack"
    )


def unpack(response):
    return msgpack.unpackb(response.data)


VALID_TITLES = {
    m: set(pd.read_csv(f"../data/processed_data/{m}.csv")[f"{m}_id"])
    for m in ["manga", "anime"]
}


def import_media_list(source, medium, path):
    if source == "mal":
        s = mal
    elif source == "anilist":
        s = anilist
    elif source == "kitsu":
        s = kitsu
    elif source == "animeplanet":
        s = animeplanet
    src = os.path.join(path, f"user_{medium}_list.{source}.csv")
    dst = os.path.join(path, f"user_{medium}_list.raw.csv")
    data = s.preprocess(src, medium, s.INPUT_HEADER, s.TEXT_FIELDS)
    data["sentiments"] = s.compute_sentiments(list(data["texts"]))
    s.process(src, dst, medium, data, s.parse_fields)
    df = pd.read_csv(dst)
    df = df.loc[lambda x: x["mediaid"].isin(VALID_TITLES[medium])]
    df.to_csv(dst, index=False)
    return dst


def process_media_list(source, medium, path):
    src = os.path.join(path, f"user_{medium}_list.raw.csv")
    dst = os.path.join(path, f"user_{medium}_list.processed.csv")
    userids = list(pd.read_csv(src)["userid"].unique())
    if len(userids) == 0:
        userid_map = {}
    elif len(userids) == 1:
        userid_map = {userids[0]: 0}
    else:
        assert False
    pml.process_media_list(src, dst, userid_map)
    return dst


def generate_media_list(fn):
    df = pd.read_csv(fn)
    df = df.sort_values(by=["update_order", "updated_at"]).reset_index(drop=True)
    df["unit"] = 1
    df["update_order"] = (
        df.groupby("userid", group_keys=False)["unit"]
        .apply(lambda x: x.cumsum()[::-1])
        .values
    )
    df = df.drop("unit", axis=1)
    df = df.rename({"mediaid": "itemid"}, axis=1)
    return df


@app.route("/query", methods=['POST'])
def query():
    source = request.args.get("source", type=str)
    medium = request.args.get("medium", type=str)
    data = unpack(request)
    df = pd.DataFrame.from_dict({x: data[x] for x in data["_columns"]})    
    with tempfile.TemporaryDirectory() as path:
        df.to_csv(os.path.join(path, f"user_{medium}_list.{source}.csv"), index=False)
        import_media_list(source, medium, path)
        fn = process_media_list(source, medium, path)
        df = generate_media_list(fn)
        return pack(df.to_dict("list"))


@app.route("/wake")
def wake():
    return pack({"success": True})


if __name__ == "__main__":
    app.run()