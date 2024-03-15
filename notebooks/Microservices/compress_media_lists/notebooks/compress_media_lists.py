from flask import Flask, abort, jsonify, request

app = Flask(__name__)
import os
import tempfile

import pandas as pd
import ProcessData.process_media_lists_helper as pml
from ImportDatasets.Sources import anilist, animeplanet, kitsu, mal

VALID_TITLES = {
    m: set(pd.read_csv(f"../data/processed_data/{m}.csv")[f"{m}_id"])
    for m in ["manga", "anime"]
}


def import_media_list(username, source, medium, path):
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


def process_media_list(username, source, medium, path):
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
    df["forward_order"] = (
        df.groupby("userid", group_keys=False)["unit"]
        .apply(lambda x: x.cumsum())
        .values
    )
    df["backward_order"] = (
        df.groupby("userid", group_keys=False)["unit"]
        .apply(lambda x: x.cumsum()[::-1])
        .values
    )
    return df


@app.route("/query", methods=['POST'])
def query():
    username = request.args.get("username", type=str)
    source = request.args.get("source", type=str)
    medium = request.args.get("medium", type=str)
    data = request.get_json()
    df = pd.DataFrame.from_dict({x: data[x] for x in data["_columns"]})    
    with tempfile.TemporaryDirectory() as path:
        df.to_csv(os.path.join(path, f"user_{medium}_list.{source}.csv"), index=False)
        import_media_list(username, source, medium, path)
        fn = process_media_list(username, source, medium, path)
        df = generate_media_list(fn)
        return jsonify(df.to_dict("list"))


@app.route("/wake")
def wake():
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run()