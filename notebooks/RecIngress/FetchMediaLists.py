from flask import Flask, jsonify, request

app = Flask(__name__)

import glob
import os
import shutil
import pandas as pd
from tqdm import tqdm
from functools import cache

source = os.getenv("RSYS_LIST_SOURCE")
allowed_sources = ["mal", "anilist", "kitsu", "animeplanet"]
assert source in allowed_sources

PROXY_NUMBER = 0
TOKEN_NUMBER = 0
API_CALL_MULT = 5
SOURCE = source
MEDIA_DIR = "../../data/processed_data"

def import_script(nb):
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(nb))
        script = os.path.basename(nb)
        exec(open(script).read(), globals())
    finally:
        os.chdir(cwd)
        
if source != "training":
    import_script(f"../API/API/{source.capitalize()}Api.py")
import_script("../ImportDatasets/ImportListsHelper.py")


def import_from_api(username, medium, source):
    pwd = os.getcwd()
    try:
        os.chdir("../API/API")
        if source == "anilist":
            if isinstance(username, str):
                username = get_userid(username)
        elif source == "kitsu":
            if isinstance(username, str):
                username = get_userid(username)
        df, ret = get_user_media_list(username, medium)
        if not ret:
            raise Exception(f"Could not resolve list for {username}")
    finally:
        os.chdir(pwd)
    return df


def import_from_training(username, medium, source):
    files = glob.glob(f"../../data/{source}/user_media_facts/user_status.*.csv")
    fn = None
    for f in tqdm(files):
        df = pd.read_csv(f, dtype={"username": str}).query(f"username == '{username}'")
        if not df.empty:
            fn = f
            break
    assert fn is not None, f"could not find {username}"
    media_fn = f.replace("user_status", f"user_{medium}_list")
    df = pd.read_csv(media_fn, dtype={"username": str}).query(
        f"username == '{username}'"
    )
    return df


def import_list(username, medium, source, training):
    if training:
        return import_from_training(username, medium, source)
    else:
        return import_from_api(username, medium, source)


def save_path(username, medium, source):
    data_path = os.path.join("../../data/recommendations", source, str(username))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return os.path.join(data_path, f"user_{medium}_list.{source}.csv")


@cache
def get_valid_titles(medium):
    return set(pd.read_csv(f"{MEDIA_DIR}/{medium}.csv")[f"{medium}_id"])


@app.route("/query", methods=["GET"])
def query():
    # fetch media list
    username = request.args.get("username", type=str)
    medium = request.args.get("medium", type=str)
    if "@" in username:
        s, username = username.split("@")
        assert s == source
        training = True
    else:
        training = False
    import_list(username, medium, source, training).to_csv(
        save_path(username, medium, source), index=False
    )       

    # import media list
    data_path = os.path.join("../../data/recommendations", source, username)
    src = os.path.join(data_path, f"user_{medium}_list.{source}.csv")
    dst = os.path.join(data_path, f"user_{medium}_list.raw.csv")
    data = preprocess(src, medium, INPUT_HEADER, TEXT_FIELDS)
    data["sentiments"] = compute_sentiments(list(data["texts"]))
    process(src, dst, medium, data)
    df = pd.read_csv(dst)
    valid_titles = get_valid_titles(medium)
    df = df.loc[lambda x: x["mediaid"].isin(valid_titles)]
    df.to_csv(dst, index=False)

    return jsonify({"result": "Success!"})


if __name__ == "__main__":
    app.run()