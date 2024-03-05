from flask import Flask, jsonify, request

app = Flask(__name__)

import glob
import os

import pandas as pd
from tqdm import tqdm


def import_script(nb):
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(nb))
        script = os.path.basename(nb)
        exec(open(script).read(), globals())
    finally:
        os.chdir(cwd)


PROXY_NUMBER = 0
TOKEN_NUMBER = 0
API_CALL_MULT = 5

source = os.getenv("RSYS_LIST_SOURCE")
allowed_sources = ["mal", "anilist", "kitsu", "animeplanet", "training"]
assert source in allowed_sources
if source != "training":
    import_script(f"../API/API/{source.capitalize()}Api.py")


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


def import_from_training(username, medium):
    s, usertag = username.split("@")
    files = glob.glob(f"../../data/{s}/user_media_facts/user_status.*.csv")
    fn = None
    for f in tqdm(files):
        df = pd.read_csv(f, dtype={"username": str}).query(f"username == '{usertag}'")
        if not df.empty:
            fn = f
            break
    assert fn is not None, f"could not find {username}"
    media_fn = f.replace("user_status", f"user_{medium}_list")
    df = pd.read_csv(media_fn, dtype={"username": str}).query(
        f"username == '{usertag}'"
    )
    return df


def import_list(username, medium, source):
    if source == "training":
        return import_from_training(username, medium)
    else:
        return import_from_api(username, medium, source)


def save_path(username, medium, source):
    if source == "training":
        source, username = username.split("@")
    data_path = os.path.join("../../data/recommendations", source, str(username))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return os.path.join(data_path, f"user_{medium}_list.{source}.csv")


@app.route("/query", methods=["GET"])
def query():
    username = request.args.get("username", type=str)
    medium = request.args.get("medium", type=str)
    import_list(username, medium, source).to_csv(
        save_path(username, medium, source), index=False
    )
    return jsonify({"result": "Success!"})


if __name__ == "__main__":
    app.run()