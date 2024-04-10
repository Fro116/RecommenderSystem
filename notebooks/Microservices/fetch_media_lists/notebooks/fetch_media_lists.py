from flask import Flask, abort, request, Response

app = Flask(__name__)
import csv
import glob
import hashlib
import os

import msgpack
import pandas as pd
from API.API import anilist_api, animeplanet_api, api_setup, kitsu_api, mal_api


PROXIES = api_setup.load_proxies("SHARED", 0, 1, ["us"])
mal_api.load_token(0)


def pack(data):
    return Response(
        msgpack.packb(data, use_single_float=True), mimetype="application/msgpack"
    )


def get_proxies(username, source, medium):
    m = hashlib.sha256()
    m.update(username.encode("utf-8"))
    m.update(source.encode("utf-8"))
    m.update(medium.encode("utf-8"))
    idx = int(m.hexdigest(), 16) % len(PROXIES)
    proxies = [PROXIES[idx]]
    for i, x in enumerate(PROXIES):
        if i != idx:
            proxies.append(x)
    return proxies


def fetch_media_list(username, source, medium):
    proxies = get_proxies(username, source, medium)
    if source == "mal":
        s = mal_api.make_session(proxies, 4)
        df, ret = mal_api.get_user_media_list(s, username, medium)
    elif source == "anilist":
        s = anilist_api.make_session(proxies, 4)
        if username[0] == "#":
            usertag = int(username[1:])
        else:
            usertag = anilist_api.get_userid(s, username)
        df, ret = anilist_api.get_user_media_list(s, usertag, medium)
    elif source == "kitsu":
        s = kitsu_api.make_session(proxies, 1)
        if username[0] == "#":
            usertag = int(username[1:])
        else:
            usertag = kitsu_api.get_userid(s, username)
        df, ret = kitsu_api.get_user_media_list(s, usertag, medium)
    elif source == "animeplanet":
        s = animeplanet_api.make_session(proxies, 20)  # need burst to fetch feed
        df, ret = animeplanet_api.get_user_media_list(s, username, medium)
    else:
        assert False
    if not ret:
        raise Exception(f"Could not resolve {medium} list for {username} at {source}")
    return df


@app.route("/query")
def query():
    try:
        username = request.args.get("username", type=str)
        source = request.args.get("source", type=str)
        medium = request.args.get("medium", type=str)
        assert (
            len(username) > 0
            and source in ["mal", "anilist", "kitsu", "animeplanet"]
            and medium in ["manga", "anime"]
        )
    except:
        abort(400)  # TODO show an error page
    try:
        df = fetch_media_list(username, source, medium)
        d = df.to_dict("list")
        d["_columns"] = list(df.columns)
        return pack(d)
    except Exception as e:
        # TODO handle kitsu users with multiple usernames
        print(e)
        abort(400)  # TODO show an error page


@app.route("/wake")
def wake():
    return pack({"success": True})


if __name__ == "__main__":
    app.run()