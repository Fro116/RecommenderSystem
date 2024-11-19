from flask import Flask, Response, abort, request

app = Flask(__name__)
import csv
import glob
import hashlib
import io
import os

import msgpack
import pandas as pd
import zstandard as zstd
from Collect.API import anilist_api, animeplanet_api, api_setup, kitsu_api, mal_api

PROXIES = api_setup.load_proxies(0, 1, ["us"])
SCP_KEY = api_setup.load_scp_key()
mal_api.load_token(0)


def pack(data):
    return Response(
        zstd.ZstdCompressor().compress(msgpack.packb(data, use_single_float=True)),
        mimetype="application/msgpack",
        headers={"Content-Encoding": "zstd", "Content-Type": "application/msgpack"},
    )


def unpack(response):
    reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(response.data))
    return msgpack.unpackb(reader.read())


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


def fetch_media_list(username, source, medium, datatype):
    proxies = get_proxies(username, source, medium)
    if source == "mal":
        s = mal_api.make_session(proxies=PROXIES, concurrency=4)
        df, ret = mal_api.get_user_media_list(s, username, medium)
    elif source == "anilist":
        s = anilist_api.make_session(proxies=PROXIES, concurrency=4)
        usertag = anilist_api.get_userid(s, username)
        df, ret = anilist_api.get_user_media_list(s, usertag, medium)
    elif source == "kitsu":
        s = kitsu_api.make_session(proxies=PROXIES, concurrency=4)
        if username[0] == "#":
            usertag = int(username[1:])
        else:
            usertag = kitsu_api.get_userid(s, username)
        df, ret = kitsu_api.get_user_media_list(s, usertag, medium)
    elif source == "animeplanet":
        s = animeplanet_api.make_session(
            proxies=PROXIES, scp_key=SCP_KEY, concurrency=8
        )
        if datatype == "user_media_data":
            df, ret = animeplanet_api.get_user_media_data(s, username, medium)
        elif datatype == "feed_data":
            df, ret = animeplanet_api.get_feed_data(s, username, medium)
            df = pd.DataFrame(df.items(), columns=["url", "updated_at"])
        else:
            assert False
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
        datatype = request.args.get("datatype", default=None, type=str)
        assert (
            len(username) > 0
            and source in ["mal", "anilist", "kitsu", "animeplanet"]
            and medium in ["manga", "anime"]
        )
        df = fetch_media_list(username, source, medium, datatype)
        return pack(df.fillna("").astype(str).to_dict("list"))
    except Exception as e:
        # TODO handle kitsu users with multiple usernames
        # TODO show an error page
        print(e)
        abort(400)


@app.route("/heartbeat")
def heartbeat():
    return pack({"success": True})


if __name__ == "__main__":
    app.run()