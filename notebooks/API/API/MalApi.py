import datetime

API_PERIOD = 8
exec(open("ApiSetup.py").read())

with open(
    get_datapath(f"../environment/mal/authentication/clientid.{TOKEN_NUMBER}.txt")
) as f:
    MAL_ACCESS_TOKEN = f.readlines()[0].strip()


def call_api(url):
    return call_api_internal(
        url,
        "GET",
        "mal",
        headers={"X-MAL-CLIENT-ID": MAL_ACCESS_TOKEN},
    )


def get_user_media_list(username, media):
    media_lists = []
    more_pages = True
    url = (
        "https://api.myanimelist.net/v2/users/"
        f"{username}/{media}list?limit=1000&fields=list_status&nsfw=true"
    )
    while more_pages:
        response = call_api(url)
        if response.status_code != 200 or "data" not in response.json():
            logger.warning(f"Error {response} received when handling {url}")
            return pd.DataFrame(), False

        json = response.json()
        media_lists.append(process_media_list_json(json, media))
        more_pages = "next" in json["paging"]
        if more_pages:
            url = json["paging"]["next"]
    user_media_list = pd.concat(media_lists, ignore_index=True)
    user_media_list["username"] = username
    return user_media_list, True


def process_media_list_json(json, media):
    entries = [parse_json_node(x, media) for x in json["data"]]
    if entries:
        return pd.concat(entries, ignore_index=True)
    else:
        return pd.DataFrame()


def parse_json_node(x, media):
    ls = x["list_status"]
    progress_col = {
        "anime": "num_episodes_watched",
        "manga": "num_chapters_read",
    }
    repeat_col = {
        "anime": "is_rewatching",
        "manga": "is_rereading",
    }
    repeat_count_col = {
        "anime": "num_times_rewatched",
        "manga": "num_times_reread",
    }
    repeat_value_col = {
        "anime": "rewatch_value",
        "manga": "reread_value",
    }
    entry = pd.DataFrame.from_dict(
        {
            "uid": [x["node"]["id"]],
            "status": [ls.get("status", "")],
            "score": [ls.get("score", "")],
            "progress": [ls.get(progress_col[media], "")],
            "progress_volumes": [ls.get("num_volumes_read", "")],
            "started_at": [ls.get("start_date", "")],
            "completed_at": [ls.get("finish_date", "")],
            "priority": [ls.get("priority", "")],
            "repeat": [ls.get(repeat_col[media], False)],
            "repeat_count": [ls.get(repeat_count_col[media], "")],
            "repeat_value": [ls.get(repeat_value_col[media], "")],
            "tags": [" ".join([sanitize_string(x) for x in ls.get("tags", [])])],
            "notes": [sanitize_string(ls.get("comments", ""))],
            "updated_at": [process_timestamp(ls.get("updated_at", None))],
        }
    )
    return entry


def process_timestamp(time):
    if time is None:
        return 0
    try:
        return to_unix_time(time, "%Y-%m-%dT%H:%M:%S+00:00")
    except:
        return 0