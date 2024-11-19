import logging
import time

import pandas as pd

from . import api_setup
from .api_setup import get_api_version, sanitize_string, to_unix_time

KITSU_TOKEN = None
KITSU_TOKEN_EXPIRY = -1


def make_session(proxies, concurrency):
    return api_setup.ProxySession(
        proxies, ratelimit_calls=concurrency, ratelimit_period=4 * concurrency
    )


def get_token(session):
    global KITSU_TOKEN
    global KITSU_TOKEN_EXPIRY
    refresh_token = (KITSU_TOKEN is None) or (time.time() >= KITSU_TOKEN_EXPIRY)
    if refresh_token:
        with open(
            api_setup.get_environment_path("kitsu/authentication/credentials.0.txt")
        ) as f:
            username, password = [x.strip() for x in f.readlines()]
        data = {
            "grant_type": "password",
            "username": username,
            "password": password,
        }
        response = api_setup.call_api(
            session, "POST", "https://kitsu.app/api/oauth/token", "kitsu", data=data
        )
        KITSU_TOKEN = response.json()
        KITSU_TOKEN_EXPIRY = time.time() + KITSU_TOKEN["expires_in"]
    return KITSU_TOKEN


def call_api(session, url):
    return api_setup.call_api(
        session,
        "GET",
        url,
        "kitsu",
        headers={"Authorization": f"Bearer {get_token(session)['access_token']}"},
    )


def get_mal_id(externalid):
    if "myanimelist" in externalid:
        externalid = externalid[len("https://myanimelist.net/anime/") :]
    uid = externalid.split("/")[0]
    if uid.isdigit():
        return uid
    else:
        # sometimes the external id field is incorrectly populated
        logging.info(f"could not parse external id {externalid}")
        return None


def get_mal_id_mapping(json, media):
    if "included" not in json:
        return {}
    mappings = json["included"]
    uid_to_mal_id = {}
    uid_to_mapping_ids = {}
    mapping_id_to_mal_id = {}
    for x in mappings:
        if x["type"] in [media, "libraryEntries"]:
            uid_to_mapping_ids[x["id"]] = [
                y["id"] for y in x["relationships"]["mappings"]["data"]
            ]
    for x in mappings:
        if (
            x["type"] == "mappings"
            and x["attributes"]["externalSite"] == f"myanimelist/{media}"
        ):
            mapping_id_to_mal_id[x["id"]] = get_mal_id(x["attributes"]["externalId"])
    for uid, map_ids in uid_to_mapping_ids.items():
        for map_id in map_ids:
            if map_id in mapping_id_to_mal_id:
                uid_to_mal_id[uid] = mapping_id_to_mal_id[map_id]
                break
    return uid_to_mal_id


def get_username(json):
    for x in json.get("included", []):
        if x["type"] == "users":
            return x["attributes"]["name"]
    return None


def process_rating(ratingTwenty, rating):
    if ratingTwenty is not None:
        return int(ratingTwenty) / 2
    if rating is not None:
        return float(rating) * 2
    return None


def process_timestamp(time):
    if time is None:
        return 0
    try:
        return to_unix_time(time, "%Y-%m-%dT%H:%M:%S.%fZ")
    except:
        return 0


def process_json(json, media):
    mal_mapping = get_mal_id_mapping(json, media)
    records = [
        (
            x["relationships"][media]["data"]["id"],
            mal_mapping.get(x["relationships"][media]["data"]["id"], None),
            process_rating(x["attributes"]["ratingTwenty"], x["attributes"]["rating"]),
            x["attributes"]["status"],
            x["attributes"]["progress"],
            x["attributes"].get("volumesOwned", ""),
            x["attributes"]["reconsuming"],
            x["attributes"]["reconsumeCount"],
            sanitize_string(x["attributes"]["notes"]),
            x["attributes"]["private"],
            x["attributes"]["reactionSkipped"],
            process_timestamp(x["attributes"]["progressedAt"]),
            process_timestamp(x["attributes"]["updatedAt"]),
            process_timestamp(x["attributes"]["createdAt"]),
            process_timestamp(x["attributes"]["startedAt"]),
            process_timestamp(x["attributes"]["finishedAt"]),
        )
        for x in json["data"]
    ]
    df = pd.DataFrame.from_records(
        records,
        columns=[
            "kitsuid",
            "malid",
            "score",
            "status",
            "progress",
            "volumes_owned",
            "repeat",
            "repeat_count",
            "notes",
            "private",
            "reaction_skipped",
            "progressed_at",
            "updated_at",
            "created_at",
            "started_at",
            "finished_at",
        ],
    )
    df["kitsuid"] = df["kitsuid"].astype(int)
    return df


def get_user_media_list(session, userid, media):
    has_next_chunk = True
    media_lists = []
    url = (
        f"https://kitsu.app/api/edge/library-entries?filter[user_id]={userid}"
        f"&filter[kind]={media}&include={media}.mappings,user&fields[{media}]=mappings"
        f"&fields[mappings]=externalSite,externalId&fields[users]=name&page[limit]=500"
    )
    username = None
    while has_next_chunk:
        response = call_api(session, url)
        if response.status_code in [403, 404, 525]:
            # 403: This can occur if the user privated their list
            # 404: This can occur if the user deleted their account
            # 525: can occur when authentication token is expired
            return pd.DataFrame(), False
        if not response.ok:
            logging.warning(f"Error {response} received when handling {url}")
            return pd.DataFrame(), False
        if "next" in response.json()["links"]:
            url = response.json()["links"]["next"]
        else:
            has_next_chunk = False
        media_lists.append(process_json(response.json(), media))
        username = get_username(response.json())

    media_list = pd.concat(media_lists)
    media_list["usertag"] = sanitize_string(username)
    media_list["api_version"] = get_api_version()
    media_list["username"] = str(userid)
    return media_list, True


def get_userid(session, username):
    url = f"https://kitsu.app/api/edge/users?filter[slug]={username}"
    response = call_api(session, url)
    json = response.json()["data"]
    if len(json) != 1:
        raise ValueError(f"there are {len(json)} users with slug {username}")
    return int(json[0]["id"])


def getstr(d, k):
    if k not in d or d[k] is None:
        return ""
    return str(d[k])


def get_media_title(titles, slug):
    for t in ["en_jp", "en_kr"]:
        if titles.get(t, None) is not None:
            return titles[t]
    english_names = [
        x
        for x in titles.keys()
        if x.startswith("en")
        and titles[x] is not None
        and x not in ["en", "en_us", "en_jp", "en_kr"]
    ]
    if len(english_names) > 1:
        logging.warning(f"Found multitples title for {titles}")
        return titles[english_names[0]]
    elif len(english_names) == 1:
        return titles[english_names[0]]
    else:
        for t in ["en", "en_us"]:
            if titles.get(t, None) is not None:
                return titles[t]
        logging.warning(f"Could not find any title for {titles}")
        return slug


def get_media_alttitle(titles):
    if titles.get("en", None) is not None:
        return titles["en"]
    if titles.get("en_us", None) is not None:
        return titles["en_us"]
    return ""


def get_media_facts(session, uid, medium):
    response = call_api(
        session,
        f"https://kitsu.app/api/edge/{medium}/{uid}?include=genres,mediaRelationships.destination",
    )
    if not response.ok:
        return pd.DataFrame(), pd.DataFrame()
    j = response.json()["data"]
    details = pd.DataFrame.from_records(
        [
            (
                j["id"],
                get_media_title(j["attributes"]["titles"], j["attributes"]["slug"]),
                get_media_alttitle(j["attributes"]["titles"]),
                getstr(j["attributes"], "subtype"),
                getstr(j["attributes"], "synopsis"),
                getstr(j["attributes"], "startDate"),
                getstr(j["attributes"], "endDate"),
                getstr(j["attributes"], "episodeCount"),
                getstr(j["attributes"], "episodeLength"),
                getstr(j["attributes"], "chapterCount"),
                getstr(j["attributes"], "volumeCount"),
                getstr(j["attributes"], "status"),
                str(
                    [
                        x["attributes"]["name"]
                        for x in response.json().get("included", [])
                        if x["type"] == "genres"
                    ]
                ),
                get_api_version(),
            )
        ],
        columns=[
            "kitsuid",
            "title",
            "alttitle",
            "type",
            "summary",
            "startdate",
            "enddate",
            "episodes",
            "duration",
            "chapters",
            "volumes",
            "status",
            "genres",
            "api_version",
        ],
    )

    records = []
    for x in [
        x
        for x in response.json().get("included", [])
        if x["type"] == "mediaRelationships"
    ]:
        records.append(
            (
                x["attributes"]["role"],
                str(uid),
                medium,
                x["relationships"]["destination"]["data"]["id"],
                x["relationships"]["destination"]["data"]["type"],
                get_api_version(),
            )
        )
    relations = pd.DataFrame.from_records(
        records,
        columns=[
            "relation",
            "source_id",
            "source_media",
            "target_id",
            "target_media",
            "api_version",
        ],
    )
    return details, relations