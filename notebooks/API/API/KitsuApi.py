import datetime

API_PERIOD = 2
exec(open("ApiSetup.py").read())

KITSU_TOKEN = None
KITSU_TOKEN_EXPIRY = -1


def get_token():
    global KITSU_TOKEN
    global KITSU_TOKEN_EXPIRY
    refresh_token = (KITSU_TOKEN is None) or (time.time() >= KITSU_TOKEN_EXPIRY)
    if refresh_token:
        with open(
            get_datapath("../environment/kitsu/authentication/credentials.0.txt")
        ) as f:
            username, password = [x.strip() for x in f.readlines()]
        data = {
            "grant_type": "password",
            "username": username,
            "password": password,
        }
        response = call_api_internal(
            "https://kitsu.io/api/oauth/token", "POST", "kitsu", data=data
        )
        KITSU_TOKEN = response.json()
        KITSU_TOKEN_EXPIRY = time.time() + KITSU_TOKEN["expires_in"]
    return KITSU_TOKEN


def call_api(url):
    return call_api_internal(
        url,
        "GET",
        "kitsu",
        headers={"Authorization": f"Bearer {get_token()['access_token']}"},
    )


def get_mal_id(externalid):
    if "myanimelist" in externalid:
        externalid = externalid[len("https://myanimelist.net/anime/") :]
    uid = externalid.split("/")[0]
    if uid.isdigit():
        return uid
    else:
        # sometimes the external id field is incorrectly populated
        logger.info(f"could not parse external id {externalid}")
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
            "uid",
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
    df = df.loc[lambda x: ~x["uid"].isna()].copy()
    df["uid"] = df["uid"].astype(int)
    return df


def get_user_media_list(userid, media):
    has_next_chunk = True
    media_lists = []
    url = (
        f"https://kitsu.io/api/edge/library-entries?filter[user_id]={userid}"
        f"&filter[kind]={media}&include={media}.mappings,user&fields[{media}]=mappings"
        f"&fields[mappings]=externalSite,externalId&fields[users]=name&page[limit]=500"
    )
    username = None
    while has_next_chunk:
        response = call_api(url)
        if response.status_code in [403, 404, 525]:
            # 403: This can occur if the user privated their list
            # 404: This can occur if the user deleted their account
            # 525: can occur when authentication token is expired
            return pd.DataFrame(), False
        if not response.ok:
            logger.warning(f"Error {response} received when handling {url}")
            return pd.DataFrame(), False
        if "next" in response.json()["links"]:
            url = response.json()["links"]["next"]
        else:
            has_next_chunk = False
        media_lists.append(process_json(response.json(), media))
        username = get_username(response.json())

    media_list = pd.concat(media_lists)
    media_list["usertag"] = sanitize_string(username)
    media_list["username"] = str(userid)
    return media_list, True


def get_userid(username):
    url = f"https://kitsu.io/api/edge/users?filter[name]={username}"
    response = call_api(url)
    json = response.json()["data"]
    assert (
        len(json) == 1
    ), f"there are multiple users with username {username}, please use userid"
    return int(json[0]["id"])