API_PERIOD = 4
exec(open("ApiSetup.py").read())


def call_api(url, json):
    return call_api_internal(url, "POST", "anilist", json=json)


def sanitize_date(x):
    def get(x, key, default):
        y = x.get(key, default)
        if y is None:
            return default
        return y

    return f'{get(x, "year", "")}-{get(x, "month", "")}-{get(x, "date", "")}'


def process_json(json):
    records = [
        (
            entry["media"]["idMal"],
            entry["score"],
            entry["status"],
            entry["progress"],
            entry["progressVolumes"],
            entry["repeat"],
            entry["priority"],
            sanitize_string(entry["notes"]),
            sanitize_date(entry["startedAt"]),
            sanitize_date(entry["completedAt"]),
            entry["updatedAt"],
            entry["createdAt"],
        )
        for x in json["data"]["MediaListCollection"]["lists"]
        for entry in x["entries"]
    ]
    df = pd.DataFrame.from_records(
        records,
        columns=[
            "uid",
            "score",
            "status",
            "progress",
            "progress_volumes",
            "repeat",
            "priority",
            "notes",
            "started_at",
            "completed_at",
            "updated_at",
            "created_at",
        ],
    )
    df = df.loc[lambda x: ~x["uid"].isna()].copy()
    df["uid"] = df["uid"].astype(int)
    return df


def get_user_media_list(userid, mediatype):
    listtype = mediatype.upper()
    has_next_chunk = True
    chunk = 0
    media_lists = []
    while has_next_chunk:
        query = """
        query ($userID: Int, $MEDIA: MediaType, $chunk: Int) {
            MediaListCollection (userId: $userID, type: $MEDIA, chunk: $chunk) {
                hasNextChunk
                lists {
                    entries
                    {
                        status
                        score(format: POINT_10_DECIMAL)
                        progress
                        progressVolumes
                        repeat
                        priority
                        notes
                        startedAt {
                            year
                            month
                            day
                        }
                        completedAt {
                            year
                            month
                            day
                        }
                        updatedAt
                        createdAt
                        media
                        {
                            idMal
                        }
                    }
                }
            }
        }
        """
        variables = {"userID": str(userid), "MEDIA": listtype, "chunk": chunk}
        url = "https://graphql.anilist.co"
        response = call_api(url, {"query": query, "variables": variables})
        if response.status_code in [403, 404]:
            # 403: This can occur if the user privated their list
            # 404: This can occur if the user deleted their account
            return pd.DataFrame(), False
        if not response.ok:
            logger.warning(f"Error {response} received when handling {url}")
            return pd.DataFrame(), False
        has_next_chunk = response.json()["data"]["MediaListCollection"]["hasNextChunk"]
        media_lists.append(process_json(response.json()))
        chunk += 1
    media_list = pd.concat(media_lists)
    # deduplicate shows that appear on multiple lists
    media_list = (
        media_list.sort_values(by=["updated_at", "created_at"])
        .groupby("uid")
        .last()
        .reset_index()
    )
    media_list["username"] = f"{userid}"
    return media_list, True


def get_userid(username):
    url = "https://graphql.anilist.co"
    query = "query ($username: String) { User (name: $username) { id } }"
    variables = {"username": str(username)}
    response = call_api(url, {"query": query, "variables": variables})
    try:
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Received error {str(e)} while accessing {url}")
        return f"{userid}"
    return response.json()["data"]["User"]["id"]


def get_username(userid):
    url = "https://graphql.anilist.co"
    query = "query ($userid: Int) { User (id: $userid) { name } }"
    variables = {"userid": str(userid)}
    response = call_api(url, {"query": query, "variables": variables})
    try:
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Received error {str(e)} while accessing {url}")
        return f"{userid}"
    return response.json()["data"]["User"]["name"]