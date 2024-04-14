from . import api_setup
import pandas as pd
from .api_setup import sanitize_string


def make_session(proxies, concurrency):
    return api_setup.ProxySession(
        proxies, ratelimit_calls=concurrency, ratelimit_period=4 * concurrency
    )


def call_api(session, url, json):
    return api_setup.call_api(session, "POST", url, "anilist", json=json)


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
            entry["media"]["id"],            
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
            "anilistid",
            "malid",
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
    df["anilistid"] = df["anilistid"].astype(int)
    return df


def get_user_media_list(session, userid, mediatype):
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
                            id
                            idMal
                        }
                    }
                }
            }
        }
        """
        variables = {"userID": str(userid), "MEDIA": listtype, "chunk": chunk}
        url = "https://graphql.anilist.co"
        response = call_api(session, url, {"query": query, "variables": variables})
        if response.status_code in [403, 404]:
            # 403: This can occur if the user privated their list
            # 404: This can occur if the user deleted their account
            return pd.DataFrame(), False
        if not response.ok:
            logging.warning(f"Error {response} received when handling {url}")
            return pd.DataFrame(), False
        has_next_chunk = response.json()["data"]["MediaListCollection"]["hasNextChunk"]
        media_lists.append(process_json(response.json()))
        chunk += 1
    media_list = pd.concat(media_lists)
    # deduplicate shows that appear on multiple lists
    media_list = (
        media_list.sort_values(by=["updated_at", "created_at"])
        .groupby("anilistid")
        .last()
        .reset_index()
    )
    media_list["username"] = f"{userid}"
    return media_list, True


def get_userid(session, username):
    url = "https://graphql.anilist.co"
    query = "query ($username: String) { User (name: $username) { id } }"
    variables = {"username": str(username)}
    response = call_api(session, url, {"query": query, "variables": variables})
    try:
        response.raise_for_status()
    except Exception as e:
        logging.warning(f"Received error {str(e)} while accessing {url}")
        return f"{userid}"
    return response.json()["data"]["User"]["id"]


def get_username(session, userid):
    url = "https://graphql.anilist.co"
    query = "query ($userid: Int) { User (id: $userid) { name } }"
    variables = {"userid": str(userid)}
    response = call_api(session, url, {"query": query, "variables": variables})
    try:
        response.raise_for_status()
    except Exception as e:
        logging.warning(f"Received error {str(e)} while accessing {url}")
        return f"{userid}"
    return response.json()["data"]["User"]["name"]