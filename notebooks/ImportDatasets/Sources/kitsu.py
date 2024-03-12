from .import_lists_helper import *

SOURCE = "kitsu"

INPUT_HEADER = [
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
    "usertag",
    "username",
]

TEXT_FIELDS = ["notes"]


def process_status(status, rewatch):
    if rewatch == "True":
        return STATUS_MAP["rewatching"]
    kitsu_status_map = {
        "completed": "completed",
        "current": "currently_watching",
        "dropped": "dropped",
        "on_hold": "on_hold",
        "planned": "planned",
    }
    return STATUS_MAP[kitsu_status_map[status]]


def parse_fields(line, medium, metadata):
    fields = line.split(",")
    get = lambda x: fields[INPUT_HEADER.index(x)]
    mediaid = parse_int(get("uid"))
    sentiment = metadata["sentiments"][get("notes")]
    return {
        "source": SOURCE_MAP[SOURCE],
        "medium": MEDIUM_MAP[medium],
        "userid": f"{SOURCE}@{parse_int(get('username'))}",
        "mediaid": mediaid,
        "status": process_status(get("status"), get("repeat")),
        "rating": process_score(get("score")),
        "updated_at": parse_int(get("updated_at")),
        "created_at": parse_int(get("created_at")),
        "started_at": parse_int(filter_negative_ts(get("started_at"))),
        "finished_at": parse_int(filter_negative_ts(get("finished_at"))),
        "update_order": 0,
        "progress": get_progress(medium, mediaid, parse_int(get("progress")), 0),
        "repeat_count": parse_int(get("repeat_count")),
        "priority": 0,
        "sentiment": SENTIMENT_MAP[sentiment["sentiment"]],
        "sentiment_score": sentiment["score"],
        "owned": get_progress(medium, mediaid, 0, parse_int(get("volumes_owned"))),
    }