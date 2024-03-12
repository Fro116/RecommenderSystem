import datetime

import pandas as pd

from .import_lists_helper import *

SOURCE = "anilist"

INPUT_HEADER = [
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
    "username",
]

TEXT_FIELDS = ["notes"]

def process_status(status):
    anilist_status_map = {
        "REPEATING": "rewatching",
        "COMPLETED": "completed",
        "CURRENT": "currently_watching",
        "PLANNING": "planned",
        "PAUSED": "on_hold",
        "DROPPED": "dropped",
    }
    return STATUS_MAP[anilist_status_map[status]]


def parse_date(x):
    fields = x.split("-")
    if len(fields) != 3:
        return 0
    if not any(x != "" for x in fields):
        return 0
    year = parse_int(fields[0], {"": 1})
    month = parse_int(fields[1], {"": 1})
    date = parse_int(fields[2], {"": 1})
    try:
        dt = datetime.datetime(year, month, date)
        return int(dt.timestamp())
    except Exception as e:
        return 0


def parse_fields(line, medium, metadata):
    fields = line.split(",")
    get = lambda x: fields[INPUT_HEADER.index(x)]
    mediaid = parse_int(get("uid"))
    sentiment = metadata["sentiments"][get("notes")]
    return {
        "source": SOURCE_MAP[SOURCE],
        "medium": MEDIUM_MAP[medium],
        "userid": f"{SOURCE}@{get('username')}",
        "mediaid": mediaid,
        "status": process_status(get("status")),
        "rating": process_score(get("score")),
        "updated_at": parse_int(filter_negative_ts(get("updated_at"))),
        "created_at": parse_int(filter_negative_ts(get("created_at"))),
        "started_at": parse_date(get("started_at")),
        "finished_at": parse_date(get("completed_at")),
        "update_order": 0,
        "progress": get_progress(
            medium,
            mediaid,
            parse_int(get("progress")),
            parse_int(get("progress_volumes"), {"": 0}),
        ),
        "repeat_count": parse_int(get("repeat")),
        "priority": parse_int(get("priority")),
        "sentiment": SENTIMENT_MAP[sentiment["sentiment"]],
        "sentiment_score": sentiment["score"],
        "owned": 0,
    }