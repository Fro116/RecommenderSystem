from .import_lists_helper import *

import datetime

SOURCE = "mal"

INPUT_HEADER = [
    "uid",
    "status",
    "score",
    "progress",
    "progress_volumes",
    "started_at",
    "completed_at",
    "priority",
    "repeat",
    "repeat_count",
    "repeat_value",
    "tags",
    "notes",
    "updated_at",
    "username",
]

TEXT_FIELDS = []


def process_status(status):
    mal_status_map = {
        "completed": "completed",
        "watching": "currently_watching",
        "plan_to_watch": "planned",
        "reading": "currently_watching",
        "plan_to_read": "planned",
        "on_hold": "on_hold",
        "dropped": "dropped",
        "": "none",
    }
    return STATUS_MAP[mal_status_map[status]]


def parse_date(x):
    if x == "":
        return 0
    year = 1
    if len(x) >= 4:
        year = int(x[:4])
    month = 1
    if len(x) >= 7:
        month = int(x[5:7])
    date = 1
    if len(x) >= 10:
        date = int(x[8:10])
    if x == "":
        return 0
    assert len(x) >= 4
    try:
        dt = datetime.datetime(year, month, date)
        return int(dt.timestamp())
    except Exception as e:
        return 0


def parse_fields(line, medium, metadata):
    fields = line.split(",")
    get = lambda x: fields[INPUT_HEADER.index(x)]
    mediaid = parse_int(get("uid"))
    return {
        "source": SOURCE_MAP[SOURCE],
        "medium": MEDIUM_MAP[medium],
        "userid": f"{SOURCE}@{get('username')}",
        "mediaid": mediaid,
        "status": process_status(get("status")),
        "rating": process_score(get("score")),
        "updated_at": parse_int(get("updated_at")),
        "created_at": 0,
        "started_at": parse_date(get("started_at")),
        "finished_at": parse_date(get("completed_at")),
        "update_order": 0,
        "progress": get_progress(
            medium,
            mediaid,
            parse_int(get("progress")),
            parse_int(get("progress_volumes"), {"": 0}),
        ),
        "repeat_count": 0,
        "priority": 0,
        "sentiment": 0,
        "sentiment_score": 0,
        "owned": 0,
    }