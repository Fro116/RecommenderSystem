from functools import cache

import pandas as pd

INPUT_HEADER = [
    "title",
    "score",
    "status",
    "progress",
    "updated_at",
    "item_order",
    "username",
]

TEXT_FIELDS = []


def process_status(status):
    animeplanet_status_map = {
        "1": "completed",
        "2": "currently_watching",
        "3": "dropped",
        "4": "planned",
        "5": "on_hold",
        "6": "wont_watch",
    }
    return STATUS_MAP[animeplanet_status_map[status]]


@cache
def get_title_mapping(medium):
    return (
        pd.read_csv(f"../../data/processed_data/animeplanet_{medium}_to_uid.csv")
        .set_index("title")[f"{medium}_id"]
        .to_dict()
    )


def process_title(title, medium):
    return get_title_mapping(medium).get(title, -1)


def parse_fields(line, medium, metadata):
    fields = line.split(",")
    get = lambda x: fields[INPUT_HEADER.index(x)]
    mediaid = process_title(get("title"), medium)
    return {
        "source": SOURCE_MAP[SOURCE],
        "medium": MEDIUM_MAP[medium],
        "userid": f"{SOURCE}@{get('username')}",
        "mediaid": mediaid,
        "status": process_status(get("status")),
        "rating": process_score(get("score")),
        "updated_at": parse_int(get("updated_at")),
        "created_at": 0,
        "started_at": 0,
        "finished_at": 0,
        "update_order": parse_int(get("item_order")),
        "progress": get_progress(medium, mediaid, parse_int(get("progress")), 0),
        "repeat_count": 0,
        "priority": 0,
        "sentiment": 0,
        "sentiment_score": 0,
        "owned": 0,
    }