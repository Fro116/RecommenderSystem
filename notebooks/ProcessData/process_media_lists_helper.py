import logging
import os
import time

import pandas as pd
from tqdm import tqdm


def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


def load_timestamps():
    def parse_line(file, field, format=int):
        line = file.readline()
        fields = line.strip().split(",")
        assert len(fields) == 2
        assert fields[0] == field
        return format(fields[1])

    with open(os.path.join(get_data_path("processed_data/timestamps.csv"))) as f:
        min_timestamp = parse_line(f, "min_timestamp")
        max_timestamp = parse_line(f, "max_timestamp")
    return min_timestamp, max_timestamp


def load_uid_mapping(fn, col):
    return (
        pd.read_csv(get_data_path(f"processed_data/{fn}")).set_index(col)["uid"].to_dict()
    )


MIN_TIMESTAMP, MAX_TIMESTAMP = load_timestamps()

MEDIA_TO_UID = {
    "0": load_uid_mapping("manga_to_uid.csv", "mediaid"),
    "1": load_uid_mapping("anime_to_uid.csv", "mediaid"),
}


def format_timestamp(ts):
    ts = int(ts)
    # manually entered timestamps can be inaccurate
    if ts < MIN_TIMESTAMP:
        return 0
    if ts > time.time():
        return 0
    return (ts - MIN_TIMESTAMP) / (MAX_TIMESTAMP - MIN_TIMESTAMP)


def format_line(line, header, username_map):
    fields = line.strip().split(",")
    for f in ["updated_at", "created_at", "started_at", "finished_at"]:
        fields[header.index(f)] = str(format_timestamp(fields[header.index(f)]))
    fields[header.index("userid")] = str(username_map[fields[header.index("userid")]])
    medium = fields[header.index("medium")]
    a = int(fields[header.index("mediaid")])
    if a not in MEDIA_TO_UID[medium]:
        logging.warning(f"Item {a} not found")
        return None
    fields[header.index("mediaid")] = str(MEDIA_TO_UID[medium][a])
    return ",".join(fields) + "\n"


def process_media_list(source, dest, username_map):
    with open(source, "r") as in_file, open(dest, "w") as out_file:
        header = False
        for line in tqdm(in_file):
            if not header:
                header = True
                header_fields = line.strip().split(",")
                out_file.write(line)
                continue
            try:
                out = format_line(line, header_fields, username_map)
                if out is not None:
                    out_file.write(out)
            except Exception as e:
                logging.warning(line)
                logging.warning(str(e))
                raise e