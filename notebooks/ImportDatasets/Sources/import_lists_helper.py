import datetime
import glob
import logging
import os

import pandas as pd
from tqdm import tqdm

OUTPUT_HEADER = [
    "source",
    "medium",
    "userid",
    "mediaid",
    "status",
    "rating",
    "updated_at",
    "created_at",
    "started_at",
    "finished_at",
    "update_order",
    "progress",
    "repeat_count",
    "priority",
    "sentiment",
    "sentiment_score",
    "owned",
]

SOURCE_MAP = {"mal": 0, "anilist": 1, "kitsu": 2, "animeplanet": 3}

STATUS_MAP = {
    "rewatching": 7,
    "completed": 6,
    "currently_watching": 5,
    "on_hold": 4,
    "planned": 3,
    "dropped": 2,
    "wont_watch": 1,
    "none": 0,
}

MEDIUM_MAP = {"manga": 0, "anime": 1}

SENTIMENT_MAP = {
    "positive": 3,
    "neutral": 2,
    "negative": 1,
    "none": 0,
}


def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


def get_media_progress(medium):
    def to_dict(df, key, val):
        d = {}
        for i in range(len(df)):
            x = str(df[val][i])
            if x.isdigit():
                d[df[key][i]] = int(x)
        return d

    if os.path.exists(get_data_path(f"processed_data/{medium}.csv")):
        fn = get_data_path(f"processed_data/{medium}.csv")
    else:
        fn = get_data_path(f"raw_data/{medium}.csv")
    df = pd.read_csv(fn)
    if medium == "anime":
        return {"episodes": to_dict(df, f"{medium}_id", "num_episodes")}
    elif medium == "manga":
        return {
            "volumes": to_dict(df, f"{medium}_id", "num_volumes"),
            "chapters": to_dict(df, f"{medium}_id", "num_chapters"),
        }
    else:
        assert False


MEDIA_PROGRESS_MAP = {x: get_media_progress(x) for x in ["manga", "anime"]}


def parse_int(x, map={}, allow_neg=False):
    if x in map:
        return map[x]
    x = int(x)
    if not allow_neg:
        assert x >= 0
    return x


def filter_negative_ts(x):
    # the api can return a negative timestamp if users manually input a bogus date
    if "-" in x:
        return "0"
    return x


def get_completion(x, xmax):
    if xmax == 0:
        return 0.0
    else:
        return min(1.0, x / xmax)


def get_progress(medium, uid, progress, progress_volumes):
    df = MEDIA_PROGRESS_MAP[medium]
    if medium == "anime":
        return get_completion(progress, df["episodes"].get(uid, 0))
    elif medium == "manga":
        return max(
            get_completion(progress, df["chapters"].get(uid, 0)),
            get_completion(progress_volumes, df["volumes"].get(uid, 0)),
        )


def compute_sentiments(texts):
    sentiments = {}
    if not texts:
        return sentiments
    for x in texts:
        sentiments[x] = {
            "sentiment": "neutral",
            "score": 0,
        }
    sentiments[""] = {
        "sentiment": "none",
        "score": 0,
    }
    return sentiments


def process_score(score):
    score = float(score)
    if not (score >= 0 and score <= 10):
        logging.warning(f"invalid score {score}, replacing with 0")
        score = 0
    return score


def preprocess(input_fn, medium, header_fields, text_fields):
    logging.info(f"Sanitizing entries in {input_fn}")
    total_lines = 0
    total_texts = set()

    partition = input_fn.split(".")[-2]
    output_fn = input_fn + "~"
    with open(input_fn, "r") as in_file:
        with open(output_fn, "w") as out_file:
            header = False
            for line in tqdm(in_file):
                if not header:
                    header = True
                    correct_header = ",".join(header_fields) + "\n"
                    if line != correct_header:
                        logging.warning(
                            f"Replacing malformed header line {line.strip()} "
                            f"with correct header {correct_header.strip()}"
                        )
                        line = correct_header
                    out_file.write(line)
                    total_lines += 1
                    continue
                fields = line.strip().split(",")
                if len(fields) != len(header_fields):
                    logging.warning(
                        f"Deleting malformed line in user_{medium}_list.csv: {line} "
                    )
                    continue
                for tf in text_fields:
                    total_texts.add(fields[header_fields.index(tf)])
                out_file.write(line)
                total_lines += 1
        os.replace(output_fn, input_fn)
    return {
        "lines": total_lines,
        "texts": total_texts,
    }


def process(infile, outfile, medium, metadata, parsefn):
    logging.info(f"processing entries in {infile}")
    with open(infile, "r") as in_file:
        with open(outfile, "w") as out_file:
            header = False
            for line in tqdm(in_file, total=metadata["lines"]):
                if not header:
                    header = True
                    out_file.write(",".join(OUTPUT_HEADER) + "\n")
                    continue
                try:
                    fields = parsefn(line.strip(), medium, metadata)
                except Exception as e:
                    logging.error(f"Error: could not parse {line}")
                    raise e
                assert len(fields) == len(OUTPUT_HEADER)
                out_file.write(",".join(str(fields[x]) for x in OUTPUT_HEADER) + "\n")
