import contextlib
import logging
import os
import shutil
import sys
import time

import pandas as pd
import requests

sys.path.append("..")
from API import animeplanet_api, api_setup, mal_web_api

PROXIES = api_setup.load_proxies(PROXY_NUMBER, PARTITION, NUM_PARTITIONS)

if source == "mal":
    SESSION = mal_web_api.make_session(PROXIES, 1)
    get_username = lambda userid: mal_web_api.get_username(SESSION, userid)
elif source == "animeplanet":
    SESSION = animeplanet_api.make_session(PROXIES, 1)
    call_api = lambda url: animeplanet_api.call_api(SESSION, url)
else:
    assert False

data_path = f"../../../data/{source}/user_facts"
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
os.chdir(data_path)


def configure_logging(logfile):
    name = "get_media"
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for stream in [
        logging.handlers.RotatingFileHandler(
            logfile, "w+", maxBytes=1000000, backupCount=1
        ),
    ]:
        stream.setFormatter(formatter)
        logger.addHandler(stream)


configure_logging(f"{name}.log")


@contextlib.contextmanager
def atomic_overwrite(filename):
    temp = filename + "~"
    with open(temp, "w") as f:
        yield f
    os.replace(temp, filename)


def atomic_to_csv(collection, filename):
    with atomic_overwrite(filename) as f:
        pd.Series(collection).to_csv(f, header=False, index=False)


def should_save(reason, max_iters=3600):
    should_save = False
    if reason not in SAVE_REASONS:
        SAVE_REASONS[reason] = (0, 1)
    iterations_since_last_write, iterations_until_next_write = SAVE_REASONS[reason]
    iterations_since_last_write += 1
    if iterations_since_last_write >= iterations_until_next_write:
        iterations_since_last_write = 0
        iterations_until_next_write = min(2 * iterations_until_next_write, max_iters)
        should_save = True
        logging.info(
            f"Writing data for {reason}. Will next write data "
            f"after {iterations_until_next_write} iterations"
        )
    SAVE_REASONS[reason] = (iterations_since_last_write, iterations_until_next_write)
    return should_save


SAVE_REASONS = {}