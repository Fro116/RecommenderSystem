import contextlib
import logging
import os
import shutil
import time

import pandas as pd
import requests


def import_script(nb):
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(nb))
        script = os.path.basename(nb)
        exec(open(script).read(), globals())
    finally:
        os.chdir(cwd)


if source == "mal":
    API_PERIOD = 4
    import_script("../API/MalWebApi.py")
elif source == "animeplanet":
    import_script("../API/AnimeplanetApi.py")
else:
    assert False

data_path = f"../../../data/{source}/user_facts"
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
os.chdir(data_path)

logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
LOG_FILE = f"{name}.log"
for stream in [
    logging.handlers.RotatingFileHandler(
        LOG_FILE, "w+", maxBytes=1000000, backupCount=1
    ),
]:
    stream.setFormatter(formatter)
    logger.addHandler(stream)


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
        logger.info(
            f"Writing data for {reason}. Will next write data "
            f"after {iterations_until_next_write} iterations"
        )
    SAVE_REASONS[reason] = (iterations_since_last_write, iterations_until_next_write)
    return should_save


SAVE_REASONS = {}