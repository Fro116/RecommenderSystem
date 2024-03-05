import logging
import os
import random
import time

import pandas as pd
import requests
from ratelimit import limits, sleep_and_retry

# Logging

try:
    logger
except NameError:
    logger = logging.getLogger("")

# Data cleaning


def sanitize_string(x):
    # to support string and newline parsing
    if x is None:
        return ""
    replacements = {
        "\n": " ",
        "\r": " ",
        ",": " ",
        "\x00": " ",
    }
    for k, v in replacements.items():
        x = x.replace(k, v)
    return x


def to_unix_time(date, fmt):
    time = int(datetime.datetime.timestamp(datetime.datetime.strptime(date, fmt)))
    if time < 0:
        logger.warning(f"Could not parse timestamp {date} {fmt} {time}")
        time = 0
    return time


# Proxies


def get_datapath(path):
    datapath = os.getcwd()
    while datapath.split("/")[-1] not in ["notebooks", "data"]:
        datapath = "/".join(datapath.split("/")[:-1])
    datapath = "/".join(datapath.split("/")[:-1])
    return os.path.join(datapath, "data", path)


def setup_proxy(index, fn):
    if not os.path.exists(fn):
        assert index == 0
        return None

    proxylist = []
    with open(fn) as f:
        for line in f:
            proxyurl, port, username, password = line.strip().split(":")
            proxylist.append(f"http://{username}:{password}@{proxyurl}:{port}")
    return {domain: proxylist[index] for domain in ["http", "https"]}


if PROXY_NUMBER == "SHARED":
    proxyfn = get_datapath("../environment/proxies/proxies.txt")
    with open(proxyfn, "r") as f:
        num_proxies = len(f.readlines())
    IN_PROXY = [
        setup_proxy(x, proxyfn)
        for x in range(num_proxies)
        if (x % NUM_PARTITIONS) == PARTITION
    ]
    OUT_PROXY = [
        setup_proxy(x, proxyfn)
        for x in range(num_proxies)
        if (x % NUM_PARTITIONS) != PARTITION
    ]
    random.shuffle(IN_PROXY)
    random.shuffle(OUT_PROXY)
    PROXY = IN_PROXY + OUT_PROXY
else:
    PROXY = [
        setup_proxy(
            int(PROXY_NUMBER),
            get_datapath("../environment/proxies/proxies.txt"),
        )
    ]


def reset_session():
    global SESSION
    global PROXY
    x = PROXY.pop(0)
    PROXY.append(x)
    SESSION = requests.Session()
    if x is not None:
        SESSION.proxies.update(x)


SESSION = None
reset_session()

# API endpoint

try:
    API_PERIOD_MULT
except NameError:
    API_PERIOD_MULT = 1

try:
    API_CALL_MULT
except NameError:
    API_CALL_MULT = 1


@sleep_and_retry
@limits(calls=API_CALL_MULT, period=API_PERIOD * API_PERIOD_MULT * API_CALL_MULT)
def call_api_internal(
    url, request_type, source, retry_timeout=1, extra_error_codes=[], **kwargs
):
    if request_type == "POST":
        request_call = SESSION.post
    elif request_type == "GET":
        request_call = SESSION.get
    else:
        raise ValueError(f"Invalid request type {request_type}")

    max_timeout = 300
    response = None
    try:
        response = request_call(url, timeout=5, **kwargs)
        if (
            response.status_code
            in [409, 429, 500, 502, 503, 504, 520, 530] + extra_error_codes
            and retry_timeout < max_timeout
        ):
            # transient errors
            raise Exception(f"{response.status_code}")
        if (
            source == "mal"
            and response.status_code == 200
            and "data" not in response.json()
            and retry_timeout < max_timeout
        ):
            # sometimes the response returns 200 but is empty
            raise Exception(f"{response.status_code}")
        if response.status_code in [401]:
            logger.error("Authentication token expired")
            return response
    except Exception as e:
        if response is not None:
            if "Retry-After" in response.headers:
                retry_timeout = int(response.headers["Retry-After"])
            logger.warning(f"Error {response} received when handling {url}")
        logger.warning(
            f"Received error '{str(e)}' while accessing {url}. Retrying in {retry_timeout} seconds"
        )
        time.sleep(retry_timeout)
        retry_timeout = min(retry_timeout * 2, max_timeout)
        reset_session()
        return call_api_internal(
            url,
            request_type,
            source,
            retry_timeout,
            extra_error_codes=extra_error_codes,
            **kwargs,
        )
    return response