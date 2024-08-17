import datetime
import logging
import os
import random
import time
from functools import cache

import pandas as pd
from curl_cffi import requests


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
        logging.warning(f"Could not parse timestamp {date} {fmt} {time}")
        time = 0
    return time


def get_environment_path(path):
    datapath = os.getcwd()
    while datapath.split("/")[-1] not in ["notebooks", "data"]:
        datapath = "/".join(datapath.split("/")[:-1])
    datapath = "/".join(datapath.split("/")[:-1])
    return os.path.join(datapath, "environment", path)


def load_proxies(partition, num_partitions, country_codes=None):
    valid_ips = set()
    geofn = get_environment_path("proxies/geolocations.txt")
    if os.path.exists(geofn) and country_codes is not None:
        with open(geofn, "r") as f:
            for line in f:
                ip, cc = line.strip().split(",")
                if cc in country_codes:
                    valid_ips.add(ip)

    proxies = []
    proxyfn = get_environment_path("proxies/proxies.txt")
    if os.path.exists(proxyfn):
        with open(proxyfn, "r") as f:
            for line in f:
                proxyurl, port, username, password = line.strip().split(":")
                if valid_ips:
                    ip = username.split("-")[-1].split(":")[0]
                    if ip not in valid_ips:
                        continue
                url = f"http://{username}:{password}@{proxyurl}:{port}"
                proxies.append({domain: url for domain in ["http", "https"]})
    else:
        proxies = [None]

    same_part = [x for i, x in enumerate(proxies) if (i % num_partitions) == partition]
    diff_part = [x for i, x in enumerate(proxies) if (i % num_partitions) != partition]
    random.shuffle(same_part)
    random.shuffle(diff_part)
    return same_part + diff_part


@cache
def get_api_version():
    fn = get_environment_path("../notebooks/API/version")
    with open(fn) as f:
        version = f.readlines()[0]
        return version.strip()


class ProxySession:
    def __init__(self, proxies, ratelimit_calls, ratelimit_period):
        self.proxies = proxies
        self.session = None
        self.ratelimit_calls = ratelimit_calls
        self.ratelimit_period = ratelimit_period
        try:
            self.ratelimit_period *= int(RATELIMIT_MULT)
        except:
            pass
        self.request_times = []
        self.reset()

    def reset(self):
        x = self.proxies.pop(0)
        self.proxies.append(x)
        session = requests.Session()
        if x is not None:
            session.proxies.update(x)
        if self.session is not None:
            self.session.close()
        self.session = session

    def ratelimit(self):
        t = time.time()
        recent_calls = [x for x in self.request_times if t - x < self.ratelimit_period]
        assert len(recent_calls) <= self.ratelimit_calls
        if len(recent_calls) == self.ratelimit_calls:
            time.sleep(recent_calls[0] + self.ratelimit_period - t)
            recent_calls = recent_calls[1:]
        recent_calls.append(time.time())
        self.request_times = recent_calls

    def call(self, request_type, url, **kwargs):
        self.ratelimit()
        if request_type == "POST":
            call = self.session.post
        elif request_type == "GET":
            call = self.session.get
        else:
            raise ValueError(f"Invalid request type {request_type}")
        res = call(url, impersonate="chrome", timeout=5, **kwargs)
        return res


def call_api(
    session, request_type, url, source, retry_timeout=1, extra_error_codes=[], **kwargs
):
    max_timeout = 300
    response = None
    try:
        response = session.call(request_type, url, **kwargs)
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
            logging.error("Authentication token expired")
            return response
    except Exception as e:
        if response is not None:
            if "Retry-After" in response.headers:
                retry_timeout = int(response.headers["Retry-After"])
            logging.warning(f"Error {response} received when handling {url}")
        logging.warning(
            f"Received error '{str(e)}' while accessing {url}. Retrying in {retry_timeout} seconds"
        )
        time.sleep(retry_timeout)
        retry_timeout = min(retry_timeout * 2, max_timeout)
        session.reset()
        return call_api(
            session,
            request_type,
            url,
            source,
            retry_timeout,
            extra_error_codes=extra_error_codes,
            **kwargs,
        )
    return response