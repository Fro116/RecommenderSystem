import datetime
import json
import logging
import os
import random
import time
import uuid
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
        with open(geofn) as f:
            for line in f:
                ip, cc = line.strip().split(",")
                if cc in country_codes:
                    valid_ips.add(ip)

    proxies = []
    proxyfn = get_environment_path("proxies/proxies.txt")
    if os.path.exists(proxyfn):
        with open(proxyfn) as f:
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


def load_scp_key():
    fn = get_environment_path("scrapfly/key.txt")
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        lines = f.readlines()
        assert len(lines) == 1
        return lines[0].strip()


@cache
def get_api_version():
    fn = get_environment_path("../notebooks/Collect/version")
    with open(fn) as f:
        version = f.readlines()[0]
        return version.strip()


class Response:
    def __init__(self, text, status_code, headers):
        self.status_code = status_code
        self.text = text
        self.headers = {k.lower(): v for (k, v) in headers.items()}

    @property
    def ok(self):
        return self.status_code == 200

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError
        return self

    def __repr__(self):
        return f"Response{(self.status_code, len(self.text))}"


class RatelimitedSession:
    def __init__(self, ratelimit_calls, ratelimit_period):
        self.ratelimit_calls = ratelimit_calls
        self.ratelimit_period = ratelimit_period
        try:
            self.ratelimit_period *= int(RATELIMIT_MULT)
        except:
            pass
        self.request_times = []

    def ratelimit(self):
        t = time.time()
        recent_calls = [x for x in self.request_times if t - x < self.ratelimit_period]
        assert len(recent_calls) <= self.ratelimit_calls
        if len(recent_calls) == self.ratelimit_calls:
            time.sleep(recent_calls[0] + self.ratelimit_period - t)
            recent_calls = recent_calls[1:]
        recent_calls.append(time.time())
        self.request_times = recent_calls

    def multiply_ratelimit_period(self, mult):
        self.ratelimit_period *= mult

    def reset(self):
        pass


class ProxySession(RatelimitedSession):
    def __init__(self, proxies, ratelimit_calls, ratelimit_period):
        RatelimitedSession.__init__(self, ratelimit_calls, ratelimit_period)
        self.proxies = proxies.copy()
        self.session = None
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

    def call(self, request_type, url, **kwargs):
        self.ratelimit()
        if request_type == "POST":
            call = self.session.post
        elif request_type == "GET":
            call = self.session.get
        else:
            raise ValueError(f"Invalid request type {request_type}")
        r = call(url, impersonate="chrome", timeout=5, **kwargs)
        return Response(r.text, r.status_code, r.headers)


class ScrapflySession(RatelimitedSession):
    def __init__(self, key, ratelimit_calls, ratelimit_period):
        RatelimitedSession.__init__(self, ratelimit_calls, ratelimit_period)
        self.key = key
        self.reset()

    def call(self, request_type, url, **kwargs):
        assert request_type == "GET"
        assert not kwargs
        self.ratelimit()
        r = requests.get(
            url="https://api.scrapfly.io/scrape",
            params={
                "session": self.sessionid,
                "key": self.key,
                "proxy_pool": "public_datacenter_pool",
                "url": url,
                "country": "us",
            },
        )
        if not r.ok:
            return Response(r.text, r.status_code, r.headers)
        data = r.json()["result"]
        return Response(data["content"], data["status_code"], data["request_headers"])

    def reset(self):
        self.sessionid = str(uuid.uuid4())


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
            if "retry-after" in response.headers:
                retry_timeout = int(response.headers["retry-after"])
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