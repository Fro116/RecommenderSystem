import re

from . import api_setup
import pandas as pd
from .api_setup import sanitize_string


def make_session(proxies, concurrency):
    return api_setup.ProxySession(
        proxies, ratelimit_calls=concurrency, ratelimit_period=4 * concurrency
    )


def call_api(session, url):
    header = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
    return api_setup.call_api(
        session,
        "GET",
        url,
        "web",
        extra_error_codes=[403],
        headers={"User-Agent": header},
    )


MATCHFIELD = """([^<>]+)"""
TITLE_REGEX = re.compile('<h3 class="cardName">' + MATCHFIELD + "</h3>")
SCORE_REGEX = re.compile('<div class="ttRating">' + MATCHFIELD + "</div>")
STATUS_REGEX = re.compile('<span class="status' + MATCHFIELD + '">')
PROGRESS_REGEXES = {
    x: re.compile("</span> " + MATCHFIELD + f" {y}</div>")
    for (x, y) in zip(["manga", "anime"], ["chs", "eps"])
}
PAGE_REGEX = re.compile('page=([0-9]*)">')
FEED_TITLE_REGEX = re.compile('">' + MATCHFIELD + f"</h5>")
FEED_TIMESTAMP_REGEX = re.compile('data-timestamp="' + MATCHFIELD + f'">')


def unpack(x):
    assert len(x) == 1
    return x[0]


def is_private_list(resp, username):
    return " has chosen to make their content private." in resp.text


def is_invalid_user(resp, username):
    return f"<title>Search Results for {username}" in resp.text


def get_title(x):
    return sanitize_string(unpack(TITLE_REGEX.findall(x)))


def get_score(x):
    scores = SCORE_REGEX.findall(x)
    if not scores:
        return "0"
    return str(2 * float(unpack(scores)))


def get_status(x):
    status = STATUS_REGEX.findall(x)
    return unpack(status)


def get_progress(x, medium):
    progress = PROGRESS_REGEXES[medium].findall(x)
    if not progress:
        return 0
    return unpack(progress)


def parse_entry(x, medium):
    return (get_title(x), get_score(x), get_status(x), get_progress(x, medium))


def get_page_entries(resp):
    return [x for x in resp.text.split("\n") if '<h3 class="cardName">' in x]


def get_page_numbers(resp):
    return set(PAGE_REGEX.findall(resp.text))


def get_feed_entries(resp):
    return [x for x in resp.text.split("\n") if "data-timestamp" in x]


def get_feed_title(x):
    return unpack(FEED_TITLE_REGEX.findall(x))


def get_feed_timestamp(x):
    return unpack(FEED_TIMESTAMP_REGEX.findall(x))


def get_feed_data(session, username, medium):
    feed_data = {}
    next_page = True
    page = 0
    while next_page and page < 15:
        page += 1
        url = (
            f"https://www.anime-planet.com/users/{username}"
            + f"/feed?type={medium}&page={page}"
        )
        resp = call_api(session, url)
        if not resp.ok:
            return {}, False
        next_page = False
        data = get_feed_entries(resp)
        if data:
            next_page = True
            for x in data:
                key = get_feed_title(x)
                if key not in feed_data:
                    feed_data[key] = get_feed_timestamp(x)
    return feed_data, True


def get_user_media_list(session, username, medium):
    page = 0
    next_page = True
    records = []
    while next_page:
        page += 1
        url = (
            f"https://www.anime-planet.com/users/{username}"
            + f"/{medium}?sort=user_updated&order=desc&per_page=560"
            + f"&page={page}"
        )
        resp = call_api(session, url)
        if (
            not resp.ok
            or is_private_list(resp, username)
            or is_invalid_user(resp, username)
        ):
            return pd.DataFrame(), False
        for x in get_page_entries(resp):
            records.append(parse_entry(x, medium))
        if str(page + 1) not in get_page_numbers(resp):
            next_page = False
    df = pd.DataFrame(
        data=list(reversed(records)), columns=["title", "score", "status", "progress"]
    )
    if len(df) > 0:
        feed_data, feed_ok = get_feed_data(session, username, medium)
        if not feed_ok:
            logging.info(f"Cannot parse feed for {username} {medium}")
    else:
        feed_data = {}
    df["updated_at"] = 0
    for i in range(len(df)):
        df.loc[i, "updated_at"] = int(feed_data.get(df.loc[i, "title"], 0))
    df["item_order"] = list(range(len(df)))
    df["username"] = username
    return df, True