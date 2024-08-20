import html
import logging
import re

import pandas as pd

from . import api_setup
from .api_setup import get_api_version, sanitize_string


def make_session(proxies, concurrency):
    return api_setup.ProxySession(
        proxies, ratelimit_calls=concurrency, ratelimit_period=8 * concurrency
    )


def quote_url(url):
    return url.replace(" ", "%20")


def call_api(session, url):
    return api_setup.call_api(
        session,
        "GET",
        quote_url(url),
        "web",
        extra_error_codes=[403],
    )


# Get media lists

MATCH_FIELD = """([^<>]+)"""
MATCH_LAZY = """([^<>]+?)"""
TITLE_REGEX = re.compile('<h3 class="cardName">' + MATCH_FIELD + "</h3>")
TITLE_URL_REGEXES = {
    x: re.compile(f'href="/{x}/' + MATCH_LAZY + '"') for x in ["manga", "anime"]
}
SCORE_REGEX = re.compile('<div class="ttRating">' + MATCH_FIELD + "</div>")
STATUS_REGEX = re.compile('<span class="status' + MATCH_FIELD + '">')
PROGRESS_REGEXES = {
    x: re.compile("</span> " + MATCH_FIELD + f" {y}</div>")
    for (x, y) in zip(["manga", "anime"], ["chs", "eps"])
}
PAGE_REGEX = re.compile('page=([0-9]*)">')
FEED_TITLE_REGEX = re.compile('">' + MATCH_FIELD + "</h5>")
FEED_TIMESTAMP_REGEX = re.compile('data-timestamp="' + MATCH_FIELD + '">')


def unpack(x):
    assert len(x) == 1
    return x[0]


def is_private_list(resp, username):
    return " has chosen to make their content private." in resp.text


def is_invalid_user(resp, username):
    return f"<title>Search Results for {username}" in resp.text


def get_title(x):
    return sanitize_string(unpack(TITLE_REGEX.findall(x)))


def get_title_url(x, medium):
    return sanitize_string(unpack(TITLE_URL_REGEXES[medium].findall(x)))


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


def parse_entry(line, prev_line, medium):
    return (
        get_title(line),
        get_title_url(prev_line, medium),
        get_score(line),
        get_status(line),
        get_progress(line, medium),
    )


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
    while next_page and page < 4:
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
        prev_line = ""
        for line in resp.text.split("\n"):
            if '<h3 class="cardName">' in line:
                records.append(parse_entry(line, prev_line, medium))
            prev_line = line
        if str(page + 1) not in get_page_numbers(resp):
            next_page = False
    df = pd.DataFrame(
        data=list(reversed(records)),
        columns=["title", "url", "score", "status", "progress"],
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
    df["api_version"] = get_api_version()
    df["username"] = username
    return df, True


# Get media

MEDIA_TITLE_REGEX = re.compile(
    '<h1 itemprop="name"(?: class="long")?>' + MATCH_FIELD + "</h1>"
)
MEDIA_ALTTITLE_REGEX = re.compile('<h2 class="aka">' + MATCH_FIELD + "</h2>")
MEDIA_YEAR_REGEX = re.compile('<span class="iconYear"> ' + MATCH_FIELD + "</span>")
MEDIA_SEASON_REGEX = re.compile("/seasons/" + MATCH_FIELD + '">')
MEDIA_STUDIO_REGEX = re.compile(">" + MATCH_FIELD + "</a>")
MEDIA_TAG_REGEX = {
    x: re.compile(f'<a href="/{x}/tags/' + MATCH_LAZY + '"') for x in ["manga", "anime"]
}
MEDIA_SUMMARY_REGEX = re.compile(
    'property="og:description" content="' + MATCH_FIELD + '" />'
)
MEDIA_RELATION_REGEX = re.compile(
    f'<a href="/(anime|manga)/' + MATCH_FIELD + '" class="RelatedEntry '
)


def get_media_title(text):
    matches = MEDIA_TITLE_REGEX.findall(text)
    if matches:
        return html.unescape(unpack(matches))
    else:
        return ""


def get_media_alttitle(text):
    matches = MEDIA_ALTTITLE_REGEX.findall(text)
    if matches:
        x = unpack(matches)
        if "Alt titles" in x:
            x = x.split("Alt titles: ")[1].strip()
            return [html.unescape(y) for y in x.split(",")]
        else:
            return html.unescape(x.split("Alt title: ")[1].strip())
    else:
        return ""


def get_media_year(text):
    matches = MEDIA_YEAR_REGEX.findall(text)
    if matches:
        return html.unescape(unpack(matches))
    else:
        return ""


def get_media_season(text):
    matches = MEDIA_SEASON_REGEX.findall(text)
    if len(matches) == 2:  # first match is a link to the current season
        return html.unescape(matches[-1])
    else:
        return ""


def get_media_studios(text, medium):
    studios = []
    for line in text.split("\n"):
        match = None
        if medium == "anime":
            if line.startswith('<a href="/anime/studios/'):
                match = line
        if medium == "manga":
            if line.startswith('<a href="/manga/magazines/'):
                match = line
                studios += MEDIA_STUDIO_REGEX.findall(line)
            elif line.startswith('<a href="/manga/publishers/'):
                match = line
        if match is not None:
            matches = MEDIA_STUDIO_REGEX.findall(match)
            if matches:
                studios += matches
    if studios:
        return studios
    return ""


def get_media_genres(text, medium):
    return MEDIA_TAG_REGEX[medium].findall(text)


def get_media_summary(text):
    matches = MEDIA_SUMMARY_REGEX.findall(text)
    if not matches:
        return ""
    return unpack(matches)


def get_media_facts(session, url, medium):
    r = call_api(session, f"https://www.anime-planet.com/{medium}/{url}")
    if not r.ok:
        logging.warning(f"Received error while accessing {url}")
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.DataFrame.from_dict(
            {
                "url": [url],
                "title": [get_media_title(r.text)],
                "alttitle": [get_media_alttitle(r.text)],
                "year": [get_media_year(r.text)],
                "season": [get_media_season(r.text)],
                "studios": [get_media_studios(r.text, medium)],
                "genres": [get_media_genres(r.text, medium)],
                "summary": [get_media_summary(r.text)],
                "api_version": [get_api_version()],
            }
        ),
        pd.DataFrame.from_records(
            [
                ("relation", url, medium, m, t, get_api_version())
                for (m, t) in MEDIA_RELATION_REGEX.findall(r.text)
            ],
            columns=[
                "relation",
                "source_id",
                "source_media",
                "target_id",
                "target_media",
                "api_version",
            ],
        ),
    )