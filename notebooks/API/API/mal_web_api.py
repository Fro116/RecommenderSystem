import html
import logging
import re

import pandas as pd

from . import api_setup


def make_session(proxies, concurrency):
    return api_setup.ProxySession(
        proxies, ratelimit_calls=concurrency, ratelimit_period=4 * concurrency
    )


def call_api(session, url):
    return api_setup.call_api(session, "GET", url, "web", extra_error_codes=[403, 405])


def unpack(x):
    assert len(x) == 1
    return x[0]


def maybe_unpack(x):
    if len(x) == 0:
        return ""
    assert len(x) == 1
    return x[0]


def unique(x):
    return unpack(list(set(x)))


MATCH_FIELD = """(.*?)"""

TITLE_REGEXES = {
    "anime": re.compile(
        '<div class="h1-title"><div itemprop="name"><h1 class="title-name h1_bold_none"><strong>'
        + MATCH_FIELD
        + "</strong></h1>",
        re.DOTALL,
    ),
    "manga": re.compile(
        '<span class="h1-title"><span itemprop="name">' + MATCH_FIELD + "</span>",
        re.DOTALL,
    ),
}

ALTTITLE_REGEXES = {
    "anime": re.compile(
        '<p class="title-english title-inherit">' + MATCH_FIELD + "</p>"
    ),
    "manga": re.compile(
        '<span class="title-english">' + MATCH_FIELD + "</span>"
    ),
}

SUMMARY_REGEXES = {
    "anime": re.compile(
        'Synopsis</h2></div><p itemprop="description">' + MATCH_FIELD + "</p>",
        re.DOTALL,
    ),
    "manga": re.compile(
        'Synopsis</h2><span itemprop="description">' + MATCH_FIELD + "</span>",
        re.DOTALL,
    ),
}

DATE_REGEXES = {
    "anime": re.compile(
        '<span class="dark_text">Aired:</span>' + MATCH_FIELD + "</div>"
    ),
    "manga": re.compile(
        '<span class="dark_text">Published:</span>' + MATCH_FIELD + "</div>"
    ),
}

PROGRESS_REGEXES = {
    "episodes": re.compile(
        '<span class="dark_text">Episodes:</span>' + MATCH_FIELD + "</div>"
    ),
    "chapters": re.compile('<span id="totalChaps".*?>' + MATCH_FIELD + "</span>"),
    "volumes": re.compile('<span id="totalVols".*?>' + MATCH_FIELD + "</span>"),
}

STATUS_REGEXES = {
    "anime": re.compile(
        '<span class="dark_text">Status:</span>' + MATCH_FIELD + "</div>"
    ),
    "manga": re.compile(
        '<span class="dark_text">Status:</span>' + MATCH_FIELD + "</div>"
    ),
}

MEDIATYPE_REGEXES = {
    "anime": re.compile(
        '<span class="dark_text">Type:</span>' + MATCH_FIELD + "</div>"
    ),
    "manga": re.compile(
        '<span class="dark_text">Type:</span>' + MATCH_FIELD + "</div>"
    ),
}

SEASON_REGEXES = {
    "anime": re.compile(
        '<span class="dark_text">Premiered:</span>' + MATCH_FIELD + "</div>"
    ),
    "manga": None,
}

STUDIO_REGEXES = {
    "anime": re.compile(
        '<span class="dark_text">Studios:</span>' + MATCH_FIELD + "</div>", re.DOTALL
    ),
    "manga": re.compile(
        '<span class="dark_text">Serialization:</span>' + MATCH_FIELD + "</div>",
        re.DOTALL,
    ),
    "title": re.compile('title="' + MATCH_FIELD + '"'),
}


def title(text, medium):
    x = unpack(TITLE_REGEXES[medium].findall(text))
    if medium == "manga":
        x = x.split('<br><span class="title-english">')[0]
    return html.unescape(x)


def english_title(text, medium):
    return html.unescape(maybe_unpack(ALTTITLE_REGEXES[medium].findall(text)))


def summary(text, medium):
    if "No synopsis information has been added to this title." in text:
        return ""
    x = unpack(SUMMARY_REGEXES[medium].findall(text))
    x = x.replace('<span itemprop="description">', "")
    x = x.replace("<br />\r", "\n")
    x = x.split("</a>")[0]
    x = html.unescape(x)
    x = re.sub(r"\n+", "\n", x).strip()
    return x


def date(text, startdate, medium):
    x = unpack(DATE_REGEXES[medium].findall(text))
    x = x.strip().split(" to ")
    if startdate:
        return x[0]
    else:
        return x[min(len(x) - 1, 1)]


def episodes(text):
    return unpack(PROGRESS_REGEXES["episodes"].findall(text)).strip()


def chapters(text):
    return unique(PROGRESS_REGEXES["chapters"].findall(text))


def volumes(text):
    return unique(PROGRESS_REGEXES["volumes"].findall(text))


def status(text, medium):
    return unpack(STATUS_REGEXES[medium].findall(text)).strip()


def season(text, medium):
    assert medium == "anime"
    matches = SEASON_REGEXES[medium].findall(text)
    if not matches:
        return ""
    x = unpack(matches).strip()
    try:
        return x.split(">")[1].split("<")[0]
    except:
        return ""


def media_type(text, medium):
    x = unpack(MEDIATYPE_REGEXES[medium].findall(text)).strip()
    if ">" in x:
        x = x.split(">")[1].split("<")[0]
    return x


def start_date(x, medium):
    return date(x, True, medium)


def end_date(x, medium):
    return date(x, False, medium)


def genres(text, medium):
    genres = re.findall(f'href="/{medium}/genre/.*?/.*?"', text)
    return [x.split("/")[-1][:-1] for x in genres]


def studios(text, medium):
    x = unpack(STUDIO_REGEXES[medium].findall(text))
    return STUDIO_REGEXES["title"].findall(x)


def process_media_details_response(response, uid, medium):
    raw_text = response.text
    text = raw_text.replace("\n", " ")
    if medium == "anime":
        df = pd.DataFrame.from_dict(
            {
                "uid": [str(uid)],
                "title": [title(text, medium)],
                "english_title": [english_title(text, medium)],
                "summary": [summary(text, medium)],
                "type": [media_type(text, medium)],
                "status": [status(text, medium)],
                "num_episodes": [episodes(text)],
                "start_date": [start_date(text, medium)],
                "end_date": [end_date(text, medium)],
                "season": [season(text, medium)],
                "genres": [genres(text, medium)],
                "studios": [studios(text, medium)],
            }
        )
    elif medium == "manga":
        df = pd.DataFrame.from_dict(
            {
                "uid": [str(uid)],
                "title": [title(text, medium)],
                "english_title": [english_title(text, medium)],
                "summary": [summary(text, medium)],
                "type": [media_type(text, medium)],
                "status": [status(text, medium)],
                "num_chapters": [chapters(text)],
                "num_volumes": [volumes(text)],
                "start_date": [start_date(text, medium)],
                "end_date": [end_date(text, medium)],
                "genres": [genres(text, medium)],
                "studios": [studios(text, medium)],
            }
        )
    else:
        assert False
    return df


def process_media_relations_response(response, uid, media):
    relation_types = {
        "Sequel": "SEQUEL",
        "Prequel": "PREQUEL",
        "Alternative Setting": "ALTERNATIVE_SETTING",
        "Alternative Version": "ALTERNATIVE_VERSION",
        "Side Story": "SIDE_STORY",
        "Summary": "SUMMARY",
        "Full Story": "FULL_STORY",
        "Parent Story": "PARENT_STORY",
        "Spin-Off": "SPIN_OFF",
        "Adaptation": "ADAPTATION",
        "Character": "CHARACTER",
        "Other": "OTHER",
    }

    records = []
    lines = re.split("<|>", response.text)
    starting_line = "Related Entries"
    if starting_line not in lines:
        return pd.DataFrame()
    start = lines.index(starting_line)
    tile_cards = True
    last_href = None
    rtype = None
    for line in lines[start:]:
        line = line.strip()
        key = line.split("(")[0].split(":")[0].strip()
        if key in relation_types:
            rtype = relation_types[key]
            if tile_cards and last_href is None:
                tile_cards = False
        if "href" in line:
            l = line
            for target_media in ["anime", "manga"]:
                for target_id in re.findall(rf"/{target_media}/[0-9]+", l):
                    target_id = int(target_id.split("/")[-1])
                    if target_id == uid and target_media == media:
                        relations = pd.DataFrame.from_records(
                            records,
                            columns=[
                                "relation",
                                "source_id",
                                "source_media",
                                "target_id",
                                "target_media",
                            ],
                        )
                        return relations
                    else:
                        if tile_cards:
                            href = (target_id, target_media)
                            if last_href is None:
                                last_href = href
                            elif last_href == href:
                                records.append(
                                    (
                                        rtype,
                                        uid,
                                        media.upper(),
                                        target_id,
                                        target_media.upper(),
                                    )
                                )
                                last_href = None
                            else:
                                assert False, f"unknown href {last_href} {href} for {media} {uid}"
                        else:
                            records.append(
                                (
                                    rtype,
                                    uid,
                                    media.upper(),
                                    target_id,
                                    target_media.upper(),
                                )
                            )
    assert False, f"could not parse {media} relations for {uid}"


def get_media_facts(session, uid, media):
    uid = int(uid)
    url = f"https://myanimelist.net/{media}/{uid}"
    response = call_api(session, url)
    try:
        response.raise_for_status()
        details = process_media_details_response(response, uid, media)
        relations = process_media_relations_response(response, uid, media)
    except Exception as e:
        logging.warning(f"Received error {str(e)} while accessing {url}")
        return pd.DataFrame(), pd.DataFrame()
    return details, relations


def get_username(session, userid):
    try:
        url = f"https://myanimelist.net/comments.php?id={userid}"
        response = call_api(session, url)
        if response.status_code in [404]:
            # the user may have deleted their account
            return ""
        if not response.ok:
            logging.warning(f"Error {response} received when handling {url}")
            return ""
        urls = re.findall('''/profile/[^"/%]+"''', response.text)
        users = [x[len("/profile/") : -len('"')] for x in urls]
        return html.unescape(users[0])
    except Exception as e:
        logging.info(f"Error with {userid}")
        logging.info(f"Error with text {response.text}")
        logging.info(f"Error with users {users}")
        raise e