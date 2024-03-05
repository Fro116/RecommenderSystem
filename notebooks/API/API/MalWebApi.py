import html
import re

API_PERIOD = 4
exec(open("ApiSetup.py").read())


def call_web_api(url):
    return call_api_internal(url, "GET", "web", extra_error_codes=[403])


def sanitize_summary(x):
    x = x.replace('<span itemprop="description">', "")
    x = x.replace("<br />\r", "\n")
    x = x.split("</a>")[0]
    x = html.unescape(x)
    return x


def parse_anime(response):
    def title(text):
        start = 'property="og:title" content="'
        end = '">'
        x = re.findall(start + ".*?" + end, text)
        assert len(x) == 1
        x = x[0][len(start) : -len(end)]
        x = html.unescape(x)
        return x

    def english_title(text):
        start = '<p class="title-english title-inherit">'
        end = "</p>"
        x = re.findall(start + ".*?" + end, text)
        if len(x) == 0:
            return ""
        assert len(x) == 1
        x = x[0][len(start) : -len(end)]
        x = html.unescape(x)
        return x

    def summary(text):
        start = '<h2>Synopsis</h2></div><p itemprop="description">'
        end = "</p>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)]
        return sanitize_summary(x)

    def date(text, startdate):
        start = '<span class="dark_text">Aired:</span>'
        end = "</div>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)].strip()
        x = x.split(" to ")
        if startdate:
            return x[0]
        else:
            return x[min(len(x) - 1, 1)]

    def episodes(text):
        start = '<span class="dark_text">Episodes:</span>'
        end = "</div>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)].strip()
        return x

    def status(text):
        start = '<span class="dark_text">Status:</span>'
        end = "</div>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)].strip()
        return x

    def media_type(text):
        start = '<span class="dark_text">Type:</span>'
        end = "</div>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)].strip()
        if ">" in x:
            x = x.split(">")[1].split("<")[0]
        return x

    def start_date(x):
        return date(x, True)

    def end_date(x):
        return date(x, False)

    def genres(text):
        genres = re.findall('href="/anime/genre/.*?/.*?"', text)
        return [x.split("/")[-1][:-1] for x in genres]

    text = response.text
    return pd.DataFrame.from_dict(
        {
            "title": [title(text)],
            "english_title": [english_title(text)],
            "summary": [summary(text)],
            "type": [media_type(text)],
            "status": [status(text)],
            "num_episodes": [episodes(text)],
            "start_date": [start_date(text)],
            "end_date": [end_date(text)],
            "genres": [genres(text)],
        }
    )


def parse_manga(response):
    def title(text):
        start = 'property="og:title" content="'
        end = '">'
        x = re.findall(start + ".*?" + end, text)
        assert len(x) == 1
        x = x[0][len(start) : -len(end)]
        x = html.unescape(x)
        return x

    def english_title(text):
        start = '<span class="title-english">'
        end = "</span>"
        x = re.findall(start + ".*?" + end, text)
        if len(x) == 0:
            return ""
        assert len(x) == 1
        x = x[0][len(start) : -len(end)]
        x = html.unescape(x)
        return x

    def summary(text):
        start = "Synopsis</h2>"
        end = "</span>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)]
        return sanitize_summary(x)

    def date(text, startdate):
        start = '<span class="dark_text">Published:</span>'
        end = "</div>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)].strip()
        x = x.split(" to ")
        if startdate:
            return x[0]
        else:
            return x[min(len(x) - 1, 1)]

    def chapters(text):
        start = '<span id="totalChaps".*?>'
        end = "</span>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        x = x[0].split(">")[1].split("<")[0]
        return x

    def volumes(text):
        start = '<span id="totalVols".*?>'
        end = "</span>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        x = x[0].split(">")[1].split("<")[0]
        return x

    def media_type(text):
        start = '<span class="dark_text">Type:</span>'
        end = "</div>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)].strip()
        if ">" in x:
            x = x.split(">")[1].split("<")[0]
        return x

    def status(text):
        start = '<span class="dark_text">Status:</span>'
        end = "</div>"
        x = re.findall(start + ".*?" + end, text.replace("\n", ""))
        assert len(x) == 1
        x = x[0][len(start) : -len(end)].strip()
        if ">" in x:
            x = x.split(">")[1].split("<")[0]
        return x

    def start_date(x):
        return date(x, True)

    def end_date(x):
        return date(x, False)

    def genres(text):
        genres = re.findall('href="/manga/genre/.*?/.*?"', text)
        return [x.split("/")[-1][:-1] for x in genres]

    text = response.text
    return pd.DataFrame.from_dict(
        {
            "title": [title(text)],
            "english_title": [english_title(text)],
            "summary": [summary(text)],
            "type": [media_type(text)],
            "status": [status(text)],
            "num_chapters": [chapters(text)],
            "num_volumes": [volumes(text)],
            "start_date": [start_date(text)],
            "end_date": [end_date(text)],
            "genres": [genres(text)],
        }
    )


def process_media_details_response(response, uid, media):
    if media == "anime":
        df = parse_anime(response)
    elif media == "manga":
        df = parse_manga(response)
    else:
        assert False
    df[f"{media}_id"] = uid
    return df


def process_media_relations_response(response, uid, media):
    relation_types = {
        "Sequel:": "SEQUEL",
        "Prequel:": "PREQUEL",
        "Alternative setting:": "ALTERNATIVE_SETTING",
        "Alternative version:": "ALTERNATIVE_VERSION",
        "Side story:": "SIDE_STORY",
        "Summary:": "SUMMARY",
        "Full story:": "FULL_STORY",
        "Parent story:": "PARENT_STORY",
        "Spin-off:": "SPIN_OFF",
        "Adaptation:": "ADAPTATION",
        "Character:": "CHARACTER",
        "Other:": "OTHER",
    }

    records = []
    lines = re.split("<|>", response.text)
    starting_line = f"Related {media.capitalize()}"
    if starting_line not in lines:
        return pd.DataFrame()
    start = lines.index(starting_line)
    for line in lines[start:]:
        if line in relation_types:
            rtype = relation_types[line]
        elif "href" in line:
            l = line
            for target_media in ["anime", "manga"]:
                for target_id in re.findall(rf"/{target_media}/[0-9]+", l):
                    target_id = int(target_id.split("/")[-1])
                    if target_id == uid:
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
                    records.append(
                        (rtype, uid, media.upper(), target_id, target_media.upper())
                    )
    assert False, f"could not parse {media} relations for {uid}"


def get_media_facts(uid, media):
    url = f"https://myanimelist.net/{media}/{uid}"
    response = call_web_api(url)
    try:
        response.raise_for_status()
        details = process_media_details_response(response, uid, media)
        relations = process_media_relations_response(response, uid, media)
    except Exception as e:
        logger.warning(f"Received error {str(e)} while accessing {url}")
        return pd.DataFrame(), pd.DataFrame()
    return details, relations


# returns all usernames that have commented on the given userid's profile
def get_username(userid):
    try:
        url = f"https://myanimelist.net/comments.php?id={userid}"
        response = call_web_api(url)
        if response.status_code in [404]:
            # the user may have deleted their account
            return ""
        if not response.ok:
            logger.warning(f"Error {response} received when handling {url}")
            return ""
        urls = re.findall('''/profile/[^"/%]+"''', response.text)
        users = [x[len("/profile/") : -len('"')] for x in urls]
        return html.unescape(users[0])
    except Exception as e:
        logger.info(f"Error with {userid}")
        logger.info(f"Error with text {response.text}")
        logger.info(f"Error with users {users}")
        raise e