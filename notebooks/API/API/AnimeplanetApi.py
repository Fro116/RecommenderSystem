import re

API_PERIOD = 4
exec(open("ApiSetup.py").read())


def call_api(url):
    header = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
    return call_api_internal(
        url, "GET", "web", extra_error_codes=[403], headers={"User-Agent": header}
    )


MATCHFIELD = """([^<>]+)"""


def unpack(x):
    assert len(x) == 1
    return x[0]


def is_private_list(resp, username):
    usernames = re.findall(
        MATCHFIELD + " has chosen to make their content private.", resp.text
    )
    if not usernames:
        return False
    assert unpack(usernames).lower().strip() == username.lower()
    return True


def is_invalid_user(resp, username):
    return f"<title>Search Results for {username}" in resp.text


def get_title(x):
    return sanitize_string(
        unpack(re.findall('<h3 class="cardName">' + MATCHFIELD + "</h3>", x))
    )


def get_score(x):
    scores = re.findall('<div class="ttRating">' + MATCHFIELD + "</div>", x)
    if not scores:
        return "0"
    return str(2 * float(unpack(scores)))


def get_status(x):
    status = re.findall('<span class="status' + MATCHFIELD + '">', x)
    return unpack(status)


def get_progress(x, medium):
    if medium == "anime":
        suffix = "eps"
    elif medium == "manga":
        suffix = "chs"
    else:
        assert False
    progress = re.findall("</span> " + MATCHFIELD + f" {suffix}</div>", x)
    if not progress:
        return 0
    return unpack(progress)


def parse_entry(x, medium):
    return (get_title(x), get_score(x), get_status(x), get_progress(x, medium))


def get_page_entries(resp):
    return [x for x in resp.text.split("\n") if '<h3 class="cardName">' in x]


def get_page_numbers(resp):
    return set(re.findall('page=([0-9]*)">', resp.text))


def get_feed_entries(resp):
    return [x for x in resp.text.split("\n") if "data-timestamp" in x]


def get_feed_title(x):
    return unpack(re.findall('">' + MATCHFIELD + f"</h5>", x))


def get_feed_timestamp(x):
    return unpack(re.findall('data-timestamp="' + MATCHFIELD + f'">', x))


def get_feed_data(username, medium):
    feed_data = {}
    next_page = True
    page = 0
    while next_page and page < 15:
        page += 1
        url = (
            f"https://www.anime-planet.com/users/{username}"
            + f"/feed?type={medium}&page={page}"
        )
        resp = call_api(url)
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


def get_user_media_list(username, medium):
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
        resp = call_api(url)
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
        feed_data, feed_ok = get_feed_data(username, medium)
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