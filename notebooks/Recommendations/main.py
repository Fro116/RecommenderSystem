import hashlib
import os
import shutil
import subprocess

from flask import Flask, render_template, request

app = Flask(__name__)


def spawn_parallel(commands):
    procs = []
    for cmd in commands:
        procs.append(
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        )
    for p in procs:
        p.wait()


def hash_files(files):
    hash_sha256 = hashlib.sha256()
    for fn in files:
        with open(fn, "rb") as f:
            while chunk := f.read(8192):
                hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def create_julia_command(script, source, username):
    script_to_port = {
        "CompressSplits.jl": 3006,
        "Baseline.jl": 3007,
        "BagOfWords.jl": 3008,
        "Nondirectional.jl": 3009,
        "Transformer.jl": 3010,
        "Ensemble.jl": 3011,
        "Recommendations.jl": 3012,
    }
    script_to_model_port = {
        "Transformer.jl": 3004,
        "BagOfWords.jl": 3005,
    }
    url = f"http://localhost:{script_to_port[script]}/query?username={username}&source={source}"
    if script in script_to_model_port:
        url += f"&modelport={script_to_model_port[script]}"
    return ["curl", url]


def save_html_page(source, username):
    source_map = {
        "MyAnimeList": "mal",
        "AniList": "anilist",
        "Kitsu": "kitsu",
        "Anime-Planet": "animeplanet",
    }
    source = source_map[source]
    if "@" in username:
        fields = username.split("@")
        assert len(fields) == 2 and fields[0] == source
    data_path = os.path.abspath("../../data/recommendations")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if "@" in username:
        rec_dir = os.path.join(data_path, *username.split("@"))
    else:
        rec_dir = os.path.join(data_path, source, str(username))
    if os.path.exists(rec_dir):
        shutil.rmtree(rec_dir)

    source_to_port = {"mal": 3000, "anilist": 3001, "kitsu": 3002, "animeplanet": 3003}
    spawn_parallel(
        [
            [
                "curl",
                f"http://localhost:{source_to_port[source]}/query?username={username}&medium={medium}",
            ]
            for medium in ["manga", "anime"]
        ]
    )

    # if the lists haven't changed then serve cached recs
    hash = hash_files(
        os.path.join(rec_dir, f"user_{x}_list.csv") for x in ["manga", "anime"]
    )
    output_fn = f"templates/{hash}.html"
    if os.path.exists(output_fn):
        return output_fn

    username = str(username)
    if "@" in username:
        username = username.split("@")[1]

    julia = lambda x: create_julia_command(x, source, username)
    spawn_parallel([julia(x) for x in ["CompressSplits.jl"]])
    spawn_parallel([julia(x) for x in ["Baseline.jl"]])
    spawn_parallel(
        [julia(x) for x in ["BagOfWords.jl", "Nondirectional.jl", "Transformer.jl"]]
    )
    spawn_parallel([julia(x) for x in ["Ensemble.jl"]])
    spawn_parallel([julia(x) for x in ["Recommendations.jl"]])
    os.rename(os.path.join(rec_dir, "Recommendations.html"), output_fn)
    return output_fn


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    source = request.form["source"]
    username = request.form["username"]
    fn = save_html_page(source, username)
    return render_template(os.path.basename(fn))


if __name__ == "__main__":
    app.run()