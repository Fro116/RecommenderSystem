import os
import signal
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
procs = []
app_port = 5000
source_to_port = {"mal": 3000, "anilist": 3001, "kitsu": 3002, "animeplanet": 3003}
python_script_to_port = {
    "transformer": 3004,
    "bagofwords": 3005,
}
julia_script_to_port = {
    "CompressSplits.jl": 3006,
    "Baseline.jl": 3007,
    "BagOfWords.jl": 3008,
    "Nondirectional.jl": 3009,
    "Transformer.jl": 3010,
    "Ensemble.jl": 3011,
    "Recommendations.jl": 3012,
}


def runcmd(cmdlist, env=None, quiet=True):
    if quiet:
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
    procs.append(subprocess.Popen(cmdlist, env=env, **kwargs))


def juliacmd(cmdlist):
    port = julia_script_to_port[cmdlist[-1]]
    cmdlist.append(str(port))
    runcmd(cmdlist)


def signal_handler(sig, frame):
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

runcmd(
    [
        "gunicorn",
        "-w",
        "2",
        "main:app",
        "-b",
        f"0.0.0.0:{app_port}",
    ],
)

for source, port in source_to_port.items():
    env = os.environ.copy()
    env["RSYS_LIST_SOURCE"] = source
    runcmd(
        [
            "gunicorn",
            "-w",
            "2",
            "fetch_media_lists:app",
            "-b",
            f"localhost:{port}",
        ],
        env,
    )

juliacmd(["julia", "-t", "4", "Recommendations.jl"])

os.chdir("../InferenceAlphas")

for server, workers in zip(
    [
        "CompressSplits.jl",
        "Baseline.jl",
        "BagOfWords.jl",
        "Nondirectional.jl",
        "Transformer.jl",
        "Ensemble.jl",
    ],
    [2, 2, 8, 2, 2, 4],
):
    juliacmd(["julia", "-t", str(workers), server])

for script, workers in zip(["transformer", "bagofwords"], [2, 8]):
    runcmd(
        [
            "gunicorn",
            "-w",
            str(workers),
            f"{script}:app",
            "-b",
            f"localhost:{python_script_to_port[script]}",
        ]
    )

print(
    "Successfully started servers! To shutdown servers, "
    "send a SIGINT or SIGTERM signal to this script."
)
while True:
    time.sleep(3600)
