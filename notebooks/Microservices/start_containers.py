import os
import signal
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PROCS = []
APPS = [
    "index",    
    "fetch_media_lists", 
    "compress_media_lists", 
    "nondirectional",
    "transformer_jl",
    "transformer_py",
    "bagofwords_jl",
    "bagofwords_py",
    "ensemble",
]

def runapp(app, port):
    cmdlist = [
        "docker",
        "run",
        "--network",
        "rsys",
        "--name",
        app,
        "-p",
        f"{port}:8080",
        f"rsys/{app}",
    ]
    PROCS.append(subprocess.Popen(cmdlist))


def signal_handler(sig, frame):
    for p in PROCS:
        p.terminate()
    for p in PROCS:
        p.wait()
    subprocess.run(["docker", "network", "rm", "rsys"])
    for app in APPS:
        subprocess.run(["docker", "remove", app])
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    PROCS.append(subprocess.Popen(["docker", "network", "create", "rsys"]))
    for i, app in enumerate(APPS):
        runapp(app, 3000+i)
    while True:
        time.sleep(3600)
except Exception as e:
    signal_handler(signal.SIGINT, None)
    raise e
