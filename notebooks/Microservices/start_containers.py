import os
import signal
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PROCS = []

def runapp(app, port):
    cmdlist = [
        "docker",
        "run",
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
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    runapp("fetch_media_lists", 3000)
    runapp("compress_media_lists", 3001)
    runapp("nondirectional", 3002)
    runapp("transformer_jl", 3003)
    runapp("transformer_py", 3004)
    runapp("bagofwords_jl", 3005)
    runapp("bagofwords_py", 3006)
    runapp("ensemble", 3007)
    while True:
        time.sleep(3600)
except Exception as e:
    signal_handler(signal.SIGINT, None)
    raise e
