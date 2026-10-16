"""Akira's training queue: runs training jobs one at a time, forever.

    nohup python3 scripts/train_queue.py > models/queue.log 2>&1 &

Jobs live in models/queue.json and are re-read before each job starts, so jobs
can be added or reordered while the queue runs:

    [{"name": "core-pro", "cmd": "python3 -W ignore -m akira.native.train --brain core-pro --minutes 240"}, ...]

A job that finishes writes models/jobs/<name>.done; one that fails writes
<name>.failed (delete it to retry). Every trainer resumes from its last
checkpoint, so killing and restarting the queue loses at most a few minutes.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE = os.path.join(ROOT, "models", "queue.json")
MARKS = os.path.join(ROOT, "models", "jobs")
os.makedirs(MARKS, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def next_job():
    try:
        jobs = json.load(open(QUEUE))
    except (OSError, ValueError) as e:
        log(f"can't read queue: {e}")
        return None
    for job in jobs:
        mark = os.path.join(MARKS, job["name"])
        if not os.path.exists(mark + ".done") and not os.path.exists(mark + ".failed"):
            return job
    return None


while True:
    job = next_job()
    if job is None:
        time.sleep(60)
        continue
    log(f"start {job['name']}: {job['cmd']}")
    out = open(os.path.join(MARKS, job["name"] + ".log"), "a")
    t0 = time.time()
    code = subprocess.call(job["cmd"], shell=True, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    mark = os.path.join(MARKS, job["name"]) + (".done" if code == 0 else ".failed")
    open(mark, "w").write(f"exit {code} after {(time.time() - t0) / 60:.1f} min\n")
    log(f"{'done' if code == 0 else 'FAILED'} {job['name']} (exit {code}, {(time.time() - t0) / 60:.1f} min)")
    sys.stdout.flush()
