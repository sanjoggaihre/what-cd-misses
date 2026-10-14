#!/usr/bin/env python3
"""
timing_audit.py -- which GPUs did each training launch run on, and how long were its epochs?

For every Point-MAE log under the given run folders it prints one row per launch
(one .log file = one launch, resumes included):
  start / end      first and last timestamp in the log
  epochs           how many "[Training] EPOCH ... EpochTime" lines it holds
  mean / median    epoch time in seconds over that launch, excluding epoch 0
  GPUs             Slurm allocations (from sacct) whose time window covers the launch

Make the sacct file first (one line per allocation that had GPUs):
  sacct -u sgaihre -S 2026-07-01 -X --format=JobID,JobName%30,Start,End,AllocTRES%60 \
      | grep -i gres > gpu_jobs.txt

Then, from the Point-MAE folder:
  python timing_audit.py --sacct gpu_jobs.txt \
      experiments/pretrain_pepper_baseline/cfgs/PointMAE_baseline_exp1 \
      experiments/pretrain_pepper_baseline_dloss/cfgs/* \
      experiments/dloss_w0*/cfgs/* experiments/dcd_a50/cfgs/dcd_a50 \
      experiments_legacy/dcd_a20/cfgs/dcd_a20
"""
import argparse
import glob
import os
import re
import statistics
from datetime import datetime

TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
EP = re.compile(r"EPOCH: *(\d+) EpochTime = ([0-9.]+)")
JOB = re.compile(r"^(\S+)\s+(\S+)\s+(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\s+(\S+)\s+(\S+)")


def load_sacct(path):
    jobs = []
    for line in open(path):
        m = JOB.match(line.strip())
        if not m:
            continue
        jid, name, start, end, tres = m.groups()
        try:
            s = datetime.fromisoformat(start)
            e = datetime.now() if end in ("Unknown", "None") else datetime.fromisoformat(end)
        except ValueError:
            continue
        gpus = ",".join(f"{t}x{n}" for t, n in re.findall(r"gres/gpu:(\w+)=(\d+)", tres)) or "?"
        jobs.append((s, e, jid, name, gpus))
    return jobs


def audit_log(path):
    first = last = None
    times = []
    with open(path, errors="replace") as fh:
        for line in fh:
            t = TS.search(line)
            if t:
                ts = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")
                first = first or ts
                last = ts
            e = EP.search(line)
            if e:
                times.append((int(e.group(1)), float(e.group(2))))
    return first, last, times


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sacct", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    jobs = load_sacct(args.sacct)
    print(f"{len(jobs)} GPU allocations read from {args.sacct}\n")
    for run in args.runs:
        logs = sorted(glob.glob(os.path.join(run, "*.log")))
        if not logs:
            continue
        print(f"== {run.rstrip('/')}")
        for lg in logs:
            first, last, times = audit_log(lg)
            if first is None:
                print(f"   {os.path.basename(lg)}  (no timestamps)")
                continue
            ep = [t for e, t in times if e > 0]
            stats = (f"epochs {len(times):3d} ({times[0][0]}-{times[-1][0]})  "
                     f"mean {statistics.mean(ep):6.1f}  median {statistics.median(ep):6.1f}"
                     if ep else f"epochs {len(times):3d}")
            hits = [f"{g} (job {j})" for s, e, j, n, g in jobs if s <= first <= e]
            print(f"   {first:%m-%d %H:%M} -> {last:%m-%d %H:%M}  {stats}  "
                  f"GPUs: {'; '.join(hits) or 'no matching allocation'}")
        print()


if __name__ == "__main__":
    main()