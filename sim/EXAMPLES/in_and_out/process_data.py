#!/usr/bin/env python3

from __future__ import print_function

import os
import subprocess
import sys


def run_parse_output(filename: str) -> None:
    # The simulator output is binary; parse_output converts it to ASCII.
    # Also, some environments set LD_PRELOAD (e.g. libcuda.so) which can spam stderr;
    # we don't need it here, so drop it for cleanliness.
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)

    asc_path = filename + ".asc"
    with open(asc_path, "w") as out:
        subprocess.run(["../../parse_output", filename, "-ascii"], stdout=out, check=True, env=env)


def extract_rates(filename: str, sample_time: float = 0.2):
    rates_gbps = []
    with open(filename + ".asc", "r") as ifile:
for line in ifile:
    data = line.split()
            if not data:
                continue

            try:
                t = float(data[0])
            except Exception:
                continue

            # Expected format (example):
            # 0.200000000 Type NDP_SINK ID 1704 Ev RATE CAck ... ReorderBuffer 0 Rate 3103020000
            if t != sample_time:
                continue
            if len(data) < 3 or data[2] != "NDP_SINK":
                continue
            if "Rate" not in data:
                continue

            idx = data.index("Rate")
            if idx + 1 >= len(data):
                continue

            try:
                rate_bps = int(data[idx + 1])
            except Exception:
                continue

            rate_gbps = rate_bps * 8 / 1_000_000_000.0
            rates_gbps.append(rate_gbps)
            print(rate_gbps)

    return rates_gbps


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: process_data.py <logfile>", file=sys.stderr)
        return 2

    filename = sys.argv[1]
    run_parse_output(filename)

    rates = extract_rates(filename, sample_time=0.2)

    with open(filename + ".urates", "w+") as ofile:
for r in rates:
            print(r, file=ofile)

rates.sort()
print(rates)
    with open(filename + ".rates", "w+") as ofile:
for r in rates:
            print(r, file=ofile)

    return 0
    

if __name__ == "__main__":
    raise SystemExit(main())
