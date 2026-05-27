#!/usr/bin/env python3

import csv
import sys
from statistics import mean, median

if len(sys.argv) != 2:
    print("Usage: scripts/summarize.py bench.csv")
    sys.exit(1)

path = sys.argv[1]

metrics = [
    "prefill_ms",
    "ttft_ms",
    "p50_inter_token_ms",
    "p95_inter_token_ms",
    "p99_inter_token_ms",
    "prompt_tokens_per_sec",
    "generation_tokens_per_sec",
]

rows = []

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if not rows:
    print("No rows found")
    sys.exit(1)

print(f"runs: {len(rows)}")
print()

for metric in metrics:
    values = [float(row[metric]) for row in rows]

    print(metric)
    print(f"  mean:   {mean(values):.3f}")
    print(f"  median: {median(values):.3f}")
    print(f"  min:    {min(values):.3f}")
    print(f"  max:    {max(values):.3f}")
    print()