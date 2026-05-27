#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/tinyllama.gguf}"
PROMPT="${2:-Hello, my name is }"
RUNS="${RUNS:-10}"
OUT="${OUT:-bench.csv}"

rm -f "$OUT"

echo "Running benchmark"
echo "model: $MODEL"
echo "runs:  $RUNS"
echo "out:   $OUT"

for i in $(seq 1 "$RUNS"); do
    echo "Run $i/$RUNS" >&2

    if [ "$i" -eq 1 ]; then
        CSV_ARGS="--csv"
    else
        CSV_ARGS="--csv --csv-no-header"
    fi

    ./build/llama_latency \
        "$MODEL" \
        "$PROMPT" \
        --threads 8 \
        --ctx-size 2048 \
        --batch-size 512 \
        --max-tokens 128 \
        $CSV_ARGS \
        >> "$OUT"
done

echo "Wrote $OUT"