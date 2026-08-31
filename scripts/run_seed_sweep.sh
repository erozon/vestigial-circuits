#!/usr/bin/env bash
# Multi-seed sweep for audit addendum D (statistical credibility) and B
# (Z_113 baseline trained with matched recipe).
#
# Runs:
#   - 5 seeds × D_59 at 30% × 40K epochs  → grokking-epoch + mechanism
#   - 3 seeds × Z_113 at 30% × 40K epochs → comparison baseline (was Nanda's ~10K)
#
# Total 8 runs. Four-way parallel on M2 with OMP_NUM_THREADS=2 each.
# Wall-clock estimate: ~4h on M2 (smoke test ran at ~6 epochs/sec).
#
# Output: runs/seed_sweep/<config>_seed<N>/{metrics.jsonl,checkpoints/}

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_BASE="runs/seed_sweep"
mkdir -p "$OUT_BASE"

# Verify data exists
for dir in d59_30pct z113; do
    [ -f "data/$dir/train.txt" ] || { echo "Missing data/$dir/train.txt"; exit 1; }
done

run_one() {
    local name="$1"
    local data="$2"
    local epochs="$3"
    local seed="$4"

    local out_dir="$OUT_BASE/$name"
    mkdir -p "$out_dir/checkpoints"

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON" -m src.train \
        --train_file "data/$data/train.txt" \
        --val_file "data/$data/test.txt" \
        --epochs "$epochs" \
        --weight_decay 1.0 \
        --learning_rate 1e-3 \
        --adam_betas 0.9 0.98 \
        --warmup_steps 10 \
        --full_batch \
        --num_layers 1 \
        --d_model 128 \
        --no_layernorm \
        --learned_pos_emb \
        --nanda_init \
        --max_seq_length 8 \
        --result_only \
        --no_grad_clip \
        --print_interval 1000 \
        --eval_interval 100 \
        --log_interval 500 \
        --checkpoint_interval 10000 \
        --device cpu \
        --seed "$seed" \
        --metrics_file "$out_dir/metrics.jsonl" \
        --output_dir "$out_dir/checkpoints" \
        > "$out_dir/train.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE: $name" >> "$OUT_BASE/sweep.log"
}

echo "[$(date '+%H:%M:%S')] Wave 1: 4 long runs (D_59 30%, seeds 0-3)" | tee "$OUT_BASE/sweep.log"
run_one d59_30pct_seed0 d59_30pct 40000 0 &
run_one d59_30pct_seed1 d59_30pct 40000 1 &
run_one d59_30pct_seed2 d59_30pct 40000 2 &
run_one d59_30pct_seed3 d59_30pct 40000 3 &
wait

echo "[$(date '+%H:%M:%S')] Wave 2: 1 D_59 30% leftover + 3 Z_113 30% seeds" | tee -a "$OUT_BASE/sweep.log"
run_one d59_30pct_seed4 d59_30pct 40000 4 &
run_one z113_30pct_seed0 z113 40000 0 &
run_one z113_30pct_seed1 z113 40000 1 &
run_one z113_30pct_seed2 z113 40000 2 &
wait

echo "[$(date '+%H:%M:%S')] All 8 runs complete" | tee -a "$OUT_BASE/sweep.log"
