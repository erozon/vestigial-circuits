#!/usr/bin/env bash
# Resume the two D_59 30% seeds that failed to grok in 40K epochs.
# Run them to epoch 80K with finer checkpoint intervals (every 5K) so we
# can track the partial-mechanism trajectory over additional training.
# See AUDIT.md addendum (c) direction J.
#
# Output: runs/seed_sweep/<config>_resumed/{metrics.jsonl,checkpoints/}
# Two parallel jobs, OMP_NUM_THREADS=2 each. Wall-clock ~5h on M2.

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_BASE="runs/seed_sweep"

run_resume() {
    local seed="$1"
    local src_dir="$OUT_BASE/d59_30pct_seed${seed}"
    local out_dir="$OUT_BASE/d59_30pct_seed${seed}_resumed"
    mkdir -p "$out_dir/checkpoints"

    # Copy the seed's metrics so the resumed run's metrics file picks up
    # where the original left off (epoch 40K) rather than starting empty.
    cp "$src_dir/metrics.jsonl" "$out_dir/metrics.jsonl"

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON" -m src.train \
        --train_file data/d59_30pct/train.txt \
        --val_file data/d59_30pct/test.txt \
        --epochs 80000 \
        --resume_from "$src_dir/checkpoints/checkpoint_epoch_40000.pt" \
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
        --checkpoint_interval 5000 \
        --device cpu \
        --seed "$seed" \
        --metrics_file "$out_dir/metrics.jsonl" \
        --output_dir "$out_dir/checkpoints" \
        > "$out_dir/train.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE: seed${seed} resume" >> "$OUT_BASE/resume.log"
}

echo "[$(date '+%H:%M:%S')] Resume wave: seeds 2 and 4 from epoch 40K to 80K" | tee "$OUT_BASE/resume.log"
run_resume 2 &
run_resume 4 &
wait

echo "[$(date '+%H:%M:%S')] Resume complete" | tee -a "$OUT_BASE/resume.log"
