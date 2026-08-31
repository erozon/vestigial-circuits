#!/usr/bin/env bash
# Prime-vs-composite mechanism sweep (2026-07-08).
#
# Tests whether the one-sided output-type override (clean in prime D_59, absent in
# composite D_45) is a PRIME-vs-COMPOSITE distinction. Trains size-matched primes
# and composites with the frozen Nanda/D_45 recipe (80K epochs), 3 seeds each,
# 4-way parallel (OMP=2 on 8 cores). Idempotent: skips any (n,seed) whose
# final_model.pt already exists. Analysis runs at the end AND can be run anytime
# via:  python3 -m instruments.prime_composite_analyze
#
# Size-matched pairs: 47/49, 53/55, 61/63  (+51 as a semiprime).  Anchors already
# trained: D_59 (prime), D_45 (composite).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_BASE="runs/prime_composite_sweep"
LOG="$OUT_BASE/sweep.log"
mkdir -p "$OUT_BASE"

PRIMES="47 53 61"
COMPOSITES="49 51 55 63"
SEEDS="0 1 2"
EPOCHS=80000

echo "[$(date '+%F %H:%M:%S')] === prime/composite sweep start ===" | tee -a "$LOG"

# --- 1. data ---
for n in $PRIMES $COMPOSITES; do
    d="data/d${n}_30pct"
    if [ ! -f "$d/train.txt" ]; then
        echo "[$(date '+%H:%M:%S')] gen data d$n" | tee -a "$LOG"
        "$PYTHON" -m src.generate_data --n "$n" --train_fraction 0.3 --output_dir "$d" >> "$LOG" 2>&1
    fi
done

# --- 2. train ---
train_one() {
    local n="$1" seed="$2"
    local out="$OUT_BASE/d${n}_seed${seed}"
    if [ -f "$out/checkpoints/final_model.pt" ]; then
        echo "[$(date '+%H:%M:%S')] skip d$n s$seed (done)" >> "$LOG"; return
    fi
    mkdir -p "$out/checkpoints"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON" -m src.train \
        --train_file "data/d${n}_30pct/train.txt" --val_file "data/d${n}_30pct/test.txt" \
        --epochs "$EPOCHS" --weight_decay 1.0 --learning_rate 1e-3 --adam_betas 0.9 0.98 \
        --warmup_steps 10 --full_batch --num_layers 1 --d_model 128 --no_layernorm \
        --learned_pos_emb --nanda_init --max_seq_length 8 --result_only --no_grad_clip \
        --print_interval 2000 --eval_interval 200 --log_interval 1000 --checkpoint_interval 20000 \
        --device cpu --seed "$seed" \
        --metrics_file "$out/metrics.jsonl" --output_dir "$out/checkpoints" \
        > "$out/train.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE d$n s$seed" >> "$LOG"
}

# interleave prime/composite so each wave of 4 is balanced; seed0 of all orders first
ORDER="47 49 53 55 61 63 51"
TASKS=""
for seed in $SEEDS; do for n in $ORDER; do TASKS="$TASKS ${n}:${seed}"; done; done

i=0
for t in $TASKS; do
    n="${t%%:*}"; s="${t##*:}"
    train_one "$n" "$s" &
    i=$((i + 1))
    [ $((i % 4)) -eq 0 ] && wait
done
wait
echo "[$(date '+%F %H:%M:%S')] === all training done ===" | tee -a "$LOG"

# --- 3. analysis ---
"$PYTHON" -m instruments.prime_composite_analyze > "$OUT_BASE/RESULTS.txt" 2>&1
echo "[$(date '+%F %H:%M:%S')] === analysis -> $OUT_BASE/RESULTS.txt ===" | tee -a "$LOG"
