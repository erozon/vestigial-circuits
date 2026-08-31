#!/usr/bin/env bash
# Dense-checkpoint sweep (2026-07-11) -- the load-bearing experiment for the
# localization spine:
#   "The type-conditioning is ALWAYS computed (every grokked model gets rr/rs/sr/ss
#    right). What varies is whether it LOCALIZES into a low-rank ablatable override
#    circuit or stays DISTRIBUTED. When during training does that fork commit, and
#    can an EARLY checkpoint predict the endpoint?"
#
# Phase-1/rate-batch runs checkpoint every 20000 epochs -- far coarser than the
# ~12-58K grok times, so the transition is unresolved. This sweep re-trains a
# focused set with checkpoint_interval=1000 so the grokking window is densely
# sampled, then computes override_score at every checkpoint (trajectory analysis).
#
# Orders span the fork:
#   D_47  reliable localizer (seeds so far ~98-100)
#   D_61  variable-rate / pivotal (seeds gave 23 / 2 / 86 -- both modes present)
#   D_63  composite that SOMETIMES localizes (seeds gave -7 / 1 / 90)
# x10 seeds each so both modes populate. Per-order horizons cover each grok window.
#
# Chained AFTER the trimmed phase-2 rate batch (polls phase2.log for
# "RATE BATCH COMPLETE") so the two don't fight over the 8 cores. Set
# DENSE_NOWAIT=1 to skip the wait and start immediately.
# Idempotent: skips any (n,seed) whose final_model.pt exists. Trajectory analysis
# runs at the end AND anytime via:
#   python3 -m instruments.localization_trajectory
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_BASE="runs/dense_checkpoint_sweep"
PC_LOG="runs/prime_composite_sweep/phase2.log"
LOG="$OUT_BASE/dense.log"
mkdir -p "$OUT_BASE"

ORDERS="47 61 63"
SEEDS="0 1 2 3 4 5 6 7 8 9"
CKPT_INTERVAL=1000

epochs_for() {  # per-order horizon: enough to cover the grok window + tail
    case "$1" in
        47) echo 80000 ;;   # groks late (~41-58K)
        61) echo 40000 ;;   # groks ~16-21K
        63) echo 40000 ;;   # groks ~12-20K
        *)  echo 80000 ;;
    esac
}

# --- wait for the rate batch to free the cores (skip with DENSE_NOWAIT=1) ---
if [ "${DENSE_NOWAIT:-0}" != "1" ]; then
    echo "[$(date '+%F %H:%M:%S')] dense sweep queued; waiting for rate batch to finish..." | tee -a "$LOG"
    while ! grep -q "RATE BATCH COMPLETE" "$PC_LOG" 2>/dev/null; do sleep 300; done
    echo "[$(date '+%F %H:%M:%S')] rate batch done -> starting dense sweep" | tee -a "$LOG"
fi

# --- data (reuse existing d{n}_30pct) ---
for n in $ORDERS; do
    [ -f "data/d${n}_30pct/train.txt" ] || {
        echo "[$(date '+%H:%M:%S')] gen d$n" >> "$LOG"
        "$PYTHON" -m src.generate_data --n "$n" --train_fraction 0.3 --output_dir "data/d${n}_30pct" >> "$LOG" 2>&1; }
done

train_one() {
    local n="$1" seed="$2" ep
    ep="$(epochs_for "$n")"
    local out="$OUT_BASE/d${n}_seed${seed}"
    if [ -f "$out/checkpoints/final_model.pt" ]; then
        echo "[$(date '+%H:%M:%S')] skip d$n s$seed (done)" >> "$LOG"; return
    fi
    mkdir -p "$out/checkpoints"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON" -m src.train \
        --train_file "data/d${n}_30pct/train.txt" --val_file "data/d${n}_30pct/test.txt" \
        --epochs "$ep" --weight_decay 1.0 --learning_rate 1e-3 --adam_betas 0.9 0.98 \
        --warmup_steps 10 --full_batch --num_layers 1 --d_model 128 --no_layernorm \
        --learned_pos_emb --nanda_init --max_seq_length 8 --result_only --no_grad_clip \
        --print_interval 2000 --eval_interval 200 --log_interval 1000 --checkpoint_interval "$CKPT_INTERVAL" \
        --device cpu --seed "$seed" \
        --metrics_file "$out/metrics.jsonl" --output_dir "$out/checkpoints" > "$out/train.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE d$n s$seed" >> "$LOG"
}

echo "[$(date '+%F %H:%M:%S')] === dense sweep start (orders $ORDERS x seeds $SEEDS, ckpt every $CKPT_INTERVAL) ===" | tee -a "$LOG"
# seed-major so every order gets early coverage; 4-parallel waves
TASKS=""; for s in $SEEDS; do for n in $ORDERS; do TASKS="$TASKS ${n}:${s}"; done; done
i=0
for t in $TASKS; do
    n="${t%%:*}"; s="${t##*:}"
    train_one "$n" "$s" &
    i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done
wait
echo "[$(date '+%F %H:%M:%S')] === dense training done -> trajectory analysis ===" | tee -a "$LOG"

# [not in this release: localization_trajectory is exploratory]
# "$PYTHON" -m instruments.localization_trajectory > "$OUT_BASE/TRAJECTORY_RESULTS.txt" 2>&1
echo "[$(date '+%F %H:%M:%S')] === DENSE SWEEP COMPLETE -> $OUT_BASE/TRAJECTORY_RESULTS.txt ===" | tee -a "$LOG"
