#!/usr/bin/env bash
# Prime/composite sweep -- PHASE 2 (TRIMMED 2026-07-11): rate batch ONLY.
#
# The original phase-2 big-orders battery (64/77/79/81/105/119/121/127, 120K-220K
# epochs) was CUT. It served the old prime-vs-composite framing and is the wrong
# compute for the localization spine ("when does the type-gating circuit localize
# into an ablatable override vs stay distributed, and can an early checkpoint
# predict which?"). Its 20000-epoch checkpoint spacing can't resolve the grokking
# transition, and the big orders only broaden a rate table we don't need broadened.
#
# What survives is the RATE BATCH -- extra seeds on the two pivotal orders, to turn
# "sometimes localizes" into a measured per-order RATE:
#   D_61 x5  (variable-rate: seeds so far gave 23 / 2 / 86 -- is it bimodal?)
#   D_47 x2  (reliable-localizer control)
# The straggler-resume stage was also dropped (not part of the rate batch).
#
# On completion writes "RATE BATCH COMPLETE" to phase2.log so
# run_dense_checkpoint_sweep.sh (the load-bearing experiment) starts only once the
# cores are free -- no oversubscription of the 8-core CPU box.
#
# Runs AFTER phase 1 (polls sweep.log). Idempotent: skips any (n,seed) done.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_BASE="runs/prime_composite_sweep"
LOG="$OUT_BASE/phase2.log"
mkdir -p "$OUT_BASE"

echo "[$(date '+%F %H:%M:%S')] phase2 (trimmed=rate-batch) queued; waiting for phase1 to finish..." | tee -a "$LOG"
while ! grep -q "all training done" "$OUT_BASE/sweep.log" 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %H:%M:%S')] phase1 done -> starting rate batch" | tee -a "$LOG"

# data for 61/47 already exists from phase1; regenerate only if missing
for n in 61 47; do
    [ -f "data/d${n}_30pct/train.txt" ] || \
        "$PYTHON" -m src.generate_data --n "$n" --train_fraction 0.3 --output_dir "data/d${n}_30pct" >> "$LOG" 2>&1
done

train_one() {  # n seed epochs
    local n="$1" seed="$2" ep="$3"
    local out="$OUT_BASE/d${n}_seed${seed}"
    [ -f "$out/checkpoints/final_model.pt" ] && { echo "[$(date '+%H:%M:%S')] skip d$n s$seed" >> "$LOG"; return; }
    mkdir -p "$out/checkpoints"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON" -m src.train \
        --train_file "data/d${n}_30pct/train.txt" --val_file "data/d${n}_30pct/test.txt" \
        --epochs "$ep" --weight_decay 1.0 --learning_rate 1e-3 --adam_betas 0.9 0.98 \
        --warmup_steps 10 --full_batch --num_layers 1 --d_model 128 --no_layernorm \
        --learned_pos_emb --nanda_init --max_seq_length 8 --result_only --no_grad_clip \
        --print_interval 2000 --eval_interval 200 --log_interval 1000 --checkpoint_interval 20000 \
        --device cpu --seed "$seed" \
        --metrics_file "$out/metrics.jsonl" --output_dir "$out/checkpoints" > "$out/train.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE d$n s$seed" >> "$LOG"
}

RATE_TASKS="61:3:60000 61:4:60000 61:5:60000 61:6:60000 61:7:60000 47:3:100000 47:4:100000"
echo "[$(date '+%H:%M:%S')] rate-batch (D_61 x5, D_47 x2) starting" | tee -a "$LOG"
i=0
for t in $RATE_TASKS; do
    n="${t%%:*}"; rest="${t#*:}"; s="${rest%%:*}"; ep="${rest#*:}"
    train_one "$n" "$s" "$ep" &
    i=$((i + 1)); [ $((i % 4)) -eq 0 ] && wait
done
wait
echo "[$(date '+%H:%M:%S')] rate-batch done" | tee -a "$LOG"

"$PYTHON" -m instruments.prime_composite_analyze > "$OUT_BASE/RESULTS_ratebatch.txt" 2>&1
echo "[$(date '+%F %H:%M:%S')] === RATE BATCH COMPLETE -> $OUT_BASE/RESULTS_ratebatch.txt ===" | tee -a "$LOG"
