#!/bin/bash
# Label-shuffle control (planned 2026-07-06): train memorizers on
# shuffled labels for D_59 and D_45, then run conc + lattice analysis.
# Unattended: train D_59 -> train D_45 -> analyze -> write RESULTS file.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMMON="--weight_decay 1.0 --learning_rate 1e-3 --adam_betas 0.9 0.98 \
  --warmup_steps 10 --full_batch --num_layers 1 --d_model 128 --no_layernorm \
  --learned_pos_emb --nanda_init --max_seq_length 8 --result_only --no_grad_clip \
  --device cpu --print_interval 500 --eval_interval 500 --log_interval 1000 \
  --checkpoint_interval 5000 --epochs 5000"

echo "[$(date)] === D_59 label-shuffle training ==="
python -m src.train --train_file data/d59_shuffle/train.txt \
  --val_file data/d59_shuffle/test.txt $COMMON \
  --metrics_file runs/d59_shuffle/metrics.jsonl \
  --output_dir runs/d59_shuffle/checkpoints

echo "[$(date)] === D_45 label-shuffle data + training ==="
mkdir -p data/d45_shuffle runs/d45_shuffle/checkpoints
python -m instruments.make_label_shuffle \
  data/d45_30pct/train.txt data/d45_shuffle/train.txt 0
cp data/d45_30pct/test.txt data/d45_shuffle/test.txt
python -m src.train --train_file data/d45_shuffle/train.txt \
  --val_file data/d45_shuffle/test.txt $COMMON \
  --metrics_file runs/d45_shuffle/metrics.jsonl \
  --output_dir runs/d45_shuffle/checkpoints

echo "[$(date)] === analysis ==="
python -m instruments.shuffle_analysis \
  > runs/shuffle_control_RESULTS.txt 2>&1
echo "[$(date)] === DONE. Results: runs/shuffle_control_RESULTS.txt ==="
