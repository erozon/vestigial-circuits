"""
Training loop for D_n group multiplication grokking experiment.

Simplified from grok-prop-logic train_grokking.py. Single mode: full batch
with Nanda-aligned hyperparameters.

Usage:
    python -m src.train \
        --train_file data/d59/train.txt \
        --val_file data/d59/test.txt \
        --epochs 40000 \
        --weight_decay 1.0 \
        --learning_rate 1e-3 \
        --full_batch \
        --num_layers 1 \
        --d_model 128 \
        --no_layernorm \
        --learned_pos_emb \
        --nanda_init \
        --max_seq_length 8 \
        --result_only \
        --metrics_file runs/d59/metrics.jsonl \
        --output_dir runs/d59/checkpoints
"""
import json
import platform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
from pathlib import Path
from tqdm import tqdm

from src.model import DihedralModel
from src.dataset import DihedralDataset
from src.tokenizer import create_tokenizer, create_cyclic_tokenizer
from src.evaluate import eval_argmax_accuracy, eval_quadrant_accuracy, read_exercises


def _get_best_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def log_metrics(metrics_file, metrics_dict):
    if metrics_file:
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics_dict) + '\n')


def compute_val_loss(model, val_input_ids, val_target_ids, val_attention_mask,
                     criterion):
    model.eval()
    with torch.no_grad():
        logits = model(val_input_ids, val_attention_mask)
        loss = criterion(logits.view(-1, logits.shape[-1]), val_target_ids.view(-1))
    return loss.item()


def _read_group_from_data(train_file):
    """Infer (group_type, order) from the data directory's split_info.json.

    Returns ('dihedral', n) for D_n or ('cyclic', p) for Z_p.
    """
    split_info_path = Path(train_file).parent / 'split_info.json'
    if split_info_path.exists():
        with open(split_info_path) as f:
            info = json.load(f)
        if 'p' in info:
            return 'cyclic', info['p']
        return 'dihedral', info['n']
    # Fallback: sniff element prefixes
    max_idx = 0
    has_z = has_rs = False
    with open(train_file, 'r') as f:
        for line in f:
            for tok in line.strip().split():
                if not tok or not tok[1:].isdigit():
                    continue
                if tok[0] == 'z':
                    has_z = True
                if tok[0] in ('r', 's'):
                    has_rs = True
                max_idx = max(max_idx, int(tok[1:]))
    if has_z and not has_rs:
        return 'cyclic', max_idx + 1
    return 'dihedral', max_idx + 1


def main():
    parser = argparse.ArgumentParser(
        description='Training for D_n group multiplication grokking')

    # Data
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)

    # Training
    parser.add_argument('--epochs', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (ignored if --full_batch)')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--full_batch', action='store_true')
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.98],
                        metavar=('BETA1', 'BETA2'))
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        choices=['constant', 'cosine'],
                        help='LR schedule after warmup (default: constant)')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='Minimum LR for cosine schedule (default: 0)')
    parser.add_argument('--no_grad_clip', action='store_true',
                        help='Disable gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--result_only', action='store_true',
                        help='Only compute loss on the result token')

    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--max_seq_length', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--no_layernorm', action='store_true')
    parser.add_argument('--learned_pos_emb', action='store_true')
    parser.add_argument('--nanda_init', action='store_true')

    # Resume
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for model init and data shuffling (default: 42)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--freeze_unembed', type=str, default=None,
                        help='Path to a donor checkpoint whose fc_out (unembedding) '
                             'weights are copied in and held FIXED for the whole run. '
                             'Everything else trains normally from --seed init. '
                             'Used by the unembedding-transplant experiment.')

    # Logging
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--print_interval', type=int, default=20)
    parser.add_argument('--checkpoint_interval', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--metrics_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='runs/dihedral/checkpoints')

    # Early stopping
    parser.add_argument('--early_stop', action='store_true',
                        help='Enable early stopping checks')
    parser.add_argument('--early_stop_val_acc', type=float, default=0.99,
                        help='Stop when val_acc >= this for --early_stop_patience consecutive evals')
    parser.add_argument('--early_stop_patience', type=int, default=2000,
                        help='Epochs val_acc must stay above threshold to trigger stop')
    parser.add_argument('--early_stop_memo_epoch', type=int, default=15000,
                        help='Stop if train_acc < 0.3 after this many epochs')
    parser.add_argument('--early_stop_gen_epoch', type=int, default=60000,
                        help='Stop if val_acc < 0.05 after this many epochs')

    # Device
    parser.add_argument('--device', type=str, default=_get_best_device())

    args = parser.parse_args()

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.metrics_file:
        Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    # Recorded because floating-point reduction order -- and so bit-exact
    # reproduction -- depends on all four of these.
    print(f"Environment: python {platform.python_version()}, torch "
          f"{torch.__version__}, {platform.platform()}, "
          f"{torch.get_num_threads()} threads")

    # Infer group from data
    group_type, n = _read_group_from_data(args.train_file)
    if group_type == 'cyclic':
        print(f"Detected Z_{n} (order {n})")
        tokenizer = create_cyclic_tokenizer(n)
    else:
        print(f"Detected D_{n} (order {2*n})")
        tokenizer = create_tokenizer(n)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    pad_id = tokenizer.token_to_id[tokenizer.pad_token]

    # Create datasets
    print(f"Loading training data from {args.train_file}...")
    train_dataset = DihedralDataset(args.train_file, tokenizer, args.max_seq_length,
                                    result_only=args.result_only)

    val_dataset = None
    if Path(args.val_file).exists():
        print(f"Loading validation data from {args.val_file}...")
        val_dataset = DihedralDataset(args.val_file, tokenizer, args.max_seq_length,
                                      result_only=args.result_only)

    # Load raw exercises for argmax accuracy eval
    train_raw = read_exercises(args.train_file)
    val_raw = read_exercises(args.val_file) if val_dataset else []

    # Cache data on device
    if args.full_batch:
        print(f"Using full batch training (batch_size={len(train_dataset)})")
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset),
                                  shuffle=False, num_workers=0)
        _batch = next(iter(train_loader))
        train_input_ids = _batch['input_ids'].to(device)
        train_target_ids = _batch['target_ids'].to(device)
        train_attention_mask = _batch['attention_mask'].to(device)
        del _batch, train_loader
    else:
        print(f"Using minibatch training (batch_size={args.batch_size})")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),
                                shuffle=False, num_workers=0)
        _vbatch = next(iter(val_loader))
        val_input_ids = _vbatch['input_ids'].to(device)
        val_target_ids = _vbatch['target_ids'].to(device)
        val_attention_mask = _vbatch['attention_mask'].to(device)
        del _vbatch, val_loader

    # Reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create model
    print("Creating model...")
    use_layernorm = not args.no_layernorm
    model = DihedralModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
        use_layernorm=use_layernorm,
        learned_pos_emb=args.learned_pos_emb,
        nanda_init=args.nanda_init,
    )
    print(f"Model: Transformer, layers={args.num_layers}, d_model={args.d_model}, "
          f"heads={args.num_heads}, d_ff={args.d_ff}")
    print(f"  LayerNorm: {use_layernorm}, Learned pos emb: {args.learned_pos_emb}, "
          f"Nanda init: {args.nanda_init}")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Unembedding transplant: copy a donor's fc_out and hold it fixed.
    if args.freeze_unembed:
        donor = torch.load(args.freeze_unembed, map_location=device, weights_only=False)
        donor_sd = donor.get('model_state_dict', donor) if isinstance(donor, dict) else donor
        W = donor_sd['fc_out.weight']
        if W.shape != model.fc_out.weight.shape:
            raise ValueError(f"Donor unembedding {tuple(W.shape)} does not match model "
                             f"{tuple(model.fc_out.weight.shape)} — wrong group order?")
        if 'fc_out.bias' in donor_sd and model.fc_out.bias is not None:
            model.fc_out.bias.data.copy_(donor_sd['fc_out.bias'].to(device))
            model.fc_out.bias.requires_grad = False
        with torch.no_grad():
            model.fc_out.weight.copy_(W.to(device))
        model.fc_out.weight.requires_grad = False
        print(f"Unembedding FROZEN from donor: {args.freeze_unembed}")

    # Optimizer and loss (frozen params are excluded so weight decay cannot touch them)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.learning_rate,
                      weight_decay=args.weight_decay,
                      betas=tuple(args.adam_betas))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # LR schedule: warmup then constant or cosine decay
    scheduler = None
    if args.lr_schedule != 'constant' or args.warmup_steps > 0:
        warmup = args.warmup_steps
        total = args.epochs
        lr_min_ratio = args.lr_min / args.learning_rate if args.learning_rate > 0 else 0

        if args.lr_schedule == 'cosine':
            import math
            def lr_lambda(step):
                # Linear warmup
                if step < warmup:
                    return (step + 1) / warmup
                # Cosine decay from 1.0 to lr_min_ratio
                progress = (step - warmup) / max(total - warmup, 1)
                return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            def lr_lambda(step):
                return min((step + 1) / warmup, 1.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume_from:
        print(f"\nResuming from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint with optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"  Loaded full checkpoint: epoch {start_epoch}, best_val_acc {best_val_acc:.4f}")
        else:
            raise ValueError(f"Invalid checkpoint format in {args.resume_from} — expected full checkpoint with 'model_state_dict' key")

    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Adam betas: {tuple(args.adam_betas)}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Grad clipping: {not args.no_grad_clip}" +
          (f" (max_norm={args.max_grad_norm})" if not args.no_grad_clip else ""))
    lr_detail = f" (min={args.lr_min})" if args.lr_schedule == 'cosine' else ""
    print(f"  LR schedule: {args.lr_schedule}{lr_detail}")
    print(f"  Full batch: {args.full_batch}")
    print(f"  Result only: {args.result_only}")
    if start_epoch > 0:
        print(f"  Resuming from epoch: {start_epoch}")
        print(f"  Remaining epochs: {args.epochs - start_epoch}")

    # Early stopping state
    grok_since = None

    # Training loop
    print("\nStarting training...")

    pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch",
                initial=start_epoch, total=args.epochs)
    for epoch in pbar:
        model.train()
        if args.full_batch:
            logits = model(train_input_ids, train_attention_mask)
            loss = criterion(logits.view(-1, logits.shape[-1]),
                             train_target_ids.view(-1))
            # MPS sync: materialize loss before backward
            train_loss = loss.item()
            if torch.isnan(loss):
                tqdm.write(f"NaN loss at epoch {epoch}, stopping training")
                break
            optimizer.zero_grad()
            loss.backward()
            clip_norm = args.max_grad_norm if not args.no_grad_clip else float('inf')
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        max_norm=clip_norm).item()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            total_loss = 0.0
            num_batches = 0
            grad_norm = 0.0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, logits.shape[-1]),
                                 target_ids.view(-1))
                batch_loss = loss.item()
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                            max_norm=clip_norm).item()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += batch_loss
                num_batches += 1
            train_loss = total_loss / num_batches if num_batches > 0 else float('nan')

        pbar.set_postfix(loss=f"{train_loss:.4f}", gnorm=f"{grad_norm:.1f}")

        if args.print_interval and (epoch + 1) % args.print_interval == 0:
            tqdm.write(f"Epoch {epoch + 1:>5}/{args.epochs} | loss {train_loss:.4f} | gnorm {grad_norm:.1f}")

        # Evaluate at intervals
        if (epoch + 1) % args.eval_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            metrics = {'epoch': epoch + 1, 'train_loss': train_loss, 'grad_norm': grad_norm,
                       'lr': current_lr}

            if val_dataset:
                val_loss = compute_val_loss(model, val_input_ids, val_target_ids,
                                            val_attention_mask, criterion)
                metrics['val_loss'] = val_loss

            # Argmax accuracy
            if train_raw:
                train_result = eval_argmax_accuracy(
                    model, tokenizer, train_raw, device, max_examples=0)
                metrics['train_acc'] = train_result['accuracy']

            if val_raw:
                val_result = eval_quadrant_accuracy(
                    model, tokenizer, val_raw, device, max_examples=0)
                metrics['val_acc'] = val_result['accuracy']
                for q in ('rr', 'rs', 'sr', 'ss'):
                    metrics[f'val_{q}_acc'] = val_result[f'{q}_acc']

                if val_result['accuracy'] > best_val_acc:
                    best_val_acc = val_result['accuracy']
                    model_path = f"{args.output_dir}/best_model.pt"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_acc': best_val_acc,
                        **(({'scheduler_state_dict': scheduler.state_dict()}
                            if scheduler is not None else {})),
                    }, model_path)

                metrics['best_val_acc'] = best_val_acc

            log_metrics(args.metrics_file, metrics)

            postfix = {'loss': f"{train_loss:.4f}"}
            if 'train_acc' in metrics:
                postfix['tr_acc'] = f"{metrics['train_acc']:.3f}"
            if 'val_acc' in metrics:
                postfix['val_acc'] = f"{metrics['val_acc']:.3f}"
            if 'best_val_acc' in metrics:
                postfix['best'] = f"{metrics['best_val_acc']:.3f}"
            pbar.set_postfix(**postfix)

            if (epoch + 1) % args.log_interval == 0:
                tqdm.write(f"Epoch {epoch + 1}/{args.epochs} - "
                      f"Train Loss: {train_loss:.4f}" +
                      (f" - Val Loss: {metrics.get('val_loss', 0):.4f}" if val_dataset else "") +
                      (f" - Train Acc: {metrics['train_acc']:.4f}" if 'train_acc' in metrics else "") +
                      (f" - Val Acc: {metrics['val_acc']:.4f}" if 'val_acc' in metrics else "") +
                      (f" - Best Val Acc: {best_val_acc:.4f}" if 'val_acc' in metrics else ""))

            # Early stopping checks
            if args.early_stop:
                current_epoch = epoch + 1
                stop_reason = None

                # Grokked: val_acc above threshold for patience epochs
                if 'val_acc' in metrics and metrics['val_acc'] >= args.early_stop_val_acc:
                    if grok_since is None:
                        grok_since = current_epoch
                    elif current_epoch - grok_since >= args.early_stop_patience:
                        stop_reason = (f"GROKKED: val_acc >= {args.early_stop_val_acc} "
                                       f"for {args.early_stop_patience} epochs (since epoch {grok_since})")
                else:
                    grok_since = None

                # Failed to memorize
                if ('train_acc' in metrics and current_epoch >= args.early_stop_memo_epoch
                        and metrics['train_acc'] < 0.3):
                    stop_reason = (f"FAILED TO MEMORIZE: train_acc={metrics['train_acc']:.3f} "
                                   f"at epoch {current_epoch}")

                # No generalization
                if ('val_acc' in metrics and current_epoch >= args.early_stop_gen_epoch
                        and best_val_acc < 0.05):
                    stop_reason = (f"NO GENERALIZATION: best_val_acc={best_val_acc:.3f} "
                                   f"at epoch {current_epoch}")

                if stop_reason:
                    tqdm.write(f"Early stop: {stop_reason}")
                    break

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = f"{args.output_dir}/checkpoint_epoch_{epoch + 1}.pt"
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            tqdm.write(f"  Saved checkpoint to {checkpoint_path}")

    # Save final model (full checkpoint)
    final_path = f"{args.output_dir}/final_model.pt"
    final_data = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }
    if scheduler is not None:
        final_data['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(final_data, final_path)
    print(f"\nTraining complete!")
    print(f"  Final model: {final_path}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
