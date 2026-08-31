"""
Generate train/test data for D_n group multiplication grokking experiment.

Enumerates all (2n)^2 ordered pairs, computes products, splits 30/70.

Usage:
    python -m src.generate_data --n 59 --train_fraction 0.3 --output_dir data/d59
"""
import argparse
import json
import random
from pathlib import Path

from src.dihedral import elements, multiply, multiplication_table, verify_group_axioms


def main():
    parser = argparse.ArgumentParser(description='Generate D_n multiplication data')
    parser.add_argument('--n', type=int, default=59,
                        help='Order parameter n for D_n (group has 2n elements)')
    parser.add_argument('--train_fraction', type=float, default=0.3,
                        help='Fraction of pairs for training (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')
    parser.add_argument('--output_dir', type=str, default='data/d59',
                        help='Output directory')
    args = parser.parse_args()

    n = args.n
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating data for D_{n} (order {2*n}, {(2*n)**2} pairs)")

    # Verify group axioms
    print("Verifying group axioms...")
    verify_group_axioms(n)
    print("  All axioms verified.")

    # Generate all pairs
    elems = elements(n)
    table = multiplication_table(n)

    all_exercises = []
    for g in elems:
        for h in elems:
            result = table[(g, h)]
            all_exercises.append(f"{g} {h} {result}")

    print(f"Total exercises: {len(all_exercises)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_exercises)

    split_idx = int(len(all_exercises) * args.train_fraction)
    train_exercises = all_exercises[:split_idx]
    test_exercises = all_exercises[split_idx:]

    print(f"Train: {len(train_exercises)} ({len(train_exercises)/len(all_exercises)*100:.1f}%)")
    print(f"Test:  {len(test_exercises)} ({len(test_exercises)/len(all_exercises)*100:.1f}%)")

    # Save files
    with open(output_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_exercises) + '\n')

    with open(output_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_exercises) + '\n')

    with open(output_dir / 'elements.txt', 'w') as f:
        f.write('\n'.join(elems) + '\n')

    # Save multiplication table as JSON
    table_json = {f"{g},{h}": v for (g, h), v in table.items()}
    with open(output_dir / 'multiplication_table.json', 'w') as f:
        json.dump(table_json, f, indent=2)

    # Save split info
    split_info = {
        'n': n,
        'group_order': 2 * n,
        'total_pairs': len(all_exercises),
        'train_count': len(train_exercises),
        'test_count': len(test_exercises),
        'train_fraction': args.train_fraction,
        'seed': args.seed,
    }
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nFiles saved to {output_dir}/")
    print(f"  train.txt ({len(train_exercises)} exercises)")
    print(f"  test.txt ({len(test_exercises)} exercises)")
    print(f"  elements.txt ({len(elems)} elements)")
    print(f"  multiplication_table.json")
    print(f"  split_info.json")


if __name__ == '__main__':
    main()
