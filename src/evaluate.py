"""
Evaluation for D_n group multiplication model.

Computes argmax accuracy: for each exercise "g h result", feed "g h" as prompt
(with BOS), take argmax of logits at the result position, compare to expected.
"""
import torch
from typing import Dict, List


def _prepare_eval_batch(exercise_strings, tokenizer, device, max_examples=0):
    """Parse exercises and build a batched input tensor.

    Returns (input_ids, expected_ids, quadrants) where:
        input_ids: (N, 3) tensor of [BOS, tok1, tok2]
        expected_ids: (N,) tensor of expected result token ids
        quadrants: list of 2-char strings like 'rr', 'rs', 'sr', 'ss'
    """
    bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    examples = exercise_strings
    if max_examples > 0:
        examples = examples[:max_examples]

    all_input_ids = []
    all_expected_ids = []
    all_quadrants = []

    for exercise in examples:
        toks = exercise.strip().split()
        if len(toks) < 2:
            continue
        expected_id = tokenizer.token_to_id.get(toks[-1])
        if expected_id is None:
            continue

        prompt_tokens = tokenizer.tokenize(' '.join(toks[:-1]))
        token_ids = [bos_id] + tokenizer.convert_tokens_to_ids(prompt_tokens)
        all_input_ids.append(token_ids)
        all_expected_ids.append(expected_id)
        all_quadrants.append(toks[0][0] + toks[1][0])

    input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=device)
    expected_ids = torch.tensor(all_expected_ids, dtype=torch.long, device=device)
    return input_ids, expected_ids, all_quadrants


def eval_argmax_accuracy(model, tokenizer, exercise_strings: List[str], device,
                         max_examples: int = 0) -> Dict:
    """
    Evaluate accuracy via argmax on result token (batched).

    Args:
        model: Trained DihedralModel
        tokenizer: DihedralTokenizer
        exercise_strings: List of "g h result" strings
        device: torch device
        max_examples: Max exercises to evaluate (0=all)

    Returns:
        Dict with 'accuracy', 'correct', 'total'
    """
    model.eval()
    input_ids, expected_ids, _ = _prepare_eval_batch(
        exercise_strings, tokenizer, device, max_examples)
    total = len(expected_ids)
    if total == 0:
        return {'accuracy': 0, 'correct': 0, 'total': 0}

    with torch.no_grad():
        logits = model(input_ids)
        preds = logits[:, -1, :].argmax(dim=-1)
        correct = (preds == expected_ids).sum().item()

    return {
        'accuracy': correct / total,
        'correct': correct,
        'total': total
    }


def eval_quadrant_accuracy(model, tokenizer, exercise_strings: List[str], device,
                           max_examples: int = 0) -> Dict:
    """
    Evaluate accuracy via argmax on result token, with per-quadrant breakdown (batched).

    Quadrants: rr (rot*rot), rs (rot*ref), sr (ref*rot), ss (ref*ref).

    Returns:
        Dict with 'accuracy', 'correct', 'total', plus per-quadrant
        '{q}_acc', '{q}_total' for q in {rr, rs, sr, ss}.
    """
    model.eval()
    input_ids, expected_ids, quadrants = _prepare_eval_batch(
        exercise_strings, tokenizer, device, max_examples)
    total = len(expected_ids)
    if total == 0:
        result = {'accuracy': 0, 'correct': 0, 'total': 0}
        for q in ('rr', 'rs', 'sr', 'ss'):
            result[f'{q}_acc'] = 0
            result[f'{q}_total'] = 0
        return result

    with torch.no_grad():
        logits = model(input_ids)
        preds = logits[:, -1, :].argmax(dim=-1)
        is_correct = (preds == expected_ids)

    correct = is_correct.sum().item()
    is_correct_cpu = is_correct.cpu()

    quad_correct = {'rr': 0, 'rs': 0, 'sr': 0, 'ss': 0}
    quad_total = {'rr': 0, 'rs': 0, 'sr': 0, 'ss': 0}
    for i, q in enumerate(quadrants):
        if q not in quad_total:
            # Non-dihedral group (e.g. Z_p, where every quadrant is 'zz').
            # Per-quadrant breakdown is not meaningful here.
            continue
        quad_total[q] += 1
        if is_correct_cpu[i]:
            quad_correct[q] += 1

    result = {
        'accuracy': correct / total,
        'correct': correct,
        'total': total,
    }
    for q in ('rr', 'rs', 'sr', 'ss'):
        result[f'{q}_acc'] = quad_correct[q] / quad_total[q] if quad_total[q] > 0 else 0
        result[f'{q}_total'] = quad_total[q]

    return result


def read_exercises(path) -> List[str]:
    """Read exercise strings from file."""
    exercises = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                exercises.append(line)
    return exercises
