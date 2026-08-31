"""
Dataset class for D_n group multiplication exercises.

Exercise format: "r3 s1 s4" (3 space-separated tokens: element1 element2 result)
With BOS prepended: 4 tokens total. Loss computed only on result position.
"""
import torch
from torch.utils.data import Dataset


class DihedralDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=8, result_only=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.result_only = result_only
        self.examples = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(line)

        print(f"Loaded {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        bos_id = self.tokenizer.token_to_id[self.tokenizer.bos_token]
        eos_id = self.tokenizer.token_to_id[self.tokenizer.eos_token]
        pad_id = self.tokenizer.token_to_id[self.tokenizer.pad_token]

        input_ids = [bos_id] + token_ids
        target_ids = token_ids + [eos_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            target_ids = target_ids[:self.max_length]

        # Mask non-result positions: only compute loss on the result token
        if self.result_only:
            result_pos = len(token_ids) - 1  # last token before EOS
            target_ids = [pad_id] * result_pos + [target_ids[result_pos]] + [pad_id]
            target_ids = target_ids[:len(input_ids)]

        # Pad to max_length
        pad_length = self.max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_length
        target_ids = target_ids + [pad_id] * pad_length

        attention_mask = [1] * (len(input_ids) - pad_length) + [0] * pad_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
