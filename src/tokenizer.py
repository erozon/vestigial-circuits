"""
Simple tokenizer for D_n group elements.

Vocab: r0, r1, ..., r{n-1}, s0, s1, ..., s{n-1}, <PAD>, <UNK>, <BOS>, <EOS>
"""
from transformers import PreTrainedTokenizer

from src.dihedral import elements


class DihedralVocab:
    def __init__(self, n):
        self.n = n
        self.vocab = elements(n) + ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)


class DihedralTokenizer(PreTrainedTokenizer):
    def __init__(self, token_to_id, id_to_token, **kwargs):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self._vocab_size = len(token_to_id)

        super().__init__(
            pad_token="<PAD>",
            unk_token="<UNK>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            **kwargs,
        )

    def get_vocab(self):
        return self.token_to_id.copy()

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token, 0))

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.token_to_id[self.bos_token]] + token_ids + [self.token_to_id[self.eos_token]]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        import json
        import os
        vocab_file = os.path.join(save_directory, f"{filename_prefix or ''}vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.token_to_id, f)
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


def create_tokenizer(n):
    """Create a DihedralTokenizer for D_n."""
    vocab = DihedralVocab(n)
    return DihedralTokenizer(vocab.token_to_id, vocab.id_to_token)


class CyclicVocab:
    """Vocab for Z_p: z0..z{p-1} plus special tokens."""
    def __init__(self, p):
        self.p = p
        from src.cyclic import elements as cyclic_elements
        self.vocab = cyclic_elements(p) + ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)


def create_cyclic_tokenizer(p):
    """Create a tokenizer for Z_p (same DihedralTokenizer class, different vocab)."""
    vocab = CyclicVocab(p)
    return DihedralTokenizer(vocab.token_to_id, vocab.id_to_token)
