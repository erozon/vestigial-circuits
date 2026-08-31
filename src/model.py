"""
Transformer model for dihedral group multiplication (D_n grokking).

Ported from grok-prop-logic PropLogicModel with all Nanda features:
- nanda_init (randn/sqrt(fan_in))
- no_layernorm option
- learned_pos_emb option
- Pre-norm transformer blocks
- Causal masking
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (Nanda-style)."""
    def __init__(self, max_seq_length, d_model, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, attn_bias=True):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model, bias=attn_bias)
        self.key = nn.Linear(d_model, d_model, bias=attn_bias)
        self.value = nn.Linear(d_model, d_model, bias=attn_bias)
        self.fc_out = nn.Linear(d_model, d_model, bias=attn_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -1e9)
            else:
                scores = scores + mask

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        output = self.fc_out(context)
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_layernorm=True,
                 attn_bias=True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, attn_bias=attn_bias)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        attn_output, attn_weights = self.attention(x, x, x, mask)
        if self.use_layernorm:
            x = self.norm1(x + self.dropout(attn_output))
        else:
            x = x + self.dropout(attn_output)

        ff_output = self.feed_forward(x)
        if self.use_layernorm:
            x = self.norm2(x + self.dropout(ff_output))
        else:
            x = x + self.dropout(ff_output)

        if return_attention:
            return x, attn_weights
        return x


class DihedralModel(nn.Module):
    """
    Transformer model for D_n group multiplication.

    Args:
        vocab_size: Size of vocabulary (2n + 4 special tokens)
        d_model: Dimension of model
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
        use_layernorm: Whether to use LayerNorm
        learned_pos_emb: Use learned positional embeddings
        nanda_init: Use Nanda-style weight initialization
    """
    def __init__(self, vocab_size, d_model=128, num_layers=1, num_heads=4,
                 d_ff=512, max_seq_length=8, dropout=0.0,
                 use_layernorm=True, learned_pos_emb=False,
                 nanda_init=False):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        attn_bias = not nanda_init
        output_bias = not nanda_init
        self.embed_scale = 1.0 if nanda_init else (math.sqrt(d_model) if use_layernorm else 1.0)

        self.embedding = nn.Embedding(vocab_size, d_model)
        if learned_pos_emb:
            self.pos_encoding = LearnedPositionalEmbedding(max_seq_length, d_model, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_layernorm,
                             attn_bias=attn_bias)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size, bias=output_bias)
        self.dropout = nn.Dropout(dropout)

        if nanda_init:
            self._nanda_init()

    def _nanda_init(self):
        """Initialize weights following Nanda et al. (2023): W ~ N(0, 1/sqrt(d_in))."""
        d = self.d_model
        nn.init.normal_(self.embedding.weight, std=1 / math.sqrt(d))
        if hasattr(self.pos_encoding, 'embedding'):
            nn.init.normal_(self.pos_encoding.embedding.weight, std=1 / math.sqrt(d))
        for block in self.transformer_blocks:
            for layer in [block.attention.query, block.attention.key,
                          block.attention.value, block.attention.fc_out]:
                nn.init.normal_(layer.weight, std=1 / math.sqrt(d))
            d_ff = block.feed_forward.fc1.out_features
            nn.init.normal_(block.feed_forward.fc1.weight, std=1 / math.sqrt(d))
            nn.init.zeros_(block.feed_forward.fc1.bias)
            nn.init.normal_(block.feed_forward.fc2.weight, std=1 / math.sqrt(d_ff))
            nn.init.zeros_(block.feed_forward.fc2.bias)
        nn.init.normal_(self.fc_out.weight, std=1 / math.sqrt(d))

    def forward(self, input_ids, attention_mask=None, return_attention=False):
        x = self.embedding(input_ids) * self.embed_scale
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Causal mask (use -1e9 instead of -inf for MPS compatibility)
        _MASK_VALUE = -1e9
        if attention_mask is not None:
            seq_length = input_ids.shape[1]
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), _MASK_VALUE), diagonal=1
            ).to(input_ids.device)
            padding_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            padding_mask = padding_mask.masked_fill(attention_mask == 0, _MASK_VALUE)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            mask = causal_mask.unsqueeze(0) + padding_mask
        else:
            seq_length = input_ids.shape[1]
            mask = torch.triu(
                torch.full((seq_length, seq_length), _MASK_VALUE), diagonal=1
            ).to(input_ids.device)

        attention_weights_all = []
        for block in self.transformer_blocks:
            if return_attention:
                x, attn_w = block(x, mask, return_attention=True)
                attention_weights_all.append(attn_w)
            else:
                x = block(x, mask)

        logits = self.fc_out(x)

        if return_attention:
            return logits, attention_weights_all
        return logits
