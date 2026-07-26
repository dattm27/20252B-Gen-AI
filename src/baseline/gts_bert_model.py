"""BERT-based Grid Tagging Scheme (GTS) model for ASTE -- a genuine neural, non-generative
baseline, grounded in Wu et al. (Findings of EMNLP 2020), "Grid Tagging Scheme for
Aspect-oriented Fine-grained Opinion Extraction" (https://aclanthology.org/2020.findings-emnlp.234/).

Needs `torch`/`transformers` and a GPU to train in reasonable time -- meant to run on Kaggle via
notebooks/train-gts-bert-for-aste-*.ipynb, mirroring the T5 notebooks' local/Kaggle split: the
grid-building/decoding logic (src/data/gts_grid.py) is pure Python and tested locally
(tests/test_gts_grid.py); only this model needs GPU, and only inside the notebook.

Architecture (paper Section 3.1, BERT variant): BERT encodes the sentence into per-token hidden
states h_1..h_n; for every word-pair (i, j), a pairwise representation r_ij is built and
classified into one of 6 grid tags (N/A/O/POS/NEU/NEG, see src/data/gts_grid.py). The paper does
not fully specify its attention-layer formula for r_ij in the text we had access to, so this is
an independent design choice -- r_ij = MLP(concat(h_i, h_j, h_i * h_j)), a standard pairwise
feature construction (same spirit as biaffine parsers) -- not a literal reproduction of the
authors' code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.data.gts_grid import TAGS


class GTSBertModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(TAGS)),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns tag logits of shape (batch, seq_len, seq_len, num_tags). Only the upper
        triangle (i <= j) is meaningful for a valid GTS grid -- callers should restrict the loss
        and decoding to it (see `src.data.gts_grid.IGNORE_INDEX` for the matching label
        convention: fill the lower triangle and special-token rows/cols with -100)."""
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden_states = self.dropout(hidden_states)

        batch, seq_len, hidden_dim = hidden_states.shape
        h_i = hidden_states.unsqueeze(2).expand(batch, seq_len, seq_len, hidden_dim)
        h_j = hidden_states.unsqueeze(1).expand(batch, seq_len, seq_len, hidden_dim)
        pair_repr = torch.cat([h_i, h_j, h_i * h_j], dim=-1)

        return self.classifier(pair_repr)


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """`labels` has shape (batch, seq_len, seq_len) with `IGNORE_INDEX` (-100) on the unused
    lower triangle and special-token positions -- matches `torch.nn.CrossEntropyLoss`'s default
    `ignore_index`, so those cells contribute no gradient."""
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits.reshape(-1, len(TAGS)), labels.reshape(-1))
