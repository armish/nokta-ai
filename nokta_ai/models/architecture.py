"""
Neural network architecture for Turkish diacritics restoration
"""

import torch
import torch.nn as nn


class DiacriticsRestorationModel(nn.Module):
    """Neural network model for diacritics restoration"""

    def __init__(self, vocab_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Bidirectional LSTM for context understanding
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism for focusing on relevant context
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)

        # LSTM encoding
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size * 2)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Output projection
        logits = self.output_projection(attn_out)  # (batch_size, seq_len, vocab_size)

        return logits