# modules/keypoint_encoder.py

import torch
import torch.nn as nn


class KeypointEncoder(nn.Module):
    """
    Encodes keypoint sequences using a Transformer encoder.
    Input: (B, T, D)
    Output: (B, T, H)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, n_layers: int = 4, n_heads: int = 8, dropout: float = 0.1):
        super(KeypointEncoder, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.positional_encoding = PositionalEncoding(hidden_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.positional_encoding(x)
        return self.encoder(x)


class PositionalEncoding(nn.Module):
    """Adds positional encodings to the input sequence."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("positional_encodings", self.pe)

    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)
        return x + self.positional_encodings[:, :seq_len, :]
