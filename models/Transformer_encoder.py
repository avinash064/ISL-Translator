import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (unchanged once created)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    """Modular, decoder‑agnostic Transformer encoder for keypoint sequences.

    Args
    ----
    input_dim:    Dimensionality of raw keypoint vector per frame
    hidden_dim:   Model/embedding size of Transformer
    n_heads:      Multi‑head attention heads
    num_layers:   Number of encoder layers
    dropout:      Dropout probability
    max_len:      Maximum expected sequence length (positional embedding size)
    pos_type:     'learnable' | 'sin' – type of positional embedding
    output_mode:  'sequence' | 'cls' – return full sequence or logits for classification
    pool:         'mean' | 'first' – pooling strategy when output_mode='cls'
    num_classes:  Required if output_mode='cls'
    """

    def __init__(
        self,
        input_dim: int = 150,
        hidden_dim: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 200,
        pos_type: str = "learnable",
        output_mode: str = "sequence",
        pool: str = "mean",
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        assert output_mode in {"sequence", "cls"}
        if output_mode == "cls":
            assert num_classes is not None, "num_classes must be set when output_mode='cls'"

        self.output_mode = output_mode
        self.pool = pool
        self.hidden_dim = hidden_dim

        # Feature normalisation & projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_linear = nn.Linear(input_dim, hidden_dim)

        # Positional embeddings
        if pos_type == "learnable":
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        else:  # sinusoidal
            self.pos_embedding = None
            self.sin_pe = PositionalEncoding(hidden_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head (optional)
        if self.output_mode == "cls":
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    # ------------------------------------------------------------------
    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embedding is not None:  # learnable
            x = x + self.pos_embedding[:, : x.size(1)]
            return x
        # sinusoidal
        return self.sin_pe(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,  # (B, T, input_dim)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True for PAD
    ):
        # Dynamic mask if none provided – frames where all features ~0 -> pad
        if src_key_padding_mask is None:
            with torch.no_grad():
                src_key_padding_mask = (x.abs().sum(dim=-1) < 1e-6)

        # Pre‑norm & projection
        x = self.input_norm(x)
        x = self.input_linear(x)  # (B, T, hidden_dim)

        # Positional encoding
        x = self._add_positional_encoding(x)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if self.output_mode == "sequence":
            return x  # (B, T, hidden_dim)

        # Classification path
        if self.pool == "mean":
            pooled = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0).sum(dim=1)
            lens = (~src_key_padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = pooled / lens  # mean over valid frames
        else:  # first token pooling
            pooled = x[:, 0, :]

        logits = self.classifier(pooled)
        return logits


# -----------------------------------------------------------------------------
# Quick sanity check
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, D = 4, 100, 150
    dummy = torch.randn(B, T, D)
    enc = TransformerEncoder(input_dim=D, hidden_dim=256, num_layers=2, output_mode="sequence")
    seq_out = enc(dummy)
    print("Sequence output:", seq_out.shape)  # (B, T, 256)

    clf = TransformerEncoder(input_dim=D, hidden_dim=256, num_layers=2, output_mode="cls", num_classes=10)
    logits = clf(dummy)
    print("Classification logits:", logits.shape)  # (B, 10)
