import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (buffered)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        return x + self.pe[:, : x.size(1)]


class TransformerDecoder(nn.Module):
    """Enhanced Transformer Decoder with causal masking, flexible positional encodings,
    tied embeddings, output LayerNorm, and temperature scaling.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 200,
        pos_type: str = "learnable",  # 'learnable' | 'sin'
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pos_type = pos_type

        # Embedding & positional encoding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)
        if pos_type == "learnable":
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        else:
            self.pos_embedding = PositionalEncoding(hidden_dim, max_len=max_len)

        # Decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection (tied weights)
        self.output_ln = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight  # weight tying

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    # ------------------------------------------------------------------
    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_type == "learnable":
            return x + self.pos_embedding[:, : x.size(1)]
        return self.pos_embedding(x)  # sinusoidal module handles addition

    # ------------------------------------------------------------------
    def forward(
        self,
        tgt_input: torch.Tensor,  # [B, T_tgt] token indices
        memory: torch.Tensor,     # [B, T_src, H]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_tgt]
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_src]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass.

        Args
        ----
        tgt_input: target token indices (B, T_tgt)
        memory: encoder output (B, T_src, H)
        tgt_key_padding_mask: mask for target padding tokens (True=pad)
        memory_key_padding_mask: mask for source padding (True=pad)
        temperature: scale logits during training/inference (default 1.0)
        """
        # Embedding + Positional + Dropout
        tgt_embed = self.embedding(tgt_input)  # (B, T_tgt, H)
        tgt_embed = self._add_positional(tgt_embed)
        tgt_embed = self.embed_dropout(tgt_embed)

        # Causal mask (subsequent tokens) – shape (T_tgt, T_tgt)
        tgt_seq_len = tgt_input.size(1)
        causal_mask = self._generate_causal_mask(tgt_seq_len, tgt_input.device)

        # Decoder forward
        decoder_out = self.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T_tgt, H)

        logits = self.output_layer(self.output_ln(decoder_out)) / temperature
        return logits  # (B, T_tgt, vocab_size)


# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T_src, T_tgt, H, V = 2, 50, 20, 256, 1000
    memory = torch.randn(B, T_src, H)
    tgt_tokens = torch.randint(0, V, (B, T_tgt))

    dec = TransformerDecoder(vocab_size=V, hidden_dim=H, num_layers=2, pos_type="learnable")
    logits = dec(tgt_tokens, memory)
    print("Logits:", logits.shape)  # (B, T_tgt, V)
#     parser.add_argument(  
#         "--out_dir",
#         type=str,
#         default="/data/UG/Avinash/MedicalQ&A/SignLanguage/data/keypoints",
#         help="Directory to save extracted keypoints",
#     )