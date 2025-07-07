# """multilingual_decoder.py – Advanced multilingual Transformer decoder
# -----------------------------------------------------------------------
# This module implements a **language‑aware Transformer decoder** designed to plug
# into your sign‑language→text pipeline. It is compatible with mBART‑50
# (`facebook/mbart-large-50-many-to-many-mmt`) and other multilingual checkpoints.

# Highlights
# ~~~~~~~~~~
# * **Language adapters** inside every decoder layer
# * **Beam‑search** & greedy decoding helpers
# * **Token/embedding weight tying**
# * **Padding & causal masks** automatically applied
# * **BLEU scoring** (`sacrebleu`) utility
# * **ONNX export** for deployment
# * **Debug logging** toggle
# """

# from __future__ import annotations

# import logging
# import math
# from functools import lru_cache
# from typing import List, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# try:
#     from sacrebleu import corpus_bleu  # optional BLEU
# except ImportError:
#     corpus_bleu = None  # allow import even if sacrebleu missing

# try:
#     from langdetect import detect as lang_detect  # optional auto‑detect
# except ImportError:
#     lang_detect = None

# logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())


# # -----------------------------------------------------------------------------
# # Positional Encoding helper
# # -----------------------------------------------------------------------------
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 1000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
#         return x + self.pe[:, : x.size(1)]


# # -----------------------------------------------------------------------------
# # Adapter bottleneck layer
# # -----------------------------------------------------------------------------
# class Adapter(nn.Module):
#     def __init__(self, hidden_dim: int, bottleneck_ratio: float = 0.25):
#         super().__init__()
#         bottleneck = int(hidden_dim * bottleneck_ratio)
#         self.net = nn.Sequential(
#             nn.Linear(hidden_dim, bottleneck),
#             nn.ReLU(inplace=True),
#             nn.Linear(bottleneck, hidden_dim),
#         )
#         self.ln = nn.LayerNorm(hidden_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.ln(x + self.net(x))


# # -----------------------------------------------------------------------------
# class MultilingualTransformerDecoder(nn.Module):
#     """Language‑aware Transformer decoder.

#     Parameters
#     ----------
#     tokenizer : transformers.MBart50Tokenizer(Fast)
#         Tokenizer containing vocab & `lang_code_to_id` mapping.
#     hidden_dim : int
#         Model dimension.
#     n_layers : int
#         Number of decoder layers.
#     n_heads : int
#         Attention heads.
#     ff_dim : int
#         Feed‑forward dim.
#     dropout : float
#     max_len : int
#         Maximum target length supported (positional enc table).
#     adapter_ratio : float
#         Bottleneck size for language adapters (0 → disable).
#     debug : bool
#         If True, prints tensor shapes.
#     """

#     def __init__(
#         self,
#         tokenizer,
#         hidden_dim: int = 512,
#         n_layers: int = 6,
#         n_heads: int = 8,
#         ff_dim: int = 2048,
#         dropout: float = 0.1,
#         max_len: int = 256,
#         adapter_ratio: float = 0.25,
#         debug: bool = False,
#     ) -> None:
#         super().__init__()
#         self.debug = debug
#         self.tokenizer = tokenizer
#         self.pad_id = tokenizer.pad_token_id
#         self.eos_id = tokenizer.eos_token_id
#         self.lang2id = tokenizer.lang_code_to_id  # dict[str,int]

#         self.hidden_dim = hidden_dim
#         vocab_size = len(tokenizer)

#         # Embeddings
#         self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
#         self.lang_emb = nn.Embedding(len(self.lang2id), hidden_dim)
#         self.pos_enc = PositionalEncoding(hidden_dim, max_len)

#         # Decoder stack with optional adapters
#         self.layers = nn.ModuleList()
#         for _ in range(n_layers):
#             dec_layer = nn.TransformerDecoderLayer(
#                 d_model=hidden_dim,
#                 nhead=n_heads,
#                 dim_feedforward=ff_dim,
#                 dropout=dropout,
#                 batch_first=True,
#             )
#             if adapter_ratio > 0.0:
#                 adapter = Adapter(hidden_dim, adapter_ratio)
#             else:
#                 adapter = nn.Identity()
#             self.layers.append(nn.ModuleDict({"core": dec_layer, "adapter": adapter}))

#         self.norm = nn.LayerNorm(hidden_dim)
#         self.proj = nn.Linear(hidden_dim, vocab_size)
#         self.proj.weight = self.tok_emb.weight  # weight tying

#     # ------------------------------------------------------------------
#     @lru_cache(maxsize=128)
#     def _code_to_id(self, code: str | int) -> int:
#         if isinstance(code, int):
#             return code
#         if code in self.lang2id:
#             return self.lang2id[code]
#         raise ValueError(f"Unsupported language code: {code}")

#     # ------------------------------------------------------------------
#     def _embed(self, tgt_tokens: torch.Tensor, lang_id: int) -> torch.Tensor:
#         x = self.tok_emb(tgt_tokens)
#         x = x + self.lang_emb(torch.tensor(lang_id, device=x.device))
#         x = self.pos_enc(x)
#         return x

#     # ------------------------------------------------------------------
#     def forward(
#         self,
#         tgt_tokens: torch.Tensor,         # (B, T)
#         memory: torch.Tensor,             # (B, S, H)
#         lang: str | int,
#         tgt_padding_mask: Optional[torch.Tensor] = None,  # (B, T)
#     ) -> torch.Tensor:                    # logits (B, T, V)
#         lang_id = self._code_to_id(lang)
#         x = self._embed(tgt_tokens, lang_id)

#         T = tgt_tokens.size(1)
#         causal = torch.triu(torch.ones(T, T, device=x.device), 1).bool()

#         if tgt_padding_mask is None:
#             tgt_padding_mask = tgt_tokens.eq(self.pad_id)

#         for layer in self.layers:
#             x = layer["core"](
#                 x,
#                 memory,
#                 tgt_mask=causal,
#                 tgt_key_padding_mask=tgt_padding_mask,
#             )
#             x = layer["adapter"](x)
#         x = self.norm(x)
#         return self.proj(x)

#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def greedy_decode(
#         self,
#         memory: torch.Tensor,
#         bos_id: int,
#         lang: str | int,
#         max_len: int = 60,
#     ) -> torch.Tensor:
#         B = memory.size(0)
#         ys = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
#         for _ in range(max_len - 1):
#             logits = self.forward(ys, memory, lang)  # (B,T,V)
#             next_tok = logits[:, -1].argmax(-1, keepdim=True)
#             ys = torch.cat([ys, next_tok], dim=1)
#             if (next_tok == self.eos_id).all():
#                 break
#         return ys

#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def beam_search(
#         self,
#         memory: torch.Tensor,
#         bos_id: int,
#         lang: str | int,
#         beam_width: int = 5,
#         max_len: int = 60,
#         length_penalty: float = 1.0,
#     ) -> torch.Tensor:
#         assert memory.size(0) == 1, "Beam search supports batch=1 for simplicity"
#         device = memory.device
#         beams = [(torch.tensor([[bos_id]], device=device), 0.0)]
#         for _ in range(max_len - 1):
#             all_candidates: List[Tuple[torch.Tensor, float]] = []
#             for seq, score in beams:
#                 logits = self.forward(seq, memory, lang)[:, -1]
#                 logp = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
#                 topk = torch.topk(logp, beam_width)
#                 for idx, lp in zip(topk.indices, topk.values):
#                     new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
#                     new_score = score + lp.item()
#                     all_candidates.append((new_seq, new_score))
#             beams = sorted(
#                 all_candidates,
#                 key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
#                 reverse=True,
#             )[: beam_width]
#             if all(seq[0, -1] == self.eos_id for seq, _ in beams):
#                 break
#         return beams[0][0]

#     # ------------------------------------------------------------------
#     def translate(
#         self,
#         memory: torch.Tensor,
#         tgt_lang: str | int,
#         beam: bool = True,
#         beam_width: int = 5,
#         max_len: int = 60,
#     ) -> str:
#         bos_id = self._code_to_id(tgt_lang)
#         if beam:
#             seq = self.beam_search(memory, bos_id, tgt_lang, beam_width, max_len)
#         else:
#             seq = self.greedy_decode(memory, bos_id, tgt_lang, max_len)
#         return self.tokenizer.decode(seq.squeeze(), skip_special_tokens=True)

#     # ------------------------------------------------------------------
#     def score_bleu(self, preds: List[str], refs: List[str]) -> float:
#         if corpus_bleu is None:
#             raise RuntimeError("Install sacrebleu for BLEU scoring")
#         return corpus_bleu(preds, [refs]).score

#     # ------------------------------------------------------------------
#     def export_onnx(self, path: str | Tuple[str, ...], opset: int = 13):
#         if isinstance(path, (list, tuple)):
#             path = path[0]
#         dummy_tgt = torch.ones(1, 10, dtype=torch.long)
#         dummy_mem = torch.randn(1, 60, self.hidden_dim)
#         torch.onnx.export(
#             self,
#             (dummy_tgt, dummy_mem, 0),
#             path,
#             input_names=["tgt", "memory", "lang_id"],
#             output_names=["logits"],
#             dynamic_axes={"tgt": {1: "tgt_len"}, "memory": {1: "src_len"}},
#             opset_version=opset,
#         )
#         logger.info("Decoder ONNX exported to %s", path)


# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     from transformers import MBart50TokenizerFast

#     logging.basicConfig(level=logging.INFO)
#     tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
#     decoder = MultilingualTransformerDecoder(tok, debug=True)
#     memory = torch.randn(1, 60, decoder.hidden_dim)
#     translation = decoder.translate(memory, tgt_lang="hi_IN", beam=True)
#     print("Translation:", translation[:100])
"""multilingual_decoder.py – Advanced multilingual Transformer decoder
-----------------------------------------------------------------------
This module implements a language‑aware Transformer decoder for multilingual translation
compatible with mBART‑50 (`facebook/mbart-large-50-many-to-many-mmt`).

Key Features
~~~~~~~~~~~~
* Language adapters in every decoder layer
* Greedy & beam‑search decoding
* Token/embedding weight tying
* Automatic padding & causal masks
* BLEU scoring utility (sacrebleu)
* ONNX export helper
* Debug logging toggle
"""

import logging
import math
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sacrebleu import corpus_bleu
except ImportError:
    corpus_bleu = None

try:
    from langdetect import detect as lang_detect
except ImportError:
    lang_detect = None

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Adapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_ratio: float = 0.25):
        super().__init__()
        bottleneck = int(hidden_dim * bottleneck_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, hidden_dim),
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))


class MultilingualTransformerDecoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        adapter_ratio: float = 0.25,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.lang2id = tokenizer.lang_code_to_id

        self.hidden_dim = hidden_dim
        vocab_size = len(tokenizer)
        max_lang_id = max(self.lang2id.values())

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.lang_emb = nn.Embedding(max_lang_id + 1, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)

        # Decoder layers with adapters
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            core = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            )
            adapter = Adapter(hidden_dim, adapter_ratio) if adapter_ratio > 0.0 else nn.Identity()
            self.layers.append(nn.ModuleDict({"core": core, "adapter": adapter}))

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.proj.weight = self.tok_emb.weight

    @lru_cache(maxsize=128)
    def _code_to_id(self, code: str | int) -> int:
        if isinstance(code, int):
            return code
        if code in self.lang2id:
            return self.lang2id[code]
        raise ValueError(f"Unsupported language code: {code}")

    def _embed(self, tgt_tokens: torch.Tensor, lang_id: int) -> torch.Tensor:
        token_emb = self.tok_emb(tgt_tokens)
        lang_vec = self.lang_emb(torch.tensor(lang_id, device=token_emb.device))
        x = token_emb + lang_vec
        return self.pos_enc(x)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        lang: str | int,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lang_id = self._code_to_id(lang)
        x = self._embed(tgt_tokens, lang_id)

        T = tgt_tokens.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        if tgt_padding_mask is None:
            tgt_padding_mask = tgt_tokens.eq(self.pad_id)

        for layer in self.layers:
            x = layer["core"](
                x,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )
            x = layer["adapter"](x)
        x = self.norm(x)
        return self.proj(x)

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,
        bos_id: int,
        lang: str | int,
        max_len: int = 60,
    ) -> torch.Tensor:
        B = memory.size(0)
        ys = torch.full((B, 1), bos_id, device=memory.device, dtype=torch.long)
        for _ in range(max_len - 1):
            logits = self.forward(ys, memory, lang)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == self.eos_id).all():
                break
        return ys

    @torch.no_grad()
    def beam_search(
        self,
        memory: torch.Tensor,
        bos_id: int,
        lang: str | int,
        beam_width: int = 5,
        max_len: int = 60,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        assert memory.size(0) == 1, "Beam search supports batch=1"
        device = memory.device
        beams: List[Tuple[torch.Tensor, float]] = [(torch.tensor([[bos_id]], device=device), 0.0)]
        for _ in range(max_len - 1):
            candidates = []
            for seq, score in beams:
                logits = self.forward(seq, memory, lang)[:, -1]
                logp = F.log_softmax(logits, dim=-1).squeeze(0)
                topk = torch.topk(logp, beam_width)
                for idx, lp in zip(topk.indices, topk.values):
                    new_seq = torch.cat([seq, idx.view(1,1)], dim=1)
                    new_score = score + lp.item()
                    candidates.append((new_seq, new_score))
            beams = sorted(
                candidates,
                key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                reverse=True,
            )[:beam_width]
            if all(seq[0,-1] == self.eos_id for seq,_ in beams):
                break
        return beams[0][0]

    def translate(
        self,
        memory: torch.Tensor,
        tgt_lang: str | int,
        beam: bool = True,
        beam_width: int = 5,
        max_len: int = 60,
    ) -> str:
        bos_id = self._code_to_id(tgt_lang)
        if beam:
            seq = self.beam_search(memory, bos_id, tgt_lang, beam_width, max_len)
        else:
            seq = self.greedy_decode(memory, bos_id, tgt_lang, max_len)
        return self.tokenizer.decode(seq.squeeze(), skip_special_tokens=True)

    def score_bleu(self, preds: List[str], refs: List[str]) -> float:
        if not corpus_bleu:
            raise RuntimeError("Install sacrebleu for BLEU scoring")
        return corpus_bleu(preds, [refs]).score

    def export_onnx(self, path: str, opset: int = 13) -> None:
        dummy_tgt = torch.ones(1,10, dtype=torch.long)
        dummy_mem = torch.randn(1,60,self.hidden_dim)
        torch.onnx.export(
            self,
            (dummy_tgt, dummy_mem, 0),
            path,
            input_names=["tgt","memory","lang_id"],
            output_names=["logits"],
            dynamic_axes={"tgt":{1:"tgt_len"},"memory":{1:"src_len"}},
            opset_version=opset,
        )
        logger.info("Decoder ONNX exported to %s", path)


if __name__ == "__main__":
    from transformers import MBart50TokenizerFast

    logging.basicConfig(level=logging.INFO)
    tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    decoder = MultilingualTransformerDecoder(tok, debug=True)
    memory = torch.randn(1,60,decoder.hidden_dim)
    print(decoder.translate(memory, tgt_lang="hi_IN", beam=True)[:100])
