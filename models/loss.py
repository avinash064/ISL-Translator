# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiTaskSign2TextLoss(nn.Module):
#     """
#     Production‑ready multi‑task loss for sign‑language→text.
#     Components:
#       • LSCE + focal modulation on decoder logits
#       • CTC on encoder log‑probs
#       • Triplet on encoder embeddings
#       • Uncertainty‑based dynamic weighting of CTC & triplet losses

#     References:
#       • Kendall et al., ‘Multi‑Task Learning Using Uncertainty to Weigh Losses’ (2018)
#       • Lin et al., ‘Focal Loss for Dense Object Detection’ (2017)
#     """
#     def __init__(
#         self,
#         pad_token_id: int,
#         ctc_blank_id: int,
#         label_smoothing: float = 0.1,
#         focal_gamma: float = 2.0,
#         triplet_margin: float = 1.0,
#         use_ctc: bool = True,
#         use_triplet: bool = True,
#     ):
#         super().__init__()
#         # LSCE
#         self.lsce = nn.CrossEntropyLoss(
#             ignore_index=pad_token_id,
#             label_smoothing=label_smoothing,
#         )
#         self.gamma = focal_gamma

#         # CTC
#         self.use_ctc = use_ctc
#         if use_ctc:
#             self.ctc = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)
#             # learnable log-variance for uncertainty weighting
#             self.log_var_ctc = nn.Parameter(torch.zeros(()))

#         # Triplet
#         self.use_triplet = use_triplet
#         if use_triplet:
#             self.triplet = nn.TripletMarginLoss(margin=triplet_margin)
#             self.log_var_trip = nn.Parameter(torch.zeros(()))

#     def forward(
#         self,
#         dec_logits: torch.Tensor,      # (B, T_dec, V)
#         tgt_seq: torch.LongTensor,     # (B, T_dec)
#         enc_logprobs: torch.Tensor,    # (B, T_enc, V_enc)
#         src_lens: torch.LongTensor,    # (B,)
#         tgt_lens: torch.LongTensor,    # (B,)
#         embeddings: torch.Tensor,      # (B, H) pooled
#         triplet_labels: torch.LongTensor,  # (B,) gloss IDs
#     ) -> Tuple[torch.Tensor, Dict[str, float]]:
#         B, T_dec, V = dec_logits.size()
#         metrics = {}

#         # 1) Label‑smoothed CE
#         flat_logits = dec_logits.view(-1, V)
#         flat_targets = tgt_seq.view(-1)
#         lsce_loss = self.lsce(flat_logits, flat_targets)
#         metrics['lsce'] = lsce_loss.item()

#         # 2) Focal scaling on LSCE
#         with torch.no_grad():
#             logp = F.log_softmax(dec_logits, dim=-1).view(-1, V)
#             pt = torch.exp(-self.lsce(logp, flat_targets))
#         focal_term = (1 - pt).pow(self.gamma)
#         focal_loss = (focal_term * self.lsce(logp, flat_targets)).mean()
#         metrics['focal'] = focal_loss.item()

#         total_loss = lsce_loss + focal_loss

#         # 3) CTC Loss
#         if self.use_ctc:
#             # enc_logprobs is assumed log-softmax already
#             ctc_loss = self.ctc(
#                 enc_logprobs.transpose(0,1),  # (T_enc, B, V_enc)
#                 tgt_seq,                       # (B, T_dec)
#                 src_lens,                      # (B,)
#                 tgt_lens                       # (B,)
#             )
#             # Uncertainty weighting: 1/(2σ²) * loss + log σ
#             precision_ctc = torch.exp(-self.log_var_ctc)
#             weighted_ctc = precision_ctc * ctc_loss + self.log_var_ctc * 0.5
#             total_loss = total_loss + weighted_ctc
#             metrics['ctc'] = ctc_loss.item()

#         # 4) Triplet Loss
#         if self.use_triplet:
#             # assume embeddings arranged [A1..An, P1..Pn, N1..Nn]
#             n = embeddings.size(0) // 3
#             anc, pos, neg = embeddings[:n], embeddings[n:2*n], embeddings[2*n:3*n]
#             trip_loss = self.triplet(anc, pos, neg)
#             precision_trip = torch.exp(-self.log_var_trip)
#             weighted_trip = precision_trip * trip_loss + self.log_var_trip * 0.5
#             total_loss = total_loss + weighted_trip
#             metrics['triplet'] = trip_loss.item()

#         metrics['total'] = total_loss.item()
#         return total_loss, metrics
import torch
from typing import Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskSign2TextLoss(nn.Module):
    """
    Production-ready multi-task loss for sign-language -> text.
    Combines:
      • LSCE + focal modulation on decoder logits
      • CTC on encoder log-probs
      • (Optional) Triplet on encoder embeddings
      • Uncertainty-based dynamic weighting of CTC & Triplet losses

    References:
      • Kendall et al., 'Multi-Task Learning Using Uncertainty to Weigh Losses' (2018)
      • Lin et al., 'Focal Loss for Dense Object Detection' (2017)
    """
    def __init__(
        self,
        pad_token_id: int,
        ctc_blank_id: int,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        triplet_margin: float = 1.0,
        use_ctc: bool = True,
        use_triplet: bool = False,
    ):
        super().__init__()
        # Label-smoothed cross-entropy
        self.lsce = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing
        )
        self.gamma = focal_gamma

        # CTC
        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)
            self.log_var_ctc = nn.Parameter(torch.zeros(()))

        # Triplet
        self.use_triplet = use_triplet
        if use_triplet:
            self.triplet = nn.TripletMarginLoss(margin=triplet_margin)
            self.log_var_trip = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        dec_logits: torch.Tensor,      # (B, T_dec, V)
        tgt_seq: torch.LongTensor,     # (B, T_dec)
        enc_logprobs: torch.Tensor,    # (B, T_enc, V_enc)
        src_lens: torch.LongTensor,    # (B,)
        tgt_lens: torch.LongTensor,    # (B,)
        embeddings: torch.Tensor,      # (B, H)
        triplet_labels: torch.LongTensor = None,  # (B,) gloss IDs for triplet
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        B, T_dec, V = dec_logits.shape

        # 1) Label-smoothed CE
        flat_logits = dec_logits.view(-1, V)
        flat_targets = tgt_seq.view(-1)
        lsce_loss = self.lsce(flat_logits, flat_targets)
        metrics['lsce'] = lsce_loss.item()

        # 2) Focal modulation
        with torch.no_grad():
            logp = F.log_softmax(dec_logits, dim=-1).view(-1, V)
            pt = torch.exp(-self.lsce(logp, flat_targets))
        focal_term = (1 - pt).pow(self.gamma)
        focal_loss = (focal_term * self.lsce(logp, flat_targets)).mean()
        metrics['focal'] = focal_loss.item()

        total_loss = lsce_loss + focal_loss

        # 3) CTC Loss
        if self.use_ctc:
            # enc_logprobs should be log-softmax
            ctc_loss = self.ctc(
                enc_logprobs.transpose(0, 1),  # (T_enc, B, V_enc)
                tgt_seq,                        # (B, T_dec)
                src_lens,                       # (B,)
                tgt_lens                        # (B,)
            )
            precision_ctc = torch.exp(-self.log_var_ctc)
            weighted_ctc = precision_ctc * ctc_loss + 0.5 * self.log_var_ctc
            total_loss = total_loss + weighted_ctc
            metrics['ctc'] = ctc_loss.item()

        # 4) Triplet Loss
        if self.use_triplet and triplet_labels is not None:
            # assume embeddings arranged [A...P...N...]
            n = embeddings.size(0) // 3
            anc, pos, neg = embeddings[:n], embeddings[n:2*n], embeddings[2*n:3*n]
            trip_loss = self.triplet(anc, pos, neg)
            precision_trip = torch.exp(-self.log_var_trip)
            weighted_trip = precision_trip * trip_loss + 0.5 * self.log_var_trip
            total_loss = total_loss + weighted_trip
            metrics['triplet'] = trip_loss.item()

        metrics['total'] = total_loss.item()
        return total_loss, metrics
