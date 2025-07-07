"""seq2seq_training.py – Production​-ready training loop for sign​-language translation.
Run from project root:
    python -m training.seq2seq_training
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Absolute imports relative to project root
from .Transformer_encoder import TransformerEncoder
from .Transformer_decoder import TransformerDecoder
from .datasets.dataloader import KeypointDataset


class Seq2SeqTrainer:
    """Jointly trains encoder and decoder with mixed precision, label smoothing, gradient clipping,
    and token​-level accuracy logging."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        dataloader: DataLoader,
        tokenizer,
        device: str = "cuda",
        lr: float = 1e-4,
    ) -> None:
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = device

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
        )
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=lr
        )
        self.scaler = GradScaler()

    # ------------------------------------------------------------------
    def train_one_epoch(self):
        self.encoder.train()
        self.decoder.train()

        running_loss, correct_tokens, total_tokens = 0.0, 0, 0
        pbar = tqdm(self.dataloader, desc="Training")

        for keypoints, tgt_tokens in pbar:
            keypoints = keypoints.to(self.device)
            tgt_tokens = tgt_tokens.to(self.device)

            tgt_input = tgt_tokens[:, :-1]
            tgt_output = tgt_tokens[:, 1:]

            self.optimizer.zero_grad()
            with autocast():
                memory = self.encoder(keypoints)
                logits = self.decoder(tgt_input, memory)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

            self.scaler.scale(loss).backward()
            clip_grad_norm_(self.encoder.parameters(), 1.0)
            clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = tgt_output != self.tokenizer.pad_token_id
            correct_tokens += (preds == tgt_output).masked_fill(~mask, 0).sum().item()
            total_tokens += mask.sum().item()

            acc = correct_tokens / (total_tokens + 1e-8)
            pbar.set_postfix({
                "loss": f"{running_loss / (pbar.n + 1):.4f}",
                "acc": f"{acc:.4f}"
            })

        return running_loss / len(self.dataloader)


# -----------------------------------------------------------------------------
# Stand​-alone run (example)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer

    BATCH_SIZE = 32
    INPUT_DIM = 150
    MAX_LEN = 120
    VOCAB = "ai4bharat/indic-trans"

    tokenizer = AutoTokenizer.from_pretrained(VOCAB)

    # Build dataset (simple file list inference)
    keypoint_root = "data/keypoints/ISL-CSLTR"
    npy_files = [os.path.join(keypoint_root, f) for f in os.listdir(keypoint_root) if f.endswith(".npy")]
    dataset = KeypointDataset(npy_files, {"dummy": 0}, max_seq_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = TransformerEncoder(input_dim=INPUT_DIM, hidden_dim=256, num_layers=4, output_mode="sequence")
    decoder = TransformerDecoder(vocab_size=tokenizer.vocab_size, hidden_dim=256, num_layers=4)

    trainer = Seq2SeqTrainer(encoder, decoder, dataloader, tokenizer)
    trainer.train_one_epoch()

# -----------------------------------------------------------------------------
# Note: This script assumes the existence of a `KeypointDataset` class that loads
# keypoint data from .npy files and a tokenizer compatible with the target language.
# Adjust paths and parameters as needed for your specific dataset and model.
# -----------------------------------------------------------------------------