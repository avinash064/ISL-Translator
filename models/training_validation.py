# # # models/training_validation.py

# # import os
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from transformers import MBart50TokenizerFast
# # from tqdm import tqdm
# # from sacrebleu import corpus_bleu

# # from models.multilingual_decoder import MultilingualTransformerDecoder
# # from datasets.keypoint_loader import build_loaders
# # from modules.keypoint_encoder import KeypointEncoder  # Make sure you have this implemented


# # class Sign2TextModel(nn.Module):
# #     def __init__(self, encoder, decoder):
# #         super().__init__()
# #         self.encoder = encoder
# #         self.decoder = decoder

# #     def forward(self, x, tgt_tokens, lang):
# #         memory = self.encoder(x)
# #         return self.decoder(tgt_tokens, memory, lang)


# # def evaluate_bleu(decoder, encoder, val_loader, tokenizer, tgt_lang="en_XX", device="cpu"):
# #     decoder.eval()
# #     encoder.eval()
# #     preds, refs = [], []

# #     with torch.no_grad():
# #         for x, y in tqdm(val_loader, desc="Validation BLEU"):
# #             x = x.to(device)
# #             memory = encoder(x)
# #             for mem, label in zip(memory, y):
# #                 mem = mem.unsqueeze(0)
# #                 pred = decoder.translate(mem, tgt_lang=tgt_lang, beam=True)
# #                 target = tokenizer.decode([label.item()], skip_special_tokens=True)
# #                 preds.append(pred)
# #                 refs.append(target)

# #     bleu = corpus_bleu(preds, [refs])
# #     return bleu.score


# # def save_checkpoint(model, optimizer, epoch, path="checkpoint.pt"):
# #     torch.save({
# #         "epoch": epoch,
# #         "model_state": model.state_dict(),
# #         "optimizer_state": optimizer.state_dict()
# #     }, path)


# # def load_checkpoint(model, optimizer, path="checkpoint.pt"):
# #     if os.path.exists(path):
# #         ckpt = torch.load(path)
# #         model.load_state_dict(ckpt["model_state"])
# #         optimizer.load_state_dict(ckpt["optimizer_state"])
# #         print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']}")
# #         return ckpt["epoch"]
# #     return 0


# # def main():
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# #     train_loader, val_loader, _, _ = build_loaders(
# #         keypoint_dir="/data/UG/Avinash/MedicalQ&A/SignLanguage/data/isl_datasets/ISL_CSLRT_Corpus",
# #         labels_csv=None,
# #         batch_size=16,
# #         max_seq_len=120,
# #     )

# #     input_dim = next(iter(train_loader))[0].shape[-1]
# #     encoder = KeypointEncoder(input_dim, hidden_dim=512)
# #     decoder = MultilingualTransformerDecoder(tokenizer, hidden_dim=512)

# #     # Optional: wrap in DataParallel for multi-GPU
# #     model = Sign2TextModel(encoder, decoder)
# #     if torch.cuda.device_count() > 1:
# #         model = nn.DataParallel(model)

# #     model.to(device)
# #     optimizer = optim.Adam(model.parameters(), lr=1e-4)
# #     criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# #     start_epoch = load_checkpoint(model, optimizer, path="checkpoint.pt")
# #     num_epochs = 10

# #     for epoch in range(start_epoch, num_epochs):
# #         model.train()
# #         total_loss = 0.0
# #         for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
# #             x = x.to(device)  # (B, T, D)
# #             y = y.to(device)  # (B,)
# #             tgt = y.unsqueeze(1)  # dummy target

# #             optimizer.zero_grad()
# #             logits = model(x, tgt, lang="en_XX")  # (B, 1, V)
# #             logits = logits[:, 0, :]  # only BOS token

# #             loss = criterion(logits, y)
# #             loss.backward()
# #             optimizer.step()

# #             total_loss += loss.item()

# #         print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")
# #         save_checkpoint(model, optimizer, epoch + 1, path="checkpoint.pt")

# #         bleu = evaluate_bleu(decoder, encoder, val_loader, tokenizer, tgt_lang="en_XX", device=device)
# #         print(f"BLEU score: {bleu:.2f}")


# # if __name__ == "__main__":
# #     main()
# # # This script trains a Sign Language to Text model using keypoint data. It includes:
# # # - Model definition with a keypoint encoder and multilingual decoder
# # training_pipeline.py – End‑to‑end sign‑language → text training script
# # ------------------------------------------------------------------------
# # This file integrates:
# #   • KeypointLoader (datasets/keypoint_loader.py)
# #   • KeypointEncoder (modules/keypoint_encoder.py)
# #   • MultilingualTransformerDecoder (models/multilingual_decoder.py)
# # and provides a CLI to train, validate, BLEU‑score, and checkpoint.
# # Compatible with a single NVIDIA H100 (multi‑GPU optional).

# import argparse
# import os
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sacrebleu import corpus_bleu
# from transformers import MBart50TokenizerFast
# from tqdm import tqdm

# from datasets.keypoint_loader import build_loaders
# from modules.keypoint_encoder import KeypointEncoder
# from models.multilingual_decoder import MultilingualTransformerDecoder

# # --------------------------- helper utils --------------------------- #

# def save_checkpoint(epoch: int, encoder: nn.Module, decoder: nn.Module, optim: optim.Optimizer, path: Path):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         "epoch": epoch,
#         "encoder": encoder.state_dict(),
#         "decoder": decoder.state_dict(),
#         "optim": optim.state_dict(),
#     }, path)
#     print(f"💾 Saved checkpoint → {path}")


# def load_checkpoint(path: Path, encoder: nn.Module, decoder: nn.Module, optim: optim.Optimizer) -> int:
#     if not path.exists():
#         return 0
#     ckpt = torch.load(path, map_location="cpu")
#     encoder.load_state_dict(ckpt["encoder"])
#     decoder.load_state_dict(ckpt["decoder"])
#     optim.load_state_dict(ckpt["optim"])
#     print(f"🔄 Resumed from {path} (epoch {ckpt['epoch']})")
#     return ckpt["epoch"]


# @torch.no_grad()
# def evaluate_bleu(encoder, decoder, loader, tokenizer, device, tgt_lang="en_XX") -> float:
#     encoder.eval(); decoder.eval()
#     preds, refs = [], []
#     for x, labels in tqdm(loader, desc="BLEU eval"):
#         x = x.to(device)
#         memory = encoder(x)
#         for m, lab in zip(memory, labels):
#             sent_pred = decoder.translate(m.unsqueeze(0), tgt_lang)
#             sent_ref = tokenizer.decode([lab.item()], skip_special_tokens=True)
#             preds.append(sent_pred.strip())
#             refs.append(sent_ref.strip())
#     bleu = corpus_bleu(preds, [refs]).score
#     return bleu


# # --------------------------- training loop -------------------------- #

# def train_one_epoch(encoder, decoder, loader, criterion, optim, device, tokenizer):
#     encoder.train(); decoder.train()
#     total_loss = 0.0
#     for x, labels in tqdm(loader, desc="train"):
#         x, labels = x.to(device), labels.to(device)
#         bos = torch.full((labels.size(0), 1), tokenizer.bos_token_id, device=device)
#         tgt_in = bos  # decoder will autoregress from BOS token

#         optim.zero_grad()
#         memory = encoder(x)              # (B,T,H)
#         logits = decoder(tgt_in, memory, lang="en_XX")[:, 0, :]  # first step only
#         loss = criterion(logits, labels)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
#         torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
#         optim.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)


# # ------------------------------ main ------------------------------- #

# def main():
#     parser = argparse.ArgumentParser(description="Train sign‑language → text model (keypoints to mBART)")
#     parser.add_argument("--keypoint_dir", required=True)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--checkpoint", default="checkpoints/isl_model.pt")
#     parser.add_argument("--max_len", type=int, default=120)
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Data ------------------------------------------------------------------
#     train_loader, val_loader, _, label_enc = build_loaders(
#         keypoint_dir=args.keypoint_dir,
#         batch_size=args.batch_size,
#         max_seq_len=args.max_len,
#         val_split=0.1,
#         test_split=0.0,
#     )

#     # Models ----------------------------------------------------------------
#     input_dim = next(iter(train_loader))[0].shape[-1]
#     encoder = KeypointEncoder(input_dim, hidden_dim=512).to(device)
#     tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
#     decoder = MultilingualTransformerDecoder(tokenizer, hidden_dim=512).to(device)

#     if torch.cuda.device_count() > 1:
#         encoder = nn.DataParallel(encoder)
#         decoder = nn.DataParallel(decoder)

#     # Optim / Loss ----------------------------------------------------------
#     optim_ = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4)
#     criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

#     # Resume if checkpoint exists ------------------------------------------
#     start_epoch = load_checkpoint(Path(args.checkpoint), encoder, decoder, optim_)

#     # Training --------------------------------------------------------------
#     for epoch in range(start_epoch, args.epochs):
#         print(f"\n🚀 Epoch {epoch+1}/{args.epochs}")
#         loss = train_one_epoch(encoder, decoder, train_loader, criterion, optim_, device, tokenizer)
#         bleu = evaluate_bleu(encoder, decoder, val_loader, tokenizer, device)
#         print(f"Loss: {loss:.4f} | BLEU: {bleu:.2f}")
#         save_checkpoint(epoch+1, encoder, decoder, optim_, Path(args.checkpoint))


# if __name__ == "__main__":
#     main()
# models/training_validation.py

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sacrebleu import corpus_bleu
from transformers import MBart50TokenizerFast

from datasets.keypoint_loader import build_loaders
from modules.keypoint_encoder import KeypointEncoder
from models.multilingual_decoder import MultilingualTransformerDecoder


def save_checkpoint(epoch, encoder, decoder, optimizer, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)
    print(f"✅ Checkpoint saved: {path}")


def load_checkpoint(path: Path, encoder, decoder, optimizer):
    if path.exists():
        ckpt = torch.load(path, map_location='cpu')
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"🔄 Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt['epoch']
    return 0


@torch.no_grad()
def evaluate_bleu(encoder, decoder, loader, tokenizer, device, tgt_lang='en_XX'):
    if loader is None:
        print("⚠️ Validation loader is None, skipping BLEU evaluation")
        return 0.0
    encoder.eval()
    decoder.eval()
    preds, refs = [], []
    for x, labels in tqdm(loader, desc='Validating'):
        x = x.to(device)
        memory = encoder(x)
        for m, label in zip(memory, labels):
            m = m.unsqueeze(0)
            pred = decoder.translate(m, tgt_lang)
            ref = tokenizer.decode([label.item()], skip_special_tokens=True)
            preds.append(pred)
            refs.append(ref)
    bleu = corpus_bleu(preds, [refs]).score
    return bleu


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device, tokenizer):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for x, labels in tqdm(loader, desc='Training'):
        x, labels = x.to(device), labels.to(device)
        bos = torch.full((labels.size(0), 1), tokenizer.bos_token_id, device=device)
        optimizer.zero_grad()
        memory = encoder(x)
        logits = decoder(bos, memory, lang='en_XX')[:, 0, :]
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoint_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.pt')
    parser.add_argument('--max_len', type=int, default=120)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

    # Load data
    train_loader, val_loader, test_loader, label_enc = build_loaders(
        keypoint_dir=args.keypoint_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_len,
        val_split=0.1,
        test_split=0.0,
    )
    if train_loader is None or val_loader is None:
        raise RuntimeError('Train or validation loader is None. Check data splits.')

    # Build models
    input_dim = next(iter(train_loader))[0].shape[-1]
    encoder = KeypointEncoder(input_dim, hidden_dim=512).to(device)
    decoder = MultilingualTransformerDecoder(tokenizer, hidden_dim=512).to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    # Optimizer and loss
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Load checkpoint if exists
    start_epoch = load_checkpoint(Path(args.checkpoint), encoder, decoder, optimizer)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer, device, tokenizer)
        print(f"Train Loss: {train_loss:.4f}")
        val_bleu = evaluate_bleu(encoder, decoder, val_loader, tokenizer, device)
        print(f"Validation BLEU: {val_bleu:.2f}")
        # Save checkpoint
        save_checkpoint(epoch+1, encoder, decoder, optimizer, Path(args.checkpoint))

    print("🎉 Training complete.")

if __name__ == '__main__':
    main()
