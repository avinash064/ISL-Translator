# # # models/training_validation.py

# # import argparse
# # from pathlib import Path
# # import random

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from tqdm import tqdm
# # from sacrebleu import corpus_bleu
# # from transformers import MBart50TokenizerFast

# # from datasets.keypoint_loader import build_loaders
# # from modules.keypoint_encoder import KeypointEncoder
# # from models.multilingual_decoder import MultilingualTransformerDecoder
# # from models.loss import MultiTaskSign2TextLoss  # Import the new loss


# # def save_checkpoint(epoch, encoder, decoder, optimizer, path: Path):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     torch.save({
# #         'epoch': epoch,
# #         'encoder': encoder.state_dict(),
# #         'decoder': decoder.state_dict(),
# #         'optimizer': optimizer.state_dict()
# #     }, path)
# #     print(f"✅ Checkpoint saved: {path}")


# # def load_checkpoint(path: Path, encoder, decoder, optimizer):
# #     if path.exists():
# #         ckpt = torch.load(path, map_location='cpu')
# #         encoder.load_state_dict(ckpt['encoder'])
# #         decoder.load_state_dict(ckpt['decoder'])
# #         optimizer.load_state_dict(ckpt['optimizer'])
# #         print(f"🔄 Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
# #         return ckpt['epoch']
# #     return 0


# # @torch.no_grad()
# # def evaluate(encoder, decoder, loader, tokenizer, loss_fn, device, tgt_lang='en_XX'):
# #     if loader is None:
# #         print("⚠️ Validation loader is None, skipping evaluation")
# #         return 0.0, 0.0
# #     encoder.eval()
# #     decoder.eval()
# #     total_loss = 0.0
# #     preds, refs = [], []

# #     for x, labels in tqdm(loader, desc='Validating'):
# #         x = x.to(device)
# #         labels = labels.to(device)
# #         B = labels.size(0)

# #         # Encoder forward
# #         enc_out = encoder(x)
# #         enc_logprobs = enc_out.log_softmax(-1) if hasattr(enc_out, 'log_softmax') else torch.log_softmax(enc_out, -1)
# #         embeddings = enc_out.mean(1)
# #         src_lens = torch.full((B,), enc_out.size(1), dtype=torch.long, device=device)

# #         # Prepare target for decoder
# #         bos = torch.full((B,1), tokenizer.bos_token_id, device=device)
# #         tgt_seq = torch.full((B,1), tokenizer.pad_token_id, device=device)
# #         # CTC needs full target sequences; here using labels as sequence
# #         tgt_lens = torch.ones((B,), dtype=torch.long, device=device)

# #         # Decoder forward
# #         dec_logits = decoder(bos, enc_out, lang=tgt_lang)

# #         # Generate predictions
# #         for m, lab in zip(enc_out, labels):
# #             pred = decoder.translate(m.unsqueeze(0), tgt_lang)
# #             refs.append(tokenizer.decode([lab.item()], skip_special_tokens=True))
# #             preds.append(pred)

# #         # Compute loss
# #         loss, _ = loss_fn(
# #             dec_logits, torch.cat([bos, labels.unsqueeze(1)],1),
# #             enc_logprobs, src_lens, tgt_lens,
# #             embeddings, None  # Triplet labels not used here
# #         )
# #         total_loss += loss.item()

# #     avg_loss = total_loss / len(loader)
# #     bleu = corpus_bleu(preds, [refs]).score
# #     return avg_loss, bleu


# # def train_one_epoch(encoder, decoder, loader, loss_fn, optimizer, device, tokenizer, tgt_lang='en_XX'):
# #     encoder.train()
# #     decoder.train()
# #     total_loss = 0.0

# #     for x, labels in tqdm(loader, desc='Training'):
# #         x = x.to(device)
# #         labels = labels.to(device)
# #         B = labels.size(0)

# #         # Encoder forward
# #         enc_out = encoder(x)
# #         enc_logprobs = torch.log_softmax(enc_out, -1)
# #         embeddings = enc_out.mean(1)
# #         src_lens = torch.full((B,), enc_out.size(1), dtype=torch.long, device=device)
# #         tgt_lens = torch.ones((B,), dtype=torch.long, device=device)

# #         # Decoder forward
# #         bos = torch.full((B,1), tokenizer.bos_token_id, device=device)
# #         tgt_seq = torch.cat([bos, labels.unsqueeze(1)], dim=1)
# #         dec_logits = decoder(bos, enc_out, lang=tgt_lang)

# #         # Compute multi-task loss
# #         optimizer.zero_grad()
# #         loss, metrics = loss_fn(
# #             dec_logits, tgt_seq,
# #             enc_logprobs, src_lens, tgt_lens,
# #             embeddings, None
# #         )
# #         loss.backward()
# #         nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
# #         nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
# #         optimizer.step()

# #         total_loss += loss.item()

# #     return total_loss / len(loader)


# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--keypoint_dir', required=True)
# #     parser.add_argument('--batch_size', type=int, default=32)
# #     parser.add_argument('--epochs', type=int, default=10)
# #     parser.add_argument('--checkpoint', type=str, default='checkpoints/model.pt')
# #     parser.add_argument('--max_len', type=int, default=120)
# #     args = parser.parse_args()

# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

# #     # Data loaders
# #     train_loader, val_loader, _, label_enc = build_loaders(
# #         keypoint_dir=args.keypoint_dir,
# #         batch_size=args.batch_size,
# #         max_seq_len=args.max_len,
# #         val_split=0.1,
# #         test_split=0.0,
# #     )
# #     if train_loader is None or val_loader is None:
# #         raise RuntimeError('Empty train/val split. Check data and split ratios.')

# #     # Models
# #     input_dim = next(iter(train_loader))[0].shape[-1]
# #     encoder = KeypointEncoder(input_dim, hidden_dim=512).to(device)
# #     decoder = MultilingualTransformerDecoder(tokenizer, hidden_dim=512).to(device)

# #     if torch.cuda.device_count() > 1:
# #         encoder = nn.DataParallel(encoder)
# #         decoder = nn.DataParallel(decoder)

# #     # Loss & optimizer
# #     loss_fn = MultiTaskSign2TextLoss(
# #         pad_token_id=tokenizer.pad_token_id,
# #         ctc_blank_id=tokenizer.pad_token_id,
# #         label_smoothing=0.1,
# #         focal_gamma=2.0,
# #         triplet_margin=1.0,
# #         use_ctc=True,
# #         use_triplet=False  # turn off triplet if no labels
# #     ).to(device)
# #     optimizer = optim.AdamW(
# #         list(encoder.parameters()) + list(decoder.parameters()),
# #         lr=2e-4,
# #         weight_decay=1e-2
# #     )

# #     # Resume
# #     start_epoch = load_checkpoint(Path(args.checkpoint), encoder, decoder, optimizer)

# #     for epoch in range(start_epoch, args.epochs):
# #         print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
# #         train_loss = train_one_epoch(
# #             encoder, decoder, train_loader, loss_fn, optimizer, device, tokenizer
# #         )
# #         print(f"Train Loss: {train_loss:.4f}")
# #         val_loss, val_bleu = evaluate(
# #             encoder, decoder, val_loader, tokenizer, loss_fn, device
# #         )
# #         print(f"Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f}")
# #         save_checkpoint(epoch+1, encoder, decoder, optimizer, Path(args.checkpoint))

# #     print("🎉 Training complete.")

# # if __name__ == '__main__':
# #     main()
# # # This script trains a Sign Language to Text model using keypoint data.
# # # It includes:
# # # - Model definition with a keypoint encoder and multilingual decoder
# # # - Training loop with BLEU evaluation
# # # - Checkpointing functionality
# # # - Command-line interface for easy configuration   
# # models/training_validation.py

# import argparse
# from pathlib import Path
# import random

# import torch
# from typing import Tuple, Dict
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from sacrebleu import corpus_bleu
# from transformers import MBart50TokenizerFast

# from datasets.keypoint_loader import build_loaders
# from modules.keypoint_encoder import KeypointEncoder
# from models.multilingual_decoder import MultilingualTransformerDecoder
# from models.loss import MultiTaskSign2TextLoss  # Import the new loss


# def save_checkpoint(epoch, encoder, decoder, optimizer, path: Path):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         'epoch': epoch,
#         'encoder': encoder.state_dict(),
#         'decoder': decoder.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }, path)
#     print(f"✅ Checkpoint saved: {path}")


# def load_checkpoint(path: Path, encoder, decoder, optimizer):
#     if path.exists():
#         ckpt = torch.load(path, map_location='cpu')
#         encoder.load_state_dict(ckpt['encoder'])
#         decoder.load_state_dict(ckpt['decoder'])
#         optimizer.load_state_dict(ckpt['optimizer'])
#         print(f"🔄 Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
#         return ckpt['epoch']
#     return 0


# @torch.no_grad()
# def evaluate(encoder, decoder, loader, tokenizer, loss_fn, device, tgt_lang='en_XX'):
#     if loader is None:
#         print("⚠️ Validation loader is None, skipping evaluation")
#         return 0.0, 0.0
#     encoder.eval()
#     decoder.eval()
#     total_loss = 0.0
#     preds, refs = [], []

#     for x, labels in tqdm(loader, desc='Validating'):
#         x = x.to(device)
#         labels = labels.to(device)
#         B = labels.size(0)

#         # Encoder forward
#         enc_out = encoder(x)
#         enc_logprobs = enc_out.log_softmax(-1) if hasattr(enc_out, 'log_softmax') else torch.log_softmax(enc_out, -1)
#         embeddings = enc_out.mean(1)
#         src_lens = torch.full((B,), enc_out.size(1), dtype=torch.long, device=device)

#         # Prepare target for decoder
#         bos = torch.full((B,1), tokenizer.bos_token_id, device=device)
#         tgt_seq = torch.full((B,1), tokenizer.pad_token_id, device=device)
#         # CTC needs full target sequences; here using labels as sequence
#         tgt_lens = torch.ones((B,), dtype=torch.long, device=device)

#         # Decoder forward
#         dec_logits = decoder(bos, enc_out, lang=tgt_lang)

#         # Generate predictions
#         for m, lab in zip(enc_out, labels):
#             pred = decoder.translate(m.unsqueeze(0), tgt_lang)
#             refs.append(tokenizer.decode([lab.item()], skip_special_tokens=True))
#             preds.append(pred)

#         # Compute loss
#         loss, _ = loss_fn(
#             dec_logits, torch.cat([bos, labels.unsqueeze(1)],1),
#             enc_logprobs, src_lens, tgt_lens,
#             embeddings, None  # Triplet labels not used here
#         )
#         total_loss += loss.item()

#     avg_loss = total_loss / len(loader)
#     bleu = corpus_bleu(preds, [refs]).score
#     return avg_loss, bleu


# def train_one_epoch(encoder, decoder, loader, loss_fn, optimizer, device, tokenizer, tgt_lang='en_XX'):
#     encoder.train()
#     decoder.train()
#     total_loss = 0.0

#     for x, labels in tqdm(loader, desc='Training'):
#         x = x.to(device)
#         labels = labels.to(device)
#         B = labels.size(0)

#         # Encoder forward
#         enc_out = encoder(x)
#         enc_logprobs = torch.log_softmax(enc_out, -1)
#         embeddings = enc_out.mean(1)
#         src_lens = torch.full((B,), enc_out.size(1), dtype=torch.long, device=device)
#         tgt_lens = torch.ones((B,), dtype=torch.long, device=device)

#         # Decoder forward
#         bos = torch.full((B,1), tokenizer.bos_token_id, device=device)
#         tgt_seq = torch.cat([bos, labels.unsqueeze(1)], dim=1)
#         dec_logits = decoder(bos, enc_out, lang=tgt_lang)

#         # Compute multi-task loss
#         optimizer.zero_grad()
#         loss, metrics = loss_fn(
#             dec_logits, tgt_seq,
#             enc_logprobs, src_lens, tgt_lens,
#             embeddings, None
#         )
#         loss.backward()
#         nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
#         nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(loader)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--keypoint_dir', required=True)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--checkpoint', type=str, default='checkpoints/model.pt')
#     parser.add_argument('--max_len', type=int, default=120)
#     args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

#     # Data loaders
#     train_loader, val_loader, _, label_enc = build_loaders(
#         keypoint_dir=args.keypoint_dir,
#         batch_size=args.batch_size,
#         max_seq_len=args.max_len,
#         val_split=0.1,
#         test_split=0.0,
#     )
#     if train_loader is None or val_loader is None:
#         raise RuntimeError('Empty train/val split. Check data and split ratios.')

#     # Models
#     input_dim = next(iter(train_loader))[0].shape[-1]
#     encoder = KeypointEncoder(input_dim, hidden_dim=512).to(device)
#     decoder = MultilingualTransformerDecoder(tokenizer, hidden_dim=512).to(device)

#     if torch.cuda.device_count() > 1:
#         encoder = nn.DataParallel(encoder)
#         decoder = nn.DataParallel(decoder)

#     # Loss & optimizer
#     loss_fn = MultiTaskSign2TextLoss(
#         pad_token_id=tokenizer.pad_token_id,
#         ctc_blank_id=tokenizer.pad_token_id,
#         label_smoothing=0.1,
#         focal_gamma=2.0,
#         triplet_margin=1.0,
#         use_ctc=True,
#         use_triplet=False  # turn off triplet if no labels
#     ).to(device)
#     optimizer = optim.AdamW(
#         list(encoder.parameters()) + list(decoder.parameters()),
#         lr=2e-4,
#         weight_decay=1e-2
#     )

#     # Resume
#     start_epoch = load_checkpoint(Path(args.checkpoint), encoder, decoder, optimizer)

#     for epoch in range(start_epoch, args.epochs):
#         print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
#         train_loss = train_one_epoch(
#             encoder, decoder, train_loader, loss_fn, optimizer, device, tokenizer
#         )
#         print(f"Train Loss: {train_loss:.4f}")
#         val_loss, val_bleu = evaluate(
#             encoder, decoder, val_loader, tokenizer, loss_fn, device
#         )
#         print(f"Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f}")
#         save_checkpoint(epoch+1, encoder, decoder, optimizer, Path(args.checkpoint))

#     print("🎉 Training complete.")

# if __name__ == '__main__':
#     main()
# # This script trains a Sign Language to Text model using keypoint data.
# # It includes:      
# # - Model definition with a keypoint encoder and multilingual decoder
# # - Training loop with BLEU evaluation
# # - Checkpointing functionality
# # - Command-line interface for easy configuration
# # models/training_validation.py 
import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sacrebleu import corpus_bleu
from transformers import MBart50TokenizerFast

from datasets.keypoint_loader import build_loaders
from modules.keypoint_encoder import KeypointEncoder
from models.multilingual_decoder import MultilingualTransformerDecoder
from models.loss import MultiTaskSign2TextLoss


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


def train_one_epoch(encoder, decoder, loader, loss_fn, optimizer, device, tokenizer, loss_type, tgt_lang='en_XX'):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for x, labels in tqdm(loader, desc='Training'):
        x = x.to(device)
        labels = labels.to(device)
        B = labels.size(0)

        # Encoder forward
        enc_out = encoder(x)  # (B, T_enc, H)
        enc_logprobs = torch.log_softmax(enc_out, -1)
        embeddings = enc_out.mean(1)
        src_lens = torch.full((B,), enc_out.size(1), dtype=torch.long, device=device)
        tgt_lens = torch.ones((B,), dtype=torch.long, device=device)

        # Decoder forward (only BOS)
        bos = torch.full((B,1), tokenizer.bos_token_id, device=device)
        dec_logits = decoder(bos, enc_out, lang=tgt_lang)  # (B,1,V)

        # Prepare targets depending on loss
        if loss_type == 'multitask':
            tgt_seq = torch.cat([bos, labels.unsqueeze(1)], dim=1)
            loss, _ = loss_fn(
                dec_logits, labels,  # MultiTaskLoss expects labels as next token
                enc_logprobs, src_lens, tgt_lens,
                embeddings, None
            )
        else:  # cross entropy
            logits = dec_logits[:,0,:]
            loss = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=0.1
            )(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(encoder, decoder, loader, tokenizer, loss_fn, device, loss_type, tgt_lang='en_XX'):
    if loader is None:
        return 0.0, 0.0
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    preds, refs = [], []

    for x, labels in tqdm(loader, desc='Validating'):
        x = x.to(device)
        labels = labels.to(device)
        B = labels.size(0)

        enc_out = encoder(x)
        enc_logprobs = torch.log_softmax(enc_out, -1)
        embeddings = enc_out.mean(1)
        src_lens = torch.full((B,), enc_out.size(1), dtype=torch.long, device=device)
        tgt_lens = torch.ones((B,), dtype=torch.long, device=device)

        bos = torch.full((B,1), tokenizer.bos_token_id, device=device)
        dec_logits = decoder(bos, enc_out, lang=tgt_lang)

        # Predictions
        for m, lab in zip(enc_out, labels):
            pred = decoder.translate(m.unsqueeze(0), tgt_lang)
            refs.append(tokenizer.decode([lab.item()], skip_special_tokens=True))
            preds.append(pred)

        # Loss
        if loss_type == 'multitask':
            loss, _ = loss_fn(
                dec_logits, labels,
                enc_logprobs, src_lens, tgt_lens,
                embeddings, None
            )
        else:
            logits = dec_logits[:,0,:]
            loss = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=0.1
            )(logits, labels)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    bleu = corpus_bleu(preds, [refs]).score
    return avg_loss, bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoint_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.pt')
    parser.add_argument('--max_len', type=int, default=120)
    parser.add_argument('--loss_type', choices=['multitask','ce'], default='multitask')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')

    train_loader, val_loader, _, label_enc = build_loaders(
        keypoint_dir=args.keypoint_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_len,
        val_split=0.1,
        test_split=0.0,
    )
    if train_loader is None or val_loader is None:
        raise RuntimeError('Empty train/val split')

    input_dim = next(iter(train_loader))[0].shape[-1]
    encoder = KeypointEncoder(input_dim, hidden_dim=512).to(device)
    decoder = MultilingualTransformerDecoder(tokenizer, hidden_dim=512).to(device)
    if torch.cuda.device_count()>1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    # Loss fn
    if args.loss_type=='multitask':
        loss_fn = MultiTaskSign2TextLoss(
            pad_token_id=tokenizer.pad_token_id,
            ctc_blank_id=tokenizer.pad_token_id,
            label_smoothing=0.1,
            focal_gamma=2.0,
            use_ctc=True,
            use_triplet=False
        ).to(device)
    else:
        loss_fn = None  # will use CE inline

    optimizer = optim.AdamW(
        list(encoder.parameters())+list(decoder.parameters()),
        lr=2e-4,
        weight_decay=1e-2
    )

    start_epoch = load_checkpoint(Path(args.checkpoint), encoder, decoder, optimizer)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ({args.loss_type}) ---")
        train_loss = train_one_epoch(
            encoder, decoder, train_loader, loss_fn,
            optimizer, device, tokenizer, args.loss_type
        )
        print(f"Train Loss: {train_loss:.4f}")
        val_loss, val_bleu = evaluate(
            encoder, decoder, val_loader, tokenizer,
            loss_fn, device, args.loss_type
        )
        print(f"Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f}")
        save_checkpoint(epoch+1, encoder, decoder, optimizer, Path(args.checkpoint))

    print("🎉 Training complete.")

if __name__=='__main__':
    main()
# This script trains a Sign Language to Text model using keypoint data. 
# It includes:
# - Model definition with a keypoint encoder and multilingual decoder
# - Training loop with BLEU evaluation
# - Checkpointing functionality
# - Command-line interface for easy configuration
# models/training_validation.py
# This script trains a Sign Language to Text model using keypoint data.
# It includes: