# import os
# import json
# import random
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader


# class LabelEncoder:
#     """Encodes labels (glosses) into integers, and vice versa."""
#     def __init__(self, vocab: Optional[List[str]] = None):
#         self.str2idx: Dict[str, int] = {}
#         self.idx2str: Dict[int, str] = {}
#         if vocab:
#             self.fit(vocab)

#     def fit(self, vocab: List[str]):
#         vocab = sorted(set(vocab))
#         self.str2idx = {s: i for i, s in enumerate(vocab)}
#         self.idx2str = {i: s for s, i in self.str2idx.items()}

#     def encode(self, label: str) -> int:
#         return self.str2idx[label]

#     def decode(self, idx: int) -> str:
#         return self.idx2str[idx]

#     def save(self, path: str):
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(self.str2idx, f, indent=2)

#     def load(self, path: str):
#         with open(path, "r", encoding="utf-8") as f:
#             self.str2idx = json.load(f)
#         self.idx2str = {i: s for s, i in self.str2idx.items()}


# def pad_sequence(arr: np.ndarray, max_len: int, pad_value: float = 0.0):
#     """Pads or truncates a 2D array (T, D) to (max_len, D)"""
#     T, D = arr.shape
#     if T >= max_len:
#         return arr[:max_len]
#     pad = np.full((max_len - T, D), pad_value)
#     return np.vstack([arr, pad])


# class KeypointDataset(Dataset):
#     def __init__(
#         self,
#         file_paths: List[Path],
#         label_map: Dict[str, int],
#         max_seq_len: int = 120,
#         pad_value: float = 0.0,
#     ):
#         self.file_paths = file_paths
#         self.label_map = label_map
#         self.max_seq_len = max_seq_len
#         self.pad_value = pad_value

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         file = self.file_paths[idx]
#         x = np.load(file)  # shape: (T, D)
#         label_str = file.stem.split("_")[0]  # Assumes format: <label>_videoId.npy
#         label = self.label_map[label_str]
#         x = pad_sequence(x, self.max_seq_len, self.pad_value)
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# def build_loaders(
#     keypoint_dir: str,
#     labels_csv: Optional[str] = None,
#     batch_size: int = 32,
#     max_seq_len: int = 120,
#     val_split: float = 0.1,
#     test_split: float = 0.1,
#     shuffle: bool = True,
#     seed: int = 42,
#     num_workers: int = 4,
# ) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
#     """
#     Build PyTorch DataLoaders for train/val/test splits.
#     If labels_csv is not provided, label is inferred from the filename prefix.
#     """
#     npy_files = sorted(Path(keypoint_dir).rglob("*.npy"))
#     if not npy_files:
#         raise RuntimeError(f"No .npy files found in {keypoint_dir}")

#     # Build label map
#     if labels_csv and os.path.exists(labels_csv):
#         df = pd.read_csv(labels_csv)
#         label_map = dict(zip(df["file"], df["label"]))
#         all_labels = sorted(set(df["label"]))
#         encoder = LabelEncoder(all_labels)
#         file_label_map = {Path(f).name: l for f, l in zip(df["file"], df["label"])}
#     else:
#         inferred_labels = [f.stem.split("_")[0] for f in npy_files]
#         encoder = LabelEncoder(inferred_labels)
#         file_label_map = {f.name: f.stem.split("_")[0] for f in npy_files}

#     random.seed(seed)
#     random.shuffle(npy_files)

#     n_total = len(npy_files)
#     n_val = int(n_total * val_split)
#     n_test = int(n_total * test_split)
#     n_train = n_total - n_val - n_test

#     train_files = npy_files[:n_train]
#     val_files = npy_files[n_train:n_train + n_val]
#     test_files = npy_files[n_train + n_val:]

#     def make_loader(file_list):
#         dataset = KeypointDataset(file_list, encoder.str2idx, max_seq_len)
#         return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

#     return (
#         make_loader(train_files),
#         make_loader(val_files),
#         make_loader(test_files),
#         encoder
#     )
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LabelEncoder:
    """Encodes string labels into integer indices and vice versa."""
    def __init__(self, vocab: Optional[List[str]] = None):
        self.str2idx = {}
        self.idx2str = {}
        if vocab:
            self.fit(vocab)

    def fit(self, vocab: List[str]):
        vocab = sorted(set(vocab))
        self.str2idx = {s: i for i, s in enumerate(vocab)}
        self.idx2str = {i: s for s, i in self.str2idx.items()}

    def encode(self, label: str) -> int:
        return self.str2idx[label]

    def decode(self, idx: int) -> str:
        return self.idx2str[idx]


class KeypointDataset(Dataset):
    """Dataset for keypoint .npy files and labels encoded from filename."""
    def __init__(
        self,
        file_paths: List[Path],
        label_encoder: LabelEncoder,
        max_seq_len: int = 120,
        pad_value: float = 0.0,
    ):
        self.file_paths = file_paths
        self.label_encoder = label_encoder
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        arr = np.load(path)
        # pad or truncate
        T, D = arr.shape
        if T >= self.max_seq_len:
            arr = arr[: self.max_seq_len]
        else:
            pad = np.full((self.max_seq_len - T, D), self.pad_value, dtype=arr.dtype)
            arr = np.vstack([arr, pad])
        label_str = path.stem.split("_")[0]
        label_id = self.label_encoder.encode(label_str)
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)


def build_loaders(
    keypoint_dir: str,
    batch_size: int = 32,
    max_seq_len: int = 120,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], LabelEncoder]:
    """
    Build train/val/test DataLoaders. Returns None for splits with zero samples.
    """
    keypoint_dir = Path(keypoint_dir)
    files = sorted(keypoint_dir.rglob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {keypoint_dir}")

    labels = [p.stem.split("_")[0] for p in files]
    encoder = LabelEncoder(labels)

    # shuffle and split
    random.seed(seed)
    random.shuffle(files)
    n = len(files)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    def make_loader(subset):
        if len(subset) == 0:
            return None
        ds = KeypointDataset(subset, encoder, max_seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (
        make_loader(train_files),
        make_loader(val_files),
        make_loader(test_files),
        encoder,
    )
