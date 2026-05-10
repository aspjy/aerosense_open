"""Data loading utilities for AeroSense.

The released code uses a single processed pickle file by default:

    data/aerosense_demo_500.pkl

The file is split into train/validation/test subsets in memory. 
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

DEFAULT_FEATURE_NAMES = [
    "latitude", "longitude", "height",
    "speed", "climbOrDescendSpeed", "direction",
    "dialSpeed", "dialHeight",
    "dist2area_AP", "dist2area_AR",
    "approach_factor_AP", "approach_factor_AR",
    "is_in_AP", "is_in_AR",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]


def _load_any(path: Union[str, Path]) -> Dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
    if suffix == ".npz":
        obj = np.load(path, allow_pickle=True)
        return {k: obj[k] for k in obj.files}
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported data format: {path}")


def _normalize_loaded_object(obj: Dict) -> Tuple[List[np.ndarray], np.ndarray, Optional[List], Optional[List[str]]]:
    timestamps = obj.get("timestamps", None)
    feature_names = obj.get("feature_names", None)

    if "samples" in obj:
        samples = obj["samples"]
        X = [np.asarray(s[0], dtype=np.float32) for s in samples]
        y = np.asarray([s[1] for s in samples], dtype=np.float32)
        return X, y, timestamps, feature_names

    if "X" in obj and "y" in obj:
        X_raw = obj["X"]
        y = np.asarray(obj["y"], dtype=np.float32)
        if isinstance(X_raw, np.ndarray) and X_raw.dtype != object and X_raw.ndim == 3:
            X = [np.asarray(x, dtype=np.float32) for x in X_raw]
        else:
            X = [np.asarray(x, dtype=np.float32) for x in list(X_raw)]
        return X, y, timestamps, feature_names

    raise KeyError("Processed data must contain either 'samples' or both 'X' and 'y'.")


class TrafficFlowDataset(Dataset):
    """Variable-cardinality aircraft-state set dataset.

    Each sample is a matrix X_t with shape [num_aircraft_t, input_dim].
    The target y_t has shape [2] and follows the order [AP, AR].
    """

    def __init__(self, data_path: Union[str, Path], input_dim: int = 18):
        obj = _load_any(data_path)
        self.X, self.y, self.timestamps, self.feature_names = _normalize_loaded_object(obj)
        self.input_dim = input_dim

        if len(self.X) != len(self.y):
            raise ValueError(f"X/y length mismatch: {len(self.X)} vs {len(self.y)}")
        if self.timestamps is not None and len(self.timestamps) != len(self.X):
            raise ValueError(f"Timestamp length mismatch: {len(self.timestamps)} vs {len(self.X)}")

        for i, x in enumerate(self.X[: min(10, len(self.X))]):
            if x.ndim != 2:
                raise ValueError(f"X[{i}] must be 2-D, got shape {x.shape}")
            if x.shape[1] != input_dim:
                raise ValueError(
                    f"Feature dimension mismatch at X[{i}]: expected {input_dim}, got {x.shape[1]}"
                )
        if self.y.ndim != 2 or self.y.shape[1] != 2:
            raise ValueError(f"y must have shape [num_samples, 2], got {self.y.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            None if self.timestamps is None else self.timestamps[idx],
        )


def collate_aircraft_sets(batch):
    inputs, labels, timestamps = zip(*batch)
    batch_size = len(inputs)
    max_set_size = max(x.shape[0] for x in inputs)
    feat_dim = inputs[0].shape[1]

    padded_inputs = torch.zeros(batch_size, max_set_size, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_set_size, dtype=torch.bool)

    for i, x in enumerate(inputs):
        n = x.shape[0]
        if n > 0:
            padded_inputs[i, :n, :] = x
            mask[i, :n] = True

    labels = torch.stack(labels, dim=0)
    return padded_inputs, labels, mask, list(timestamps)


def split_indices(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    chronological: bool = True,
    seed: int = 42,
):
    """Return train/validation/test indices for one processed data file.

    The default chronological split matches the time-ordered evaluation protocol.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    indices = np.arange(n_samples)
    if not chronological:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            "The configured split produced an empty subset. "
            f"n_samples={n_samples}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
    return train_idx, val_idx, test_idx


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_aircraft_sets,
        pin_memory=torch.cuda.is_available(),
    )


def build_split_dataloaders(
    data_path: Union[str, Path],
    batch_size: int,
    input_dim: int = 18,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    chronological_split: bool = True,
    seed: int = 42,
):
    """Build train/validation/test DataLoaders from one processed pickle file."""
    dataset = TrafficFlowDataset(data_path, input_dim=input_dim)
    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        chronological=chronological_split,
        seed=seed,
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return (
        _make_loader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        _make_loader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        _make_loader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
