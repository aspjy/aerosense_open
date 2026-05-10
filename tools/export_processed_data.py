"""

Expected notebook variables:
    samples: list of (X, y)
        X has shape [num_aircraft_t, 18]
        y has shape [2] in the order [AP, AR]
    valid_timestamps: list-like timestamps
    feature_names: list of 18 feature names
"""

import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_FEATURE_NAMES = [
    "latitude", "longitude", "height",
    "speed", "climbOrDescendSpeed", "direction",
    "dialSpeed", "dialHeight",
    "dist2area_AP", "dist2area_AR",
    "approach_factor_AP", "approach_factor_AR",
    "is_in_AP", "is_in_AR",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]


def export_demo_processed_data(
    samples: Sequence[Tuple[np.ndarray, Sequence[float]]],
    valid_timestamps: Sequence,
    feature_names: List[str] = None,
    output_path: str = "data/aerosense_demo_500.pkl",
    num_samples: int = 500,
):
    """Export one pickle file containing processed AeroSense demo data."""
    feature_names = DEFAULT_FEATURE_NAMES if feature_names is None else list(feature_names)

    if len(samples) != len(valid_timestamps):
        raise ValueError("samples and valid_timestamps must have the same length.")
    if len(feature_names) != 18:
        raise ValueError(f"Expected 18 feature names, got {len(feature_names)}.")

    total = min(num_samples, len(samples))
    selected_samples = samples[:total]
    selected_timestamps = valid_timestamps[:total]

    X_list = []
    y_list = []

    for idx, (X, y) in enumerate(selected_samples):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"Sample {idx}: X must be 2-D, got shape {X.shape}.")
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Sample {idx}: feature dimension mismatch. "
                f"X.shape[1]={X.shape[1]}, feature_names={len(feature_names)}."
            )
        if y.shape[0] != 2:
            raise ValueError(f"Sample {idx}: y must have shape [2] in order [AP, AR], got {y.shape}.")

        X_list.append(X)
        y_list.append(y)

    payload = {
        "X": X_list,
        "y": np.stack(y_list, axis=0).astype(np.float32),
        "timestamps": [str(pd.Timestamp(t)) for t in selected_timestamps],
        "feature_names": feature_names,
        "metadata": {
            "dataset_name": "AeroSense demo processed data",
            "dataset_type": "demo_processed_data",
            "num_samples": total,
            "prediction_horizon_minutes": 15,
            "label_order": ["AP", "AR"],
            "input_format": "variable-cardinality aircraft-state set",
            "feature_dim": len(feature_names),
            "note": (
                "This is a small processed demo subset for verifying the AeroSense code pipeline. "
            ),
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved demo processed data -> {output_path}")
    print(f"Total samples: {total}")
    print(f"y shape: {payload['y'].shape}")
    print(f"First X shape: {payload['X'][0].shape}")
    print(f"Feature dim: {len(payload['feature_names'])}")

    return str(output_path)
