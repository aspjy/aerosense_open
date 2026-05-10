import argparse
import pickle
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "latitude", "longitude", "height",
    "speed", "climbOrDescendSpeed", "direction",
    "dialSpeed", "dialHeight",
    "dist2area_AP", "dist2area_AR",
    "approach_factor_AP", "approach_factor_AR",
    "is_in_AP", "is_in_AR",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]


def make_data(n_samples: int, rng: np.random.Generator):
    X, y, timestamps = [], [], []
    for i in range(n_samples):
        n_aircraft = int(rng.integers(5, 35))
        x = rng.normal(size=(n_aircraft, 18)).astype(np.float32)

        # Simulated AP/AR inclusion indicators.
        x[:, 12] = (rng.random(n_aircraft) < 0.20).astype(np.float32)  # is_in_AP
        x[:, 13] = (rng.random(n_aircraft) < 0.45).astype(np.float32)  # is_in_AR

        y_ap = max(0, int(x[:, 12].sum() + rng.normal(2, 1)))
        y_ar = max(0, int(x[:, 13].sum() + rng.normal(4, 2)))

        X.append(x)
        y.append([y_ap, y_ar])
        timestamps.append(f"toy_{i:05d}")

    return {
        "X": X,
        "y": np.asarray(y, dtype=np.float32),
        "timestamps": timestamps,
        "feature_names": FEATURE_NAMES,
        "metadata": {
            "dataset_type": "toy_processed_data",
            "num_samples": n_samples,
            "feature_dim": 18,
            "label_order": ["AP", "AR"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/aerosense_demo_500.pkl")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(make_data(args.num_samples, rng), f)

    print(f"Saved {args.num_samples} toy samples to {out_path}")


if __name__ == "__main__":
    main()
