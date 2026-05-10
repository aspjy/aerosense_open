import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics.

    R2 can be undefined for very small or constant demo splits; in that case
    scikit-learn returns nan. We keep the value as a float so that CSV/JSON
    logging remains explicit.
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_ap_ar(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate predictions with target order [AP, AR]."""
    ap = regression_metrics(y_true[:, 0], y_pred[:, 0])
    ar = regression_metrics(y_true[:, 1], y_pred[:, 1])
    return {
        "AP_MAE": ap["MAE"],
        "AP_RMSE": ap["RMSE"],
        "AP_R2": ap["R2"],
        "AR_MAE": ar["MAE"],
        "AR_RMSE": ar["RMSE"],
        "AR_R2": ar["R2"],
        "Overall_MAE": float((ap["MAE"] + ar["MAE"]) / 2.0),
        "Overall_RMSE": float((ap["RMSE"] + ar["RMSE"]) / 2.0),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    print("======== Evaluation Metrics ========")
    print(f"Overall MAE : {metrics['Overall_MAE']:.4f}")
    print(f"Overall RMSE: {metrics['Overall_RMSE']:.4f}")
    print("------------------------------------")
    print(f"AP MAE: {metrics['AP_MAE']:.4f} | RMSE: {metrics['AP_RMSE']:.4f} | R2: {metrics['AP_R2']:.4f}")
    print(f"AR MAE: {metrics['AR_MAE']:.4f} | RMSE: {metrics['AR_RMSE']:.4f} | R2: {metrics['AR_R2']:.4f}")
