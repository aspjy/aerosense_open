import argparse
from pathlib import Path

import pandas as pd
import torch

from dataloader import build_split_dataloaders
from model import AeroSense
from utils import ensure_dir, evaluate_ap_ar, load_json, print_metrics, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="optDir/opt.json", help="Path to JSON config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. Defaults to logdir/checkpoint_name.")
    parser.add_argument("--output", type=str, default=None, help="Prediction CSV path.")
    args = parser.parse_args()

    opt = load_json(args.opt)
    if opt.get("num_threads") is not None:
        torch.set_num_threads(int(opt["num_threads"]))
    set_seed(opt.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() and opt.get("cuda", True) else "cpu")
    exp_dir = ensure_dir(opt.get("logdir", "checkpoints"))
    checkpoint = Path(args.checkpoint) if args.checkpoint else exp_dir / opt.get("checkpoint_name", "best_model.pt")
    output_csv = Path(args.output) if args.output else exp_dir / "test_predictions.csv"

    print(f"Using device: {device}")
    print(f"Using single processed data file: {opt['data_file']}")

    _, _, test_loader = build_split_dataloaders(
        opt["data_file"],
        batch_size=opt["batch_size"],
        input_dim=opt["input_dim"],
        num_workers=opt.get("num_workers", 0),
        train_ratio=opt.get("train_ratio", 0.8),
        val_ratio=opt.get("val_ratio", 0.1),
        chronological_split=opt.get("chronological_split", True),
        seed=opt.get("seed", 42),
    )

    model = AeroSense(
        input_dim=opt["input_dim"],
        hidden_dim=opt["hidden_dim"],
        num_heads=opt["num_heads"],
        dropout=0.0,
        use_attention=opt.get("use_attention", True),
        use_decoupled_heads=opt.get("use_decoupled_heads", True),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    preds, targets, timestamps = [], [], []
    with torch.no_grad():
        for inputs, tgt, mask, ts in test_loader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            out = model(inputs, mask)
            preds.append(out.cpu())
            targets.append(tgt.cpu())
            timestamps.extend(ts)

    y_pred = torch.cat(preds, dim=0).numpy()
    y_true = torch.cat(targets, dim=0).numpy()
    metrics = evaluate_ap_ar(y_true, y_pred)
    print_metrics(metrics)
    save_json(metrics, exp_dir / "test_metrics.json")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "y_true_ap": y_true[:, 0],
        "y_pred_ap": y_pred[:, 0],
        "y_true_ar": y_true[:, 1],
        "y_pred_ar": y_pred[:, 1],
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")


if __name__ == "__main__":
    main()
