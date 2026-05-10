import argparse
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import build_split_dataloaders
from model import AeroSense
from utils import ensure_dir, evaluate_ap_ar, load_json, print_metrics, save_json, set_seed


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 0.0, path: Union[str, Path] = "best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = Path(path)
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def run_one_epoch(model, loader, criterion, device, optimizer=None, ap_loss_weight: float = 1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets, mask, _ in tqdm(loader, disable=not is_train):
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs, mask)
            loss_ap = criterion(outputs[:, 0], targets[:, 0])
            loss_ar = criterion(outputs[:, 1], targets[:, 1])
            loss = ap_loss_weight * loss_ap + loss_ar

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.item())
        all_preds.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, evaluate_ap_ar(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="optDir/opt.json", help="Path to JSON config.")
    args = parser.parse_args()

    opt = load_json(args.opt)
    if opt.get("num_threads") is not None:
        torch.set_num_threads(int(opt["num_threads"]))
    set_seed(opt.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() and opt.get("cuda", True) else "cpu")
    print(f"Using device: {device}")
    print(f"Using single processed data file: {opt['data_file']}")

    exp_dir = ensure_dir(opt.get("logdir", "checkpoints"))
    save_json(opt, exp_dir / "used_opt.json")

    train_loader, val_loader, _ = build_split_dataloaders(
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
        dropout=opt["dropout"],
        use_attention=opt.get("use_attention", True),
        use_decoupled_heads=opt.get("use_decoupled_heads", True),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt["lr"],
        weight_decay=opt.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=opt.get("lr_patience", 5)
    )
    criterion = nn.HuberLoss(delta=opt.get("huber_delta", 1.0))
    best_path = exp_dir / opt.get("checkpoint_name", "best_model.pt")
    early_stopping = EarlyStopping(patience=opt.get("patience", 30), path=best_path)

    logs = []
    for epoch in range(1, opt["epochs"] + 1):
        train_loss, train_metrics = run_one_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            ap_loss_weight=opt.get("ap_loss_weight", 1.0),
        )
        val_loss, val_metrics = run_one_epoch(
            model, val_loader, criterion, device, optimizer=None,
            ap_loss_weight=opt.get("ap_loss_weight", 1.0),
        )
        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(exp_dir / "training_log.csv", index=False)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.6f} | val loss {val_loss:.6f}")
        print("Validation:")
        print_metrics(val_metrics)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
