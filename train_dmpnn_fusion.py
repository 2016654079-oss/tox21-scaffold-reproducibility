"""Tox21 Chemprop D-MPNN and assay-specific ECFP4 gated fusion experiments.

Uses the existing random 80/10/10 split in data/tox21 and masks unobserved labels.
Run with: python train_dmpnn_fusion.py --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn import BondMessagePassing, MeanAggregation
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "tox21"
RESULTS_DIR = ROOT / "results"
TASKS: list[str] = []
FP_SIZE = 2048


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    smiles = df["smiles"].astype(str).tolist()
    ys = df[TASKS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    datapoints, keep, fps = [], [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.zeros((FP_SIZE,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fp)
        datapoints.append(MoleculeDatapoint(mol=mol, y=ys[i], x_d=fp, name=str(df.iloc[i].get("mol_id", i))))
        keep.append(i)
        fps.append(fp)
    return MoleculeDataset(datapoints), ys[keep], np.asarray(fps), [str(df.iloc[i]["smiles"]) for i in keep]


class DMPNNWithOptionalGatedFingerprint(nn.Module):
    def __init__(self, n_tasks: int, fusion: bool, hidden: int = 300, proj: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fusion = fusion
        self.message_passing = BondMessagePassing(d_h=hidden, depth=3, dropout=dropout)
        self.aggregation = MeanAggregation()
        self.graph_norm = nn.BatchNorm1d(hidden)
        self.graph_proj = nn.Sequential(nn.Linear(hidden, proj), nn.ReLU(), nn.Dropout(dropout))
        if fusion:
            self.fp_proj = nn.Sequential(nn.Linear(FP_SIZE, proj), nn.ReLU(), nn.Dropout(dropout))
            self.gate = nn.Sequential(nn.Linear(2 * proj, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, n_tasks))
            self.task_weight = nn.Parameter(torch.empty(n_tasks, proj))
            self.task_bias = nn.Parameter(torch.zeros(n_tasks))
            nn.init.xavier_uniform_(self.task_weight)
        else:
            self.head = nn.Sequential(nn.Linear(proj, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, n_tasks))

    def forward(self, batch):
        bmg, _, x_d, _, _, _, _ = batch
        h_atom = self.message_passing(bmg)
        h_graph = self.graph_norm(self.aggregation(h_atom, bmg.batch))
        z_graph = self.graph_proj(h_graph)
        if not self.fusion:
            return self.head(z_graph)
        if x_d is None:
            raise RuntimeError("ECFP4 descriptor batch is missing")
        z_fp = self.fp_proj(x_d.to(z_graph.device).float())
        gate = torch.sigmoid(self.gate(torch.cat((z_graph, z_fp), dim=-1)))
        z_fused = gate.unsqueeze(-1) * z_graph.unsqueeze(1) + (1.0 - gate.unsqueeze(-1)) * z_fp.unsqueeze(1)
        return (z_fused * self.task_weight.unsqueeze(0)).sum(dim=-1) + self.task_bias


def task_metrics(y: np.ndarray, p: np.ndarray):
    rows = []
    for task in range(y.shape[1]):
        mask = np.isfinite(y[:, task])
        yt = y[mask, task].astype(int)
        if len(np.unique(yt)) < 2:
            rows.append({"n": int(len(yt)), "roc_auc": None, "pr_auc": None})
        else:
            rows.append({"n": int(len(yt)), "roc_auc": float(roc_auc_score(yt, p[mask, task])),
                         "pr_auc": float(average_precision_score(yt, p[mask, task]))})
    return rows


def macro_metric(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def predict(model, loader, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in loader:
            batch.bmg.to(device)
            logits.append(model(batch).cpu().numpy())
    logits = np.concatenate(logits)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def fit_one(seed: int, model_name: str, datasets, labels, device, epochs: int, batch_size: int, patience: int):
    train_ds, valid_ds, test_ds = datasets
    y_train, y_valid, y_test = labels
    fusion = model_name == "DMPNN-ECFP4-Gated"
    seed_all(seed)
    model = DMPNNWithOptionalGatedFingerprint(len(TASKS), fusion).to(device)
    train_loader = build_dataloader(train_ds, batch_size=batch_size, num_workers=0, seed=seed, shuffle=True)
    valid_loader = build_dataloader(valid_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader = build_dataloader(test_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state, best_score, bad_epochs, best_epoch = None, -float("inf"), 0, 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch.bmg.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            y = batch.Y.to(device)
            mask = torch.isfinite(y)
            if not mask.any():
                continue
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_prob = predict(model, valid_loader, device)
        val_rows = task_metrics(y_valid, val_prob)
        score = macro_metric(val_rows, "roc_auc")
        if score is not None and score > best_score + 1e-5:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_prob = predict(model, test_loader, device)
    test_rows = task_metrics(y_test, test_prob)
    valid_rows = task_metrics(y_valid, predict(model, valid_loader, device))
    return {
        "model": model_name, "seed": seed, "best_epoch": best_epoch,
        "valid_macro_roc_auc": macro_metric(valid_rows, "roc_auc"),
        "valid_macro_pr_auc": macro_metric(valid_rows, "pr_auc"),
        "test_macro_roc_auc": macro_metric(test_rows, "roc_auc"),
        "test_macro_pr_auc": macro_metric(test_rows, "pr_auc"),
        "elapsed_seconds": round(time.time() - started, 2),
        "per_task": test_rows,
    }


def main():
    global TASKS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    # The five-seed comparison favors DMPNN; gated fusion remains available as an ablation.
    parser.add_argument("--models", nargs="+", default=["DMPNN"])
    args = parser.parse_args()
    TASKS = [c for c in pd.read_csv(DATA_DIR / "train.csv", nrows=1).columns if c not in ("mol_id", "smiles")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; tasks={len(TASKS)}; seeds={args.seeds}; models={args.models}", flush=True)
    splits = {name: load_split(name) for name in ("train", "valid", "test")}
    print("valid rows after RDKit parse:", {k: len(v[1]) for k, v in splits.items()}, flush=True)
    datasets = tuple(splits[s][0] for s in ("train", "valid", "test"))
    labels = tuple(splits[s][1] for s in ("train", "valid", "test"))
    out = RESULTS_DIR / "dmpnn_fusion"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"start model={model_name} seed={seed}", flush=True)
            row = fit_one(seed, model_name, datasets, labels, device, args.epochs, args.batch_size, args.patience)
            result_dir = out / f"{model_name.lower().replace('-', '_')}_seed_{seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "summary.json").write_text(json.dumps(row, indent=2, allow_nan=True), encoding="utf-8")
            results.append(row)
            print(f"done {model_name} seed={seed}: ROC-AUC={row['test_macro_roc_auc']:.4f}; PR-AUC={row['test_macro_pr_auc']:.4f}; epoch={row['best_epoch']}; sec={row['elapsed_seconds']}", flush=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["model", "seed", "best_epoch", "valid_macro_roc_auc", "valid_macro_pr_auc", "test_macro_roc_auc", "test_macro_pr_auc", "elapsed_seconds"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in results)
    summary = {}
    for name in args.models:
        rows = [r for r in results if r["model"] == name]
        summary[name] = {"runs": len(rows)}
        for metric in ("test_macro_roc_auc", "test_macro_pr_auc"):
            vals = [r[metric] for r in rows if r[metric] is not None]
            summary[name][metric + "_mean"] = statistics.mean(vals) if vals else None
            summary[name][metric + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    (out / "summary.json").write_text(json.dumps({"split": "random 80/10/10", "seeds": args.seeds, "models": summary}, indent=2), encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["model", "runs", "test_macro_roc_auc_mean", "test_macro_roc_auc_std", "test_macro_pr_auc_mean", "test_macro_pr_auc_std"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, row in summary.items():
            writer.writerow({"model": name, **row})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
