"""Isolated Tox21 scaffold-split D-MPNN experiments.

Modes differ from mean-readout D-MPNN by one planned component:
attention = task-conditioned atom attention readout;
pos_weight = square-root inverse-frequency positive BCE weights;
gate = assay-specific ECFP4 representation gate.
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

EXP = Path(__file__).resolve().parent
PROJECT = EXP
DATA_DIR = PROJECT / "data" / "tox21_scaffold"
OUT = EXP / "dmpnn_results"
TASKS: list[str] = []
FP_SIZE = 2048

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_split(name):
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    labels = frame[TASKS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    datapoints, keep = [], []
    for i, smi in enumerate(frame.smiles.astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.zeros(FP_SIZE, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol), fp)
        datapoints.append(MoleculeDatapoint(mol=mol, y=labels[i], x_d=fp, name=str(frame.iloc[i].get("mol_id", i))))
        keep.append(i)
    return MoleculeDataset(datapoints), labels[keep]

def segment_softmax(scores, groups, n_groups):
    maxima = torch.full((n_groups,), -torch.inf, dtype=scores.dtype, device=scores.device)
    maxima.scatter_reduce_(0, groups, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima[groups])
    den = torch.zeros(n_groups, dtype=scores.dtype, device=scores.device)
    den.scatter_add_(0, groups, weights)
    return weights / den[groups].clamp_min(1e-12)

class TaskAtomAttention(nn.Module):
    def __init__(self, hidden, tasks, proj, dropout):
        super().__init__()
        attn_dim = hidden // 2
        self.atom_projection = nn.Linear(hidden, attn_dim)
        self.task_query = nn.Parameter(torch.empty(tasks, attn_dim))
        nn.init.xavier_uniform_(self.task_query)
        self.projection = nn.Sequential(nn.Linear(hidden, proj), nn.ReLU(), nn.Dropout(dropout))
        self.task_weight = nn.Parameter(torch.empty(tasks, proj))
        self.task_bias = nn.Parameter(torch.zeros(tasks))
        nn.init.xavier_uniform_(self.task_weight)
        self.tasks = tasks

    def forward(self, atom_h, mol_index, batch_size):
        score = torch.tanh(self.atom_projection(atom_h)) @ self.task_query.T
        task_index = torch.arange(self.tasks, device=atom_h.device)
        groups = (mol_index[:, None] * self.tasks + task_index[None, :]).reshape(-1)
        weights = segment_softmax(score.reshape(-1), groups, batch_size * self.tasks)
        expanded_h = atom_h[:, None, :].expand(-1, self.tasks, -1).reshape(-1, atom_h.shape[-1])
        pooled = torch.zeros((batch_size * self.tasks, atom_h.shape[-1]), device=atom_h.device, dtype=atom_h.dtype)
        pooled.index_add_(0, groups, expanded_h * weights[:, None])
        z = self.projection(pooled).reshape(batch_size, self.tasks, -1)
        return (z * self.task_weight[None, :, :]).sum(-1) + self.task_bias[None, :]

class DMPNNVariant(nn.Module):
    def __init__(self, mode, tasks, hidden=300, depth=3, proj=128, dropout=0.2, pos_weight=None):
        super().__init__()
        self.mode = mode
        self.mp = BondMessagePassing(d_h=hidden, depth=depth, dropout=dropout)
        self.agg = MeanAggregation()
        self.norm = nn.BatchNorm1d(hidden)
        if mode == "attention":
            self.attn = TaskAtomAttention(hidden, tasks, proj, dropout)
        else:
            self.graph_proj = nn.Sequential(nn.Linear(hidden, proj), nn.ReLU(), nn.Dropout(dropout))
            if mode == "gate":
                self.fp_proj = nn.Sequential(nn.Linear(FP_SIZE, proj), nn.ReLU(), nn.Dropout(dropout))
                self.gate = nn.Sequential(nn.Linear(2 * proj, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, tasks))
                nn.init.constant_(self.gate[-1].bias, 2.0)  # initialize toward the graph branch
                self.task_weight = nn.Parameter(torch.empty(tasks, proj))
                self.task_bias = nn.Parameter(torch.zeros(tasks))
                nn.init.xavier_uniform_(self.task_weight)
            else:
                self.head = nn.Sequential(nn.Linear(proj, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, tasks))
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(tasks))

    def forward(self, batch):
        bmg, _, x_d, y, _, _, _ = batch
        h_atom = self.mp(bmg)
        n_mols = int(y.shape[0]) if y is not None else int(bmg.batch.max().item()) + 1
        if self.mode == "attention":
            # Task attention uses atom-level states from the same D-MPNN encoder.
            return self.attn(h_atom, bmg.batch, n_mols)
        h_graph = self.norm(self.agg(h_atom, bmg.batch))
        z_graph = self.graph_proj(h_graph)
        if self.mode != "gate":
            return self.head(z_graph)
        z_fp = self.fp_proj(x_d.to(z_graph.device).float())
        gate = torch.sigmoid(self.gate(torch.cat([z_graph, z_fp], dim=-1)))
        z = gate.unsqueeze(-1) * z_graph.unsqueeze(1) + (1 - gate.unsqueeze(-1)) * z_fp.unsqueeze(1)
        return (z * self.task_weight[None, :, :]).sum(-1) + self.task_bias[None, :]

def per_task_metrics(y, p):
    out = []
    for j in range(y.shape[1]):
        mask = np.isfinite(y[:, j]); yy = y[mask, j].astype(int)
        if len(np.unique(yy)) < 2:
            out.append({"n": int(len(yy)), "roc_auc": None, "pr_auc": None})
        else:
            out.append({"n": int(mask.sum()), "roc_auc": float(roc_auc_score(yy, p[mask, j])),
                        "pr_auc": float(average_precision_score(yy, p[mask, j]))})
    return out

def macro(rows, name):
    vals = [r[name] for r in rows if r[name] is not None]
    return float(np.mean(vals)) if vals else None

def predict(model, loader, device):
    model.eval(); logits = []
    with torch.no_grad():
        for batch in loader:
            batch.bmg.to(device)
            logits.append(model(batch).cpu().numpy())
    a = np.concatenate(logits)
    return 1 / (1 + np.exp(-np.clip(a, -40, 40)))

def fit(mode, seed, datasets, labels, cfg, device, evaluate_test=True):
    train_ds, valid_ds = datasets[:2]
    y_train, y_valid = labels[:2]
    set_seed(seed)
    n_pos = np.nansum(y_train, axis=0)
    n_neg = np.sum(np.isfinite(y_train), axis=0) - n_pos
    pos_weight = np.sqrt(n_neg / np.maximum(n_pos, 1)).clip(1, 8).astype(np.float32) if mode == "pos_weight" else None
    model = DMPNNVariant(mode, len(TASKS), cfg["hidden"], cfg["depth"], cfg["proj"], cfg["dropout"],
                         torch.tensor(pos_weight) if pos_weight is not None else None).to(device)
    train_loader = build_dataloader(train_ds, batch_size=cfg["batch_size"], num_workers=0, seed=seed, shuffle=True)
    valid_loader = build_dataloader(valid_ds, batch_size=cfg["batch_size"], num_workers=0, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_state, best_score, bad, best_epoch = None, -float("inf"), 0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch.bmg.to(device); optimizer.zero_grad(set_to_none=True)
            logits = model(batch); y = batch.Y.to(device); mask = torch.isfinite(y)
            if not mask.any(): continue
            loss_map = nn.functional.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
            if mode == "pos_weight":
                loss_map = loss_map * torch.where(y > 0.5, model.pos_weight[None, :], 1.0)
            loss = loss_map[mask].mean(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        vp = predict(model, valid_loader, device)
        vm = per_task_metrics(y_valid, vp); score = macro(vm, "roc_auc")
        if score is not None and score > best_score + 1e-5:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    if best_state is not None: model.load_state_dict(best_state)
    out = {"mode": mode, "seed": seed, "best_epoch": best_epoch,
           "valid_macro_roc_auc": best_score, "valid_macro_pr_auc": macro(vm, "pr_auc"),
           "config": cfg, "pos_weight": pos_weight.tolist() if pos_weight is not None else None}
    if evaluate_test:
        test_ds, y_test = datasets[2], labels[2]
        test_loader = build_dataloader(test_ds, batch_size=cfg["batch_size"], num_workers=0, shuffle=False)
        vp = predict(model, valid_loader, device); tp = predict(model, test_loader, device)
        valid_metrics = per_task_metrics(y_valid, vp); tm = per_task_metrics(y_test, tp)
        out.update({"valid_macro_roc_auc": macro(valid_metrics, "roc_auc"), "valid_macro_pr_auc": macro(valid_metrics, "pr_auc"),
                    "test_macro_roc_auc": macro(tm, "roc_auc"), "test_macro_pr_auc": macro(tm, "pr_auc"),
                    "per_task": tm})
        return out, best_state, vp, tp
    return out, best_state, None, None

def main():
    global TASKS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mean", "attention", "pos_weight", "gate"], default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--hidden", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--proj", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--config-id", default="selected")
    args = parser.parse_args()
    TASKS = [c for c in pd.read_csv(DATA_DIR / "train.csv", nrows=1).columns if c not in ("mol_id", "smiles")]
    names = ["train", "valid"] if args.validation_only else ["train", "valid", "test"]
    data = [read_split(n) for n in names]
    datasets, labels = tuple(x[0] for x in data), tuple(x[1] for x in data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"hidden": args.hidden, "depth": args.depth, "proj": args.proj, "dropout": args.dropout,
           "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
           "epochs": args.epochs, "patience": args.patience}
    seeds = args.seeds or [args.seed]
    print(f"mode={args.mode} seeds={seeds} device={device} sizes={[len(x) for x in datasets]} cfg={cfg}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        row, state, vp, tp = fit(args.mode, seed, datasets, labels, cfg, device, not args.validation_only)
        run_dir = OUT / f"{args.mode}_{args.config_id}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(row, indent=2, allow_nan=True), encoding="utf-8")
        if state is not None: torch.save({"state_dict": state, "mode": args.mode, "seed": seed, "config": cfg}, run_dir / "best_model.pt")
        if tp is not None:
            np.savez_compressed(run_dir / "predictions.npz", y_valid=labels[1], p_valid=vp, y_test=labels[2], p_test=tp)
        runs.append(row)
        if args.validation_only:
            print(f"seed={seed} val_ROC={row['valid_macro_roc_auc']:.5f} val_PR={row['valid_macro_pr_auc']:.5f} epoch={row['best_epoch']}", flush=True)
        else:
            print(f"seed={seed} test_ROC={row['test_macro_roc_auc']:.5f} test_PR={row['test_macro_pr_auc']:.5f} epoch={row['best_epoch']}", flush=True)
    (OUT / f"{args.mode}_{args.config_id}_runs.json").write_text(json.dumps(runs, indent=2, allow_nan=True), encoding="utf-8")
    if not args.validation_only and len(runs) > 1:
        summary = {"mode": args.mode, "runs": len(runs), "seeds": seeds}
        for key in ("test_macro_roc_auc", "test_macro_pr_auc"):
            values = [x[key] for x in runs]
            summary[key + "_mean"] = statistics.mean(values)
            summary[key + "_sd"] = statistics.stdev(values)
        (OUT / f"{args.mode}_{args.config_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__": main()
