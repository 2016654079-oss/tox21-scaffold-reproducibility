"""Run ECFP4-RF, GIN-MT and hierarchical-attention GIN on prepared Tox21 data.

The script expects data/tox21/{train,valid,test}.csv and preserves missing
assay labels through masked loss / per-task labeled-row fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "tox21"
OUT_DIR = ROOT / "results" / "initial_baselines"
SEED = 42
TASK_COLS: list[str] = []
ATOM_FIELDS = ("atomic_num", "degree", "formal_charge", "hybridization", "aromatic", "in_ring", "chirality")
ATOM_SIZES = (119, 7, 11, 8, 2, 2, 5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def labels_from(df: pd.DataFrame) -> np.ndarray:
    return df[TASK_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(y)
    yt, yp = y[valid].astype(int), p[valid]
    if len(yt) == 0:
        return {"n": 0, "roc_auc": None, "pr_auc": None}
    auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else None
    pr = float(average_precision_score(yt, yp)) if len(np.unique(yt)) == 2 else None
    return {"n": int(len(yt)), "roc_auc": auc, "pr_auc": pr}


def atom_categories(atom: Chem.Atom) -> list[int]:
    hyb = {Chem.rdchem.HybridizationType.SP: 1,
           Chem.rdchem.HybridizationType.SP2: 2,
           Chem.rdchem.HybridizationType.SP3: 3,
           Chem.rdchem.HybridizationType.SP3D: 4,
           Chem.rdchem.HybridizationType.SP3D2: 5}.get(atom.GetHybridization(), 0)
    chiral = {Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2}.get(atom.GetChiralTag(), 0)
    return [min(atom.GetAtomicNum(), 118), min(atom.GetDegree(), 6),
            max(-5, min(5, atom.GetFormalCharge())) + 5, hyb,
            int(atom.GetIsAromatic()), int(atom.IsInRing()), chiral]


def graph_record(smiles: str) -> dict[str, np.ndarray | int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    x = np.asarray([atom_categories(a) for a in mol.GetAtoms()], dtype=np.int64)
    n = mol.GetNumAtoms()

    # Cut BRICS bonds and find connected components in the remaining graph.
    cut_bonds = set()
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        cut_bonds.add(tuple(sorted((int(a), int(b)))))
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges, bond_ids = [], []
    bond_map = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2,
                Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if tuple(sorted((a, b))) not in cut_bonds:
            union(a, b)
        bt = bond_map.get(bond.GetBondType(), 0)
        edges.extend(((a, b), (b, a)))
        bond_ids.extend((bt, bt))
    roots, frag = {}, []
    for i in range(n):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
        frag.append(roots[root])
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return {"x": x, "edge_index": edge_index, "bond": np.asarray(bond_ids, dtype=np.int64),
            "frag": np.asarray(frag, dtype=np.int64), "n_frag": len(roots)}


def prepare_graphs(frame: pd.DataFrame) -> list[dict[str, np.ndarray | int]]:
    return [graph_record(s) for s in frame["smiles"].astype(str)]


def collate_graphs(batch: list[dict[str, np.ndarray | int]]) -> dict[str, torch.Tensor]:
    xs, edges, bonds, frags, mol_ids = [], [], [], [], []
    node_offset = frag_offset = 0
    for mol_id, g in enumerate(batch):
        x = g["x"]
        edge = g["edge_index"]
        frag = g["frag"]
        assert isinstance(x, np.ndarray) and isinstance(edge, np.ndarray) and isinstance(frag, np.ndarray)
        xs.append(x)
        frags.append(frag + frag_offset)
        mol_ids.append(np.full(len(x), mol_id, dtype=np.int64))
        if edge.shape[1]:
            edges.append(edge + node_offset)
            bonds.append(g["bond"])
        node_offset += len(x)
        frag_offset += int(g["n_frag"])
    edge_index = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
    bond = np.concatenate(bonds) if bonds else np.zeros(0, dtype=np.int64)
    return {
        "x": torch.from_numpy(np.concatenate(xs)),
        "edge_index": torch.from_numpy(edge_index),
        "bond": torch.from_numpy(bond),
        "frag": torch.from_numpy(np.concatenate(frags)),
        "mol": torch.from_numpy(np.concatenate(mol_ids)),
        "n_mol": torch.tensor(len(batch)), "n_frag": torch.tensor(frag_offset),
    }


def segment_attention_pool(h: torch.Tensor, group: torch.Tensor, n_groups: int,
                           scorer: nn.Module) -> torch.Tensor:
    scores = scorer(h).squeeze(-1)
    maxima = torch.full((n_groups,), -torch.inf, device=h.device, dtype=h.dtype)
    maxima.scatter_reduce_(0, group, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima[group])
    denom = torch.zeros(n_groups, device=h.device, dtype=h.dtype)
    denom.scatter_add_(0, group, weights)
    weights = weights / denom[group].clamp_min(1e-12)
    pooled = torch.zeros((n_groups, h.shape[-1]), device=h.device, dtype=h.dtype)
    pooled.index_add_(0, group, h * weights.unsqueeze(-1))
    return pooled


class MolecularGIN(nn.Module):
    def __init__(self, n_tasks: int, hierarchical: bool = False, dim: int = 128, layers: int = 3):
        super().__init__()
        self.hierarchical = hierarchical
        self.atom_embeddings = nn.ModuleList([nn.Embedding(size, dim) for size in ATOM_SIZES])
        self.bond_embedding = nn.Embedding(5, dim)
        self.eps = nn.Parameter(torch.zeros(layers))
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)) for _ in range(layers)])
        if hierarchical:
            self.atom_score = nn.Linear(dim, 1)
            self.intra_score = nn.Linear(dim, 1)
            self.fragment_score = nn.Linear(dim, 1)
            self.fusion = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(0.2))
        else:
            self.fusion = nn.Identity()
        self.head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(dim // 2, n_tasks))

    def forward(self, g: dict[str, torch.Tensor]) -> torch.Tensor:
        h = sum(emb(g["x"][:, i].clamp(0, emb.num_embeddings - 1)) for i, emb in enumerate(self.atom_embeddings))
        src, dst = g["edge_index"]
        for layer_idx, mlp in enumerate(self.layers):
            agg = torch.zeros_like(h)
            if src.numel():
                msg = h[src] + self.bond_embedding(g["bond"])
                agg.index_add_(0, dst, msg)
            h = mlp((1.0 + self.eps[layer_idx]) * h + agg)
        if self.hierarchical:
            z_mol = segment_attention_pool(h, g["mol"], int(g["n_mol"]), self.atom_score)
            z_frag = segment_attention_pool(h, g["frag"], int(g["n_frag"]), self.intra_score)
            frag_mol = torch.zeros(int(g["n_frag"]), device=h.device, dtype=torch.long)
            frag_mol.index_copy_(0, g["frag"], g["mol"])
            z_sub = segment_attention_pool(z_frag, frag_mol, int(g["n_mol"]), self.fragment_score)
            z = self.fusion(torch.cat((z_mol, z_sub), dim=-1))
        else:
            z = torch.zeros((int(g["n_mol"]), h.shape[-1]), device=h.device, dtype=h.dtype)
            z.index_add_(0, g["mol"], h)
            counts = torch.bincount(g["mol"], minlength=int(g["n_mol"])).to(h.dtype).unsqueeze(-1)
            z = z / counts.clamp_min(1)
        return self.head(z)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate_neural(model: nn.Module, graphs: list[dict[str, np.ndarray | int]], y: np.ndarray,
                    device: torch.device, batch_size: int) -> tuple[list[dict[str, float | int | None]], np.ndarray]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch_to_device(batch, device)).cpu().numpy())
    probs = 1 / (1 + np.exp(-np.concatenate(logits)))
    metrics = [safe_metrics(y[:, t], probs[:, t]) for t in range(y.shape[1])]
    return metrics, probs


def train_neural(name: str, train_frame: pd.DataFrame, valid_frame: pd.DataFrame, test_frame: pd.DataFrame,
                 train_graphs: list[dict[str, np.ndarray | int]], valid_graphs: list[dict[str, np.ndarray | int]],
                 test_graphs: list[dict[str, np.ndarray | int]], device: torch.device,
                 epochs: int, batch_size: int, dim: int, layers: int, patience: int) -> dict[str, object]:
    y_train, y_valid, y_test = labels_from(train_frame), labels_from(valid_frame), labels_from(test_frame)
    model = MolecularGIN(len(TASK_COLS), hierarchical=name == "HierGIN", dim=dim, layers=layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_auc, best_state, stale, best_epoch = -1.0, None, 0, 0
    rng = np.random.default_rng(SEED)
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_graphs))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = collate_graphs([train_graphs[int(i)] for i in idx])
            target = torch.as_tensor(y_train[idx], device=device)
            logits = model(batch_to_device(batch, device))
            mask = torch.isfinite(target)
            safe_target = torch.nan_to_num(target, nan=0.0)
            per_label = loss_fn(logits, safe_target)
            loss = (per_label * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        valid_metrics, _ = evaluate_neural(model, valid_graphs, y_valid, device, batch_size)
        aucs = [m["roc_auc"] for m in valid_metrics if m["roc_auc"] is not None]
        val_auc = float(np.mean(aucs)) if aucs else -1.0
        if val_auc > best_auc + 1e-5:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"{name} epoch {epoch:03d}: valid_macro_roc_auc={val_auc:.4f}", flush=True)
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, _ = evaluate_neural(model, test_graphs, y_test, device, batch_size)
    return {"model": name, "best_epoch": best_epoch, "valid_macro_roc_auc": best_auc,
            "test_macro_roc_auc": np.nanmean([m["roc_auc"] for m in test_metrics if m["roc_auc"] is not None]),
            "test_macro_pr_auc": np.nanmean([m["pr_auc"] for m in test_metrics if m["pr_auc"] is not None]),
            "seconds": round(time.time() - started, 1), "per_task_test": test_metrics}


def run_rf(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    started = time.time()
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def fingerprints(frame: pd.DataFrame) -> np.ndarray:
        arr = np.zeros((len(frame), 2048), dtype=np.uint8)
        for i, smiles in enumerate(frame["smiles"].astype(str)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES for fingerprint: {smiles}")
            fp = fp_generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr[i])
        return arr
    x_train, x_valid, x_test = fingerprints(train), fingerprints(valid), fingerprints(test)
    y_train, y_test = labels_from(train), labels_from(test)
    predictions = np.full_like(y_test, np.nan, dtype=np.float32)
    task_metrics = []
    for t, task in enumerate(TASK_COLS):
        mask = np.isfinite(y_train[:, t])
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1,
                                     class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=SEED + t)
        clf.fit(x_train[mask], y_train[mask, t].astype(int))
        predictions[:, t] = clf.predict_proba(x_test)[:, list(clf.classes_).index(1)]
        task_metrics.append(safe_metrics(y_test[:, t], predictions[:, t]))
        print(f"RF {task}: ROC-AUC={task_metrics[-1]['roc_auc']} PR-AUC={task_metrics[-1]['pr_auc']}", flush=True)
    return {"model": "ECFP4-RF", "best_epoch": None, "valid_macro_roc_auc": None,
            "test_macro_roc_auc": np.nanmean([m["roc_auc"] for m in task_metrics if m["roc_auc"] is not None]),
            "test_macro_pr_auc": np.nanmean([m["pr_auc"] for m in task_metrics if m["pr_auc"] is not None]),
            "seconds": round(time.time() - started, 1), "per_task_test": task_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["rf", "gin", "hiergin"], choices=["rf", "gin", "hiergin"])
    args = parser.parse_args()
    global SEED, OUT_DIR
    SEED = args.seed
    OUT_DIR = ROOT / "results" / f"baseline_seed_{SEED}"
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: read_split(name) for name in ("train", "valid", "test")}
    global TASK_COLS
    TASK_COLS = [c for c in frames["train"].columns if c not in ("mol_id", "smiles")]
    if any(list(frame.columns) != list(frames["train"].columns) for frame in frames.values()):
        raise ValueError("Train, valid, and test columns differ")
    # Some organometallic SMILES in the source file fail the current RDKit
    # sanitization rules. Drop them once, consistently, before every model.
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.error")
    for split_name, frame in frames.items():
        valid_smiles = frame["smiles"].astype(str).map(lambda s: Chem.MolFromSmiles(s) is not None)
        dropped = int((~valid_smiles).sum())
        if dropped:
            print(f"Dropped {dropped} RDKit-invalid molecules from {split_name}", flush=True)
        frames[split_name] = frame.loc[valid_smiles].reset_index(drop=True)
    RDLogger.EnableLog("rdApp.error")
    results: list[dict[str, object]] = []
    if "rf" in args.models:
        results.append(run_rf(frames["train"], frames["valid"], frames["test"]))
    neural_names = [name for name in args.models if name in ("gin", "hiergin")]
    if neural_names:
        graph_cache = {name: prepare_graphs(frame) for name, frame in frames.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Neural device: {device}; GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)
        for name in neural_names:
            nice = "GIN-MT" if name == "gin" else "HierGIN"
            results.append(train_neural(nice, frames["train"], frames["valid"], frames["test"],
                                        graph_cache["train"], graph_cache["valid"], graph_cache["test"],
                                        device, args.epochs, args.batch_size, args.dim, args.layers, args.patience))
    serializable = []
    for row in results:
        serializable.append({k: (float(v) if isinstance(v, (np.floating,)) else v)
                             for k, v in row.items() if k != "per_task_test"})
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"split": "random 80/10/10", "seed": SEED,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                   "tasks": TASK_COLS, "results": serializable,
                   "per_task": {str(r["model"]): r["per_task_test"] for r in results}}, f, indent=2, allow_nan=True)
    pd.DataFrame(serializable).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nSUMMARY")
    print(pd.DataFrame(serializable).to_string(index=False))


if __name__ == "__main__":
    main()
