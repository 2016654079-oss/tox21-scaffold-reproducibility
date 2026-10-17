"""Prepare the MoleculeNet/DeepChem Tox21 multi-task dataset.

Creates deterministic 80/10/10 random train/validation/test CSV splits.
The raw labels and missing values are preserved; do not treat missing labels
as negatives. Random splitting follows the Tox21 loader's recommended split.
"""
from __future__ import annotations

import csv
import gzip
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "tox21.csv.gz"
OUT = ROOT / "data" / "tox21"
SEED = 42


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or "smiles" not in fields:
            raise ValueError("Expected a Tox21 CSV with a 'smiles' column")
        rows = [row for row in reader if row.get("smiles", "").strip()]

    # Keep identical SMILES together to avoid exact-string duplicate leakage.
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["smiles"].strip(), []).append(row)
    group_items = list(groups.items())
    random.Random(SEED).shuffle(group_items)
    n = len(rows)
    target_train, target_valid = round(n * 0.8), round(n * 0.1)
    partitions: dict[str, list[dict[str, str]]] = {"train": [], "valid": [], "test": []}
    for _, group in group_items:
        # Assign whole SMILES groups to the partition furthest below its target.
        deficits = {
            "train": target_train - len(partitions["train"]),
            "valid": target_valid - len(partitions["valid"]),
            "test": (n - target_train - target_valid) - len(partitions["test"]),
        }
        choice = max(deficits, key=deficits.get)
        partitions[choice].extend(group)

    task_cols = [c for c in fields if c not in {"mol_id", "smiles"}]
    summary: dict[str, object] = {
        "source": "DeepChem MoleculeNet Tox21 (tox21.csv.gz)",
        "source_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "split_method": "deterministic random split; exact identical SMILES kept together",
        "seed": SEED,
        "requested_ratio": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "rows_total": n,
        "unique_smiles": len(group_items),
        "tasks": task_cols,
        "partitions": {},
    }
    for name, subset in partitions.items():
        path = OUT / f"{name}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
        assay_stats = {}
        for task in task_cols:
            observed = [r.get(task, "").strip() for r in subset]
            labeled = [x for x in observed if x not in ("", "NA", "NaN", "nan")]
            positives = sum(x == "1" for x in labeled)
            assay_stats[task] = {
                "labeled": len(labeled),
                "missing": len(subset) - len(labeled),
                "positive": positives,
                "negative": len(labeled) - positives,
            }
        summary["partitions"][name] = {
            "rows": len(subset),
            "fraction": round(len(subset) / n, 6),
            "assays": assay_stats,
        }

    with (OUT / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "partitions"}, indent=2, ensure_ascii=False))
    print("partition_rows:", {k: len(v) for k, v in partitions.items()})


if __name__ == "__main__":
    main()
