"""Create deterministic Bemis-Murcko scaffold train/valid/test partitions.

Uses DeepChem-style largest-scaffold-first assignment. Acyclic molecules share
the empty Murcko scaffold group. Invalid SMILES are recorded and excluded.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "tox21.csv.gz"
OUT = PROJECT / "data" / "tox21_scaffold"
RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
RDLogger.DisableLog("rdApp.error")

def main():
    with gzip.open(RAW, "rt", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    tasks = [c for c in rows[0] if c not in ("mol_id", "smiles")]
    groups = defaultdict(list)
    invalid = []
    scaffold_by_smiles = {}
    for row in rows:
        smi = row["smiles"].strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(row)
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups[scaffold].append(row)
        scaffold_by_smiles[smi] = scaffold

    # Largest groups first, following the conventional scaffold splitter rule.
    scaffold_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    n = sum(len(v) for v in groups.values())
    cutoffs = {"train": RATIOS["train"] * n, "valid": (RATIOS["train"] + RATIOS["valid"]) * n}
    parts = {name: [] for name in RATIOS}
    assignment = {}
    for scaffold, members in scaffold_groups:
        if len(parts["train"]) < cutoffs["train"]:
            target = "train"
        elif len(parts["train"]) + len(parts["valid"]) < cutoffs["valid"]:
            target = "valid"
        else:
            target = "test"
        assignment[scaffold] = target
        parts[target].extend(members)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, subset in parts.items():
        pd.DataFrame(subset, columns=rows[0].keys()).to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    observed = {name: set() for name in parts}
    for name, subset in parts.items():
        observed[name] = {MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(r["smiles"]), includeChirality=False) for r in subset}
    overlaps = {"train_valid": sorted(observed["train"] & observed["valid"]),
                "train_test": sorted(observed["train"] & observed["test"]),
                "valid_test": sorted(observed["valid"] & observed["test"])}
    assay_summary = {}
    for name, subset in parts.items():
        assay_summary[name] = {"rows": len(subset), "scaffold_groups": len({MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(r["smiles"]), includeChirality=False) for r in subset}), "assays": {}}
        for task in tasks:
            vals = [r.get(task, "").strip() for r in subset]
            labeled = [v for v in vals if v in ("0", "1")]
            assay_summary[name]["assays"][task] = {"labeled": len(labeled), "missing": len(subset) - len(labeled),
                                                     "positive": sum(v == "1" for v in labeled),
                                                     "negative": sum(v == "0" for v in labeled)}
    summary = {"source": "MoleculeNet/DeepChem Tox21 tox21.csv.gz", "rows_original": len(rows),
               "invalid_smiles_excluded": len(invalid), "rows_in_split": n, "tasks": tasks,
               "method": "Bemis-Murcko scaffold split; largest scaffold groups assigned first; acyclic molecules share empty scaffold group",
               "ratios_requested": RATIOS, "scaffold_groups_total": len(groups), "partition_summary": assay_summary,
               "scaffold_overlap_counts": {k: len(v) for k, v in overlaps.items()},
               "scaffold_overlaps": overlaps}
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "partition_summary" and k != "scaffold_overlaps"}, indent=2, ensure_ascii=False))
    print("partition rows:", {k: len(v) for k, v in parts.items()})
    print("scaffold groups:", {k: assay_summary[k]["scaffold_groups"] for k in parts})
    print("positive counts by assay:")
    for task in tasks:
        print(task, {k: assay_summary[k]["assays"][task]["positive"] for k in parts})

if __name__ == "__main__":
    main()
