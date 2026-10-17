"""Combine existing Tox21 baseline and Chemprop experiments into comparison tables."""
from __future__ import annotations
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEEDS = [42, 43, 44, 45, 46]
TASKS = None

def mean_std(values):
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)

def read_run(model, seed):
    if model in {"ECFP4-RF", "GIN-MT", "HierGIN"}:
        data = json.loads((RESULTS / f"baseline_seed_{seed}" / "summary.json").read_text(encoding="utf-8"))
        row = next(r for r in data["results"] if r["model"] == model)
        return row, data["per_task"][model], data["tasks"]
    folder = "dmpnn_seed" if model == "DMPNN" else "dmpnn_ecfp4_gated_seed"
    data = json.loads((RESULTS / "dmpnn_fusion" / f"{folder}_{seed}" / "summary.json").read_text(encoding="utf-8"))
    return data, data["per_task"], None

models = ["ECFP4-RF", "GIN-MT", "HierGIN", "DMPNN", "DMPNN-ECFP4-Gated"]
summary_rows = []
per_task_rows = []
for model in models:
    rows, task_results = [], []
    for seed in SEEDS:
        row, per_task, tasks = read_run(model, seed)
        rows.append(row)
        task_results.append(per_task)
    if tasks is not None:
        TASKS = tasks
    if TASKS is None:
        TASKS = json.loads((RESULTS / "baseline_seed_42" / "summary.json").read_text(encoding="utf-8"))["tasks"]
    out = {"model": model, "runs": len(rows)}
    for metric in ("test_macro_roc_auc", "test_macro_pr_auc"):
        m, sd = mean_std([float(row[metric]) for row in rows])
        out[f"{metric}_mean"], out[f"{metric}_std"] = m, sd
    summary_rows.append(out)
    for i, task in enumerate(TASKS):
        task_row = {"model": model, "task": task}
        for metric in ("roc_auc", "pr_auc"):
            vals = [float(r[i][metric]) for r in task_results if r[i][metric] is not None]
            m, sd = mean_std(vals) if vals else (None, None)
            task_row[f"{metric}_mean"], task_row[f"{metric}_std"] = m, sd
        per_task_rows.append(task_row)

with (RESULTS / "model_comparison_5seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
    fields = list(summary_rows[0])
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader(); writer.writerows(summary_rows)
with (RESULTS / "per_task_comparison_5seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
    fields = list(per_task_rows[0])
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader(); writer.writerows(per_task_rows)
(RESULTS / "model_comparison_5seed.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
print("Five-seed Tox21 comparison (mean ± sample SD)")
for row in summary_rows:
    print(f"{row['model']:20s} ROC-AUC {row['test_macro_roc_auc_mean']:.4f} ± {row['test_macro_roc_auc_std']:.4f} | PR-AUC {row['test_macro_pr_auc_mean']:.4f} ± {row['test_macro_pr_auc_std']:.4f}")
