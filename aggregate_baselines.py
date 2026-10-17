"""Aggregate the 5 initial Tox21 baseline runs (seeds 42--46)."""
import csv
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
results = root / "results"
seeds = [42, 43, 44, 45, 46]
models = ["ECFP4-RF", "GIN-MT", "HierGIN"]
metrics = ["test_macro_roc_auc", "test_macro_pr_auc"]
aggregated = {}
for model in models:
    rows = []
    task_rows = []
    for seed in seeds:
        data = json.loads((results / f"baseline_seed_{seed}" / "summary.json").read_text(encoding="utf-8"))
        row = next(r for r in data["results"] if r["model"] == model)
        rows.append(row)
        task_rows.append(data["per_task"][model])
    agg = {"runs": len(rows), "seeds": seeds}
    for metric in metrics:
        vals = [float(r[metric]) for r in rows]
        agg[f"{metric}_mean"] = statistics.mean(vals)
        agg[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    task_agg = {}
    for task_idx, task in enumerate(data["tasks"]):
        task_agg[task] = {}
        for metric_name in ("roc_auc", "pr_auc"):
            vals = [float(seed_metrics[task_idx][metric_name]) for seed_metrics in task_rows
                    if seed_metrics[task_idx][metric_name] is not None]
            task_agg[task][metric_name + "_mean"] = statistics.mean(vals) if vals else None
            task_agg[task][metric_name + "_std"] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
    agg["per_task"] = task_agg
    aggregated[model] = agg

out_json = results / "initial_baselines_5seed.json"
out_json.write_text(json.dumps({"split": "random 80/10/10", "seeds": seeds, "models": aggregated},
                               indent=2, allow_nan=True), encoding="utf-8")
with (results / "initial_baselines_5seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
    fields = ["model", "runs", "test_macro_roc_auc_mean", "test_macro_roc_auc_std",
              "test_macro_pr_auc_mean", "test_macro_pr_auc_std"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for model, row in aggregated.items():
        writer.writerow({"model": model, **{k: v for k, v in row.items() if k not in ("per_task", "seeds")}})
print(out_json)
for model, row in aggregated.items():
    print(model, "ROC-AUC", f"{row['test_macro_roc_auc_mean']:.4f} ± {row['test_macro_roc_auc_std']:.4f}",
          "PR-AUC", f"{row['test_macro_pr_auc_mean']:.4f} ± {row['test_macro_pr_auc_std']:.4f}")
