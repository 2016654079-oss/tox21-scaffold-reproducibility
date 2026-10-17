# Experiment asset manifest

This manifest is based on `Tox21_scaffold_实验报告_导师汇报版.docx`. The report itself is kept outside the code repository.

## Dataset assets

- `tox21.csv.gz`: public raw Tox21 archive (7,831 rows; 7,823 valid molecules after validation).
- Random split: deterministic 80/10/10 train, validation, and test CSV files.
- Scaffold split: Bemis–Murcko scaffold split with 6,259/782/782 molecules.
- Independent scaffold splits: `split_101`, `split_102`, and `split_103`, each with train/validation/test CSV files.

## Model and result assets

- Baselines: ECFP4-RF, GIN-MT, and HierGIN summaries across seeds 42–46.
- D-MPNN: five-seed summaries and checkpoints for the main model and ablations.
- Ablations: task-aware attention, positive-class weighting, and ECFP4-gated fusion.
- Aggregates: model-level and per-task comparison tables, including macro ROC-AUC and average precision.

## Reproduction order

1. `prepare_tox21.py`
2. `make_tox21_scaffold_split.py`
3. `run_baselines.py` and `run_dmpnn_scaffold.py`
4. `aggregate_baselines.py` and `summarize_dmpnn_results.py`

The repository excludes papers, reports, figures, and reviewer materials.
