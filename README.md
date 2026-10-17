# Tox21 Scaffold Reproducibility

Code for reproducible multi-task molecular toxicity prediction on Tox21. The repository follows a compact research-code layout: data preparation, training, baselines, and result aggregation are kept at the project root; generated data and results are ignored.

## Contents

- `download_tox21.py` downloads the public MoleculeNet/DeepChem Tox21 archive.
- `prepare_tox21.py` creates a deterministic 80/10/10 random split.
- `make_tox21_scaffold_split.py` creates a Bemis–Murcko scaffold split.
- `train_dmpnn_fusion.py` trains Chemprop 2.3.1 D-MPNN and the optional ECFP4-gated fusion ablation.
- `run_dmpnn_scaffold.py` trains the D-MPNN on the scaffold split.
- `run_baselines.py` runs the fingerprint and graph baselines.
- `aggregate_baselines.py` and `summarize_dmpnn_results.py` aggregate multi-seed results.
- `requirements-dmpnn.txt` and `requirements-baselines.txt` define the environments.

The raw dataset, split CSV files, model checkpoints, result tables, figures, papers, reports, and reviewer materials are intentionally not committed.

## Reproduction

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dmpnn.txt
python -m pip install -r requirements-baselines.txt
python download_tox21.py
python prepare_tox21.py
python train_dmpnn_fusion.py --seeds 42 43 44 45 46
python train_dmpnn_fusion.py --models DMPNN DMPNN-ECFP4-Gated --seeds 42 43 44 45 46
```

For a scaffold-generalization run:

```powershell
python make_tox21_scaffold_split.py
python run_dmpnn_scaffold.py --seeds 42 43 44 45 46
```

The scripts write generated files under `data/` and `results/`, which are ignored by Git. Use an NVIDIA CUDA-enabled PyTorch build when available; CPU execution remains supported.

## Dataset provenance

Tox21 is downloaded from the public DeepChem MoleculeNet mirror declared in `download_tox21.py`. Check the source terms before redistribution.
