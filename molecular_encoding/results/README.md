# Experiment results

This directory stores checked-in outputs from the modular BBBP experiments.
Each experiment writes a per-fold/per-split CSV, an aggregated summary CSV,
and/or a JSON file describing the data and evaluation settings. The matching
module in `experiments/` reproduces those files when run from the repository
root, e.g. `python -m experiments.run_selection_comparison`.

| Prefix | Experiment |
|---|---|
| `selection_vs_smiles_*` | Chemical coverage versus source-SMILES atom order |
| `embedding_ablation_*` | Atom properties and tail-summary ablations |
| `pauli_readout_*` | Full probability, Zi, ZZ, and combined readout comparisons |
| `robustness_*` | Scaffold-held-out and randomized-SMILES evaluations |
| `graph_pooling_comparison_*` | Experimental connected graph pooling |
| `multifeature_comparison_*` | Multi-property encoding and selector comparison |
| `pca_bbbp_*` | PCA-derived auxiliary-feature experiment |

The CSV files are descriptive benchmark artifacts, not independent test sets;
see [`../README.md`](../README.md) for sample sizes, validation design, and
limitations before interpreting the scores.
