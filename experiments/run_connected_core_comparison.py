"""Connected-core backbone versus SMILES order and chemical coverage.

Run from the repository root:

    python -m experiments.run_connected_core_comparison

Every arm uses the same 250-molecule BBBP sample, the same ten stratified
splits, exact statevector simulation, and the 78-feature Zi/ZiZj readout.
The connected-core arms change the atom-to-qubit map. One arm also replaces
Pauling electronegativity with atomic TPSA, and one arm uses the paper's
alternating electronegativity tail instead of the global two-channel tail.

Outputs are written to ``connected_core_{per_split,summary,connectivity,settings}``
in the results directory. The nested-C connected-core arms include splits where
inner CV selects C=0.1 and the held-out predictions collapse to one class
(F1=0, accuracy ~0.49); see the method README before interpreting their means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.common import (
    DEFAULT_READOUT,
    evaluate_nested_c,
    evaluate_svm,
    prepare_unique_balanced_data,
    statevector_features,
)
from molecular_encoding.analysis.capacity_analysis import backbone_connectivity_summary
from molecular_encoding.chemistry.atom_selection import Strategy
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


ARMS = (
    ("SMILES_ORDER__pauling__global", Strategy.SMILES_ORDER, "pauling_en", "global"),
    ("CANONICAL_TRAVERSAL__pauling__global", Strategy.CANONICAL_TRAVERSAL, "pauling_en", "global"),
    ("CHEMICAL_COVERAGE__pauling__global", Strategy.CHEMICAL_COVERAGE, "pauling_en", "global"),
    ("CONNECTED_CORE__pauling__global", Strategy.CONNECTED_CORE, "pauling_en", "global"),
    ("CONNECTED_CORE__tpsa__global", Strategy.CONNECTED_CORE, "tpsa_contrib", "global"),
    ("CONNECTED_CORE__pauling__sum", Strategy.CONNECTED_CORE, "pauling_en", "sum"),
)


def run_comparison(
    dataset: Path,
    per_class: int = 125,
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Score the connected-core map on the same folds as the SMILES control."""
    data, data_info = prepare_unique_balanced_data(dataset, per_class, seed)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = list(
        StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=seed
        ).split(np.zeros(len(y)), y)
    )
    connectivity = backbone_connectivity_summary(
        smiles,
        strategies=(
            Strategy.SMILES_ORDER,
            Strategy.CANONICAL_TRAVERSAL,
            Strategy.CHEMICAL_COVERAGE,
            Strategy.CONNECTED_CORE,
        ),
    )

    records = []
    for name, strategy, prop, tail_mode in ARMS:
        print(f"Generating {DEFAULT_READOUT} features for {name}...")
        features = np.stack([
            statevector_features(item, strategy, prop=prop, tail_mode=tail_mode)
            for item in smiles
        ])
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, seed, f"{name}__nested_C"))

    fold_metrics = pd.DataFrame(records)
    summary_rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        row = {"method": method}
        for metric in ("accuracy", "f1", "auc_roc"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    settings = {
        **data_info,
        "dataset": str(dataset),
        "per_class": per_class,
        "n_splits": n_splits,
        "test_size": test_size,
        "seed": seed,
        "readout": DEFAULT_READOUT,
        "arms": [
            {"method": name, "strategy": strategy.value, "prop": prop, "tail_mode": tail_mode}
            for name, strategy, prop, tail_mode in ARMS
        ],
        "classifier_fixed": "StandardScaler + SVC(RBF, C=10)",
        "classifier_nested": "C chosen from {0.1, 1, 10, 100, 1000} by 3-fold AUC inside each training split",
    }
    return fold_metrics, summary, connectivity, settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("BBBP.csv"))
    parser.add_argument("--per-class", type=int, default=125)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("molecular_encoding/results"))
    args = parser.parse_args()

    fold_metrics, summary, connectivity, settings = run_comparison(
        args.dataset, args.per_class, args.splits, args.test_size, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "connected_core_per_split.csv"
    summary_path = args.output_dir / "connected_core_summary.csv"
    connectivity_path = args.output_dir / "connected_core_connectivity.csv"
    settings_path = args.output_dir / "connected_core_settings.json"
    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    connectivity.to_csv(connectivity_path, index=False)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print("\nConnectivity of the ten-atom backbone:")
    print(connectivity.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPer-method metrics (mean ± sample SD):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {fold_path}, {summary_path}, {connectivity_path}, {settings_path}")


if __name__ == "__main__":
    main()
