"""Does an explicit molecular-length channel help the 10+2 embedding?

Run from the repository root:

    python -m experiments.run_length_channel_ablation

The length channel is opt-in and reuses the existing q10 tail rotation: a
bounded angle from heavy-atom count or topological length is blended into the
chemical tail summary. For molecules with no tail, q10 would be idle, so it
now carries the length signal alone. Everything else (selector, backbone
rotations, bond gates, readout, folds, classifier) is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.common import (
    evaluate_nested_c,
    evaluate_svm,
    prepare_unique_balanced_data,
    statevector_features,
)
from molecular_encoding.chemistry.atom_selection import Strategy

# (label, atomic property, size channel, blend weight)
ARMS = (
    ("TPSA__baseline", "tpsa_contrib", "none", 0.4),
    ("TPSA__size_heavy_atoms", "tpsa_contrib", "heavy_atoms", 0.4),
    ("TPSA__size_diameter", "tpsa_contrib", "graph_diameter", 0.4),
    ("TPSA__size_radius", "tpsa_contrib", "graph_radius", 0.4),
    ("TPSA__size_diameter_w0.8", "tpsa_contrib", "graph_diameter", 0.8),
    ("Pauling__baseline", "pauling_en", "none", 0.4),
    ("Pauling__size_diameter", "pauling_en", "graph_diameter", 0.4),
)


def run_comparison(
    dataset: Path,
    per_class: int = 125,
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare the length-channel arms on the shared BBBP folds."""
    data, data_info = prepare_unique_balanced_data(dataset, per_class, seed)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = list(
        StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=seed
        ).split(np.zeros(len(y)), y)
    )

    records = []
    for name, prop, size_channel, size_weight in ARMS:
        print(f"Generating features for {name}...")
        features = np.stack([
            statevector_features(
                item, Strategy.CHEMICAL_COVERAGE, prop=prop,
                size_channel=size_channel, size_weight=size_weight,
            )
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
        "selector": Strategy.CHEMICAL_COVERAGE.value,
        "readout": "z_zz (78 Pauli moments)",
        "arms": [
            {"method": name, "prop": prop, "size_channel": channel,
             "size_weight": weight}
            for name, prop, channel, weight in ARMS
        ],
        "classifier_fixed": "StandardScaler + SVC(RBF, C=10)",
        "classifier_nested": "C chosen from {0.1, 1, 10, 100, 1000} by 3-fold AUC inside each training split",
    }
    return fold_metrics, summary, settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("BBBP.csv"))
    parser.add_argument("--per-class", type=int, default=125)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("molecular_encoding/results"))
    args = parser.parse_args()

    fold_metrics, summary, settings = run_comparison(
        args.dataset, args.per_class, args.splits, args.test_size, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output_dir / "length_channel_per_split.csv", index=False)
    summary.to_csv(args.output_dir / "length_channel_summary.csv", index=False)
    (args.output_dir / "length_channel_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    print("\nLength-channel results (mean ± sample SD over ten splits):")
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: length_channel_per_split.csv, length_channel_summary.csv, "
          "length_channel_settings.json")


if __name__ == "__main__":
    main()
