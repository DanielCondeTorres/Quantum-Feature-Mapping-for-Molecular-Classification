"""Ablate fixed-scale single-Ry atomic properties on the same BBBP splits.

Run from the repository root with ``python -m experiments.run_angle_mixture_ablation``.
Every arm keeps atom selection, graph entanglement, tail summary, readout,
sample, and outer splits fixed; only the scalar local Ry angle changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.common import (
    evaluate_nested_c, evaluate_svm, prepare_unique_balanced_data, statevector_features,
)
from molecular_encoding.chemistry.atom_selection import Strategy


CONFIGS = (
    ("TPSA", "tpsa_contrib"),
    ("Pauling", "pauling_en"),
    ("Gasteiger", "gasteiger_charge"),
    ("Crippen_logP", "crippen_logp"),
    ("Pauling+TPSA", "pauling_en+tpsa_contrib"),
    ("Pauling+Gasteiger", "pauling_en+gasteiger_charge"),
    ("TPSA+Crippen_logP", "tpsa_contrib+crippen_logp"),
    ("Pauling+TPSA+Gasteiger", "pauling_en+tpsa_contrib+gasteiger_charge"),
    ("Pauling+TPSA+Crippen_logP", "pauling_en+tpsa_contrib+crippen_logp"),
    ("Pauling+TPSA+Gasteiger+Crippen", "pauling_en+tpsa_contrib+gasteiger_charge+crippen_logp"),
)


def run(output: Path = Path("molecular_encoding/results")) -> pd.DataFrame:
    data, info = prepare_unique_balanced_data(Path("BBBP.csv"), 125, 42)
    smiles = data["smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    splits = list(StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=42).split(y, y))
    records = []
    for name, channel_spec in CONFIGS:
        prop = "angle_mix:" + channel_spec
        print(f"Generating {name} angles...", flush=True)
        X = np.stack([statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop=prop) for s in smiles])
        records.extend(evaluate_svm(X, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(X, y, splits, 42, f"{name}__nested_C"))

    # Reference controls from the same data and splits.
    for name, strategy, prop in (
        ("SMILES_order_original_Pauling", Strategy.SMILES_ORDER, "pauling_en"),
        ("Chemical_coverage_original_Pauling", Strategy.CHEMICAL_COVERAGE, "pauling_en"),
    ):
        X = np.stack([statevector_features(s, strategy, prop=prop) for s in smiles])
        records.extend(evaluate_svm(X, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(X, y, splits, 42, f"{name}__nested_C"))

    per_split = pd.DataFrame(records)
    summary = per_split.groupby("method", sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
    ).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(output / "angle_mixture_ablation_per_split.csv", index=False)
    summary.to_csv(output / "angle_mixture_ablation_summary.csv", index=False)
    settings = {
        **info, "n_splits": 10, "test_size": .25, "seed": 42,
        "selector": "CHEMICAL_COVERAGE for all mixture arms; same source SMILES control",
        "readout": "exact NumPy statevector; 78 features (12 Zi + 66 ZiZj)",
        "outer_classifier": "StandardScaler + RBF SVC (fixed C=10 and nested C)",
        "nested_C_grid": [.1, 1, 10, 100, 1000],
        "angle_rule": "theta_i = pi * weighted_mean(bounded atom channels); equal weights",
        "channel_scales": {
            "Pauling": "electronegativity / 4",
            "TPSA": "RDKit atomic TPSA contribution / 35 A^2",
            "Gasteiger": "clip((partial_charge + 1)/2, 0, 1)",
            "Crippen_logP": "clip((atomic_logP_contribution + 1.5)/3, 0, 1)",
        },
        "feature_channels": {name: spec.split("+") for name, spec in CONFIGS},
        "data": info,
        "interpretation": "same circuit except selected atom Ry angles; no data-fitted transforms or label-informed weights",
    }
    (output / "angle_mixture_ablation_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary


if __name__ == "__main__":
    run()
