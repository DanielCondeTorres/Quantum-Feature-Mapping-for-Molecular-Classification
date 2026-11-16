"""Paired BBBP ablation of the molecular_encoding options already implemented.

Run from the repository root with ``python run_embedding_ablation.py``.
The backbone selector, data sample, folds, scaling, classifier, and exact
statevector simulator are held constant. The script compares all atom-property
embeddings with the global tail, then compares existing tail modes using the
default Pauling-electronegativity embedding.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from molecular_encoding.atom_selection import Strategy
from run_selection_comparison import (
    _evaluate_svm,
    _prepare_unique_balanced_data,
    statevector_probabilities,
)


PROPERTIES = (
    "pauling_en",
    "gasteiger_charge",
    "crippen_logp",
    "tpsa_contrib",
    "combined_bbb",
    "composite_pharmacophore",
)
TAIL_MODES = ("global", "sum", "stats", "pharm")


def run_ablation(
    dataset: Path = Path("BBBP.csv"),
    per_class: int = 125,
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
):
    data, data_info = _prepare_unique_balanced_data(dataset, per_class, seed)
    smiles = data["smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed
    )
    splits = list(splitter.split(y, y))

    # Isolate node/property embedding: same selector and global tail.
    configurations = [(prop, "global") for prop in PROPERTIES]
    # Isolate tail representation: same default node property.
    configurations.extend(("pauling_en", mode) for mode in TAIL_MODES if mode != "global")

    records = []
    for prop, tail_mode in configurations:
        name = f"{prop}__tail_{tail_mode}"
        print(f"Generating exact statevector features: {name}", flush=True)
        features = np.stack([
            statevector_probabilities(
                value,
                Strategy.CHEMICAL_COVERAGE,
                prop=prop,
                tail_mode=tail_mode,
            )
            for value in smiles
        ])
        records.extend(_evaluate_svm(features, y, splits, name))

    per_split = pd.DataFrame(records)
    summary_rows = []
    for method, group in per_split.groupby("method", sort=False):
        summary_rows.append({
            "method": method,
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(ddof=1),
            "f1_mean": group["f1"].mean(),
            "f1_std": group["f1"].std(ddof=1),
            "auc_roc_mean": group["auc_roc"].mean(),
            "auc_roc_std": group["auc_roc"].std(ddof=1),
        })
    summary = pd.DataFrame(summary_rows)

    output = Path("molecular_encoding")
    output.mkdir(exist_ok=True)
    per_split.to_csv(output / "embedding_ablation_per_split.csv", index=False)
    summary.to_csv(output / "embedding_ablation_summary.csv", index=False)
    settings = {
        **data_info,
        "dataset": str(dataset),
        "per_class": per_class,
        "n_splits": n_splits,
        "test_size": test_size,
        "seed": seed,
        "selector": Strategy.CHEMICAL_COVERAGE.value,
        "classifier": "StandardScaler + SVC(RBF, C=10)",
        "readout": "exact NumPy statevector; 4096 basis probabilities",
        "property_configurations": list(PROPERTIES),
        "tail_configurations": list(TAIL_MODES),
    }
    pd.Series(settings).to_json(output / "embedding_ablation_settings.json", indent=2)
    return per_split, summary


if __name__ == "__main__":
    _, result = run_ablation()
    print("\nMean ± sample SD across paired splits:")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
