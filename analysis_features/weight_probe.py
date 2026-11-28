"""Sensitivity of the fixed BBBP importance weights.

    python -m analysis_features weight-probe

The weights in ``community_importance`` are expert priors, not fitted to the
labels. This test asks how much the exact numbers matter: it evaluates the
default weights, equal weights and 20 random weight vectors that also sum to
one, on the same 1955 molecules, folds, circuit and SVM.

Outputs: ``results/weight_sensitivity.csv`` and ``.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .atom_order import _IMPORTANCE_WEIGHTS
from .benchmark import RESULTS_DIR, load_frame, scaffold_groups
from .embedding import embed
from .random_qubit_test import scaffold_eval

NAMES = tuple(_IMPORTANCE_WEIGHTS)


def run(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    rng = np.random.default_rng(42)

    configurations: list[tuple[str, dict[str, float]]] = [
        ("expert_priors", dict(_IMPORTANCE_WEIGHTS)),
        ("equal_weights", {name: 1.0 / len(NAMES) for name in NAMES}),
    ]
    for seed in range(20):
        values = rng.dirichlet(np.ones(len(NAMES)))
        configurations.append(
            (f"random_{seed}", dict(zip(NAMES, (float(v) for v in values))))
        )

    rows = []
    for name, weights in configurations:
        print(f"[{name}] ...", flush=True)
        X = np.asarray(
            [
                embed(
                    smiles,
                    order="community_importance",
                    strip_salts=True,
                    tail_mode="paper",
                    selection_weights=weights,
                ).features
                for smiles in frame["canonical_smiles"]
            ]
        )
        result = scaffold_eval(X, y, groups)
        rows.append(
            {
                "configuration": name,
                "weights": json.dumps({k: round(v, 3) for k, v in weights.items()}),
                "accuracy": round(result.accuracy, 4),
                "balanced_accuracy": round(result.balanced_accuracy, 4),
                "auc": round(result.auc, 4),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "weight_sensitivity.csv", index=False)
    random_rows = table[table["configuration"].str.startswith("random_")]
    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "folds": 5,
            "fold_seed": 42,
            "note": "weights are fixed priors, never optimised on BBBP labels",
        },
        "expert_priors": dict(_IMPORTANCE_WEIGHTS),
        "equal_weights_auc": float(table.loc[1, "auc"]),
        "random_auc_mean": round(float(random_rows["auc"].mean()), 4),
        "random_auc_std": round(float(random_rows["auc"].std(ddof=1)), 4),
        "random_auc_min": float(random_rows["auc"].min()),
        "random_auc_max": float(random_rows["auc"].max()),
        "rows": json.loads(table.to_json(orient="records")),
    }
    (outdir / "weight_sensitivity.json").write_text(json.dumps(payload, indent=2))
    print(
        f"\nexpert {table.loc[0, 'auc']:.4f} | equal {table.loc[1, 'auc']:.4f} | "
        f"random {payload['random_auc_mean']:.4f} ± {payload['random_auc_std']:.4f} "
        f"[{payload['random_auc_min']:.4f}, {payload['random_auc_max']:.4f}]"
    )
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
