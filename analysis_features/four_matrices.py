"""One figure: the four confusion matrices the user asked for.

    python -m analysis_features four-matrices

Four arms on the full cleaned 1955-molecule table, identical circuit (10+2,
published alternating tail), folds and RBF SVM C=10:

1. ``paper_smiles_order`` - first ten atoms in RDKit SMILES-parse order.
2. ``random_backbone``    - ten randomly chosen heavy atoms (seed 0).
3. ``importance``         - top ten by the fixed RDKit BBBP score.
4. ``community_importance`` - best atom per modularity community, then top score.

Only the atom-to-qubit selection changes. Output:
``results/confusion_matrices_four_arms.png`` and ``results/four_matrices.csv``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark import RESULTS_DIR, load_frame, scaffold_groups
from .embedding import embed
from .random_qubit_test import Evaluation, scaffold_eval

ARMS = (
    ("paper_smiles_order", "source", False, 0),
    ("random_backbone", "source_permuted", False, 0),
    ("importance", "importance", True, 0),
    ("community_importance", "community_importance", True, 0),
)


def _features(frame: pd.DataFrame, order: str, strip_salts: bool, seed: int) -> np.ndarray:
    return np.asarray(
        [
            embed(
                smiles,
                order=order,
                strip_salts=strip_salts,
                tail_mode="paper",
                random_seed=seed,
            ).features
            for smiles in (
                frame["source_smiles"] if not strip_salts else frame["canonical_smiles"]
            )
        ]
    )


def run(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])

    results: dict[str, Evaluation] = {}
    rows = []
    for name, order, strip_salts, seed in ARMS:
        print(f"[{name}] ...", flush=True)
        X = _features(frame, order, strip_salts, seed)
        result = scaffold_eval(X, y, groups)
        results[name] = result
        print(
            f"  accuracy {result.accuracy:.4f} | balanced {result.balanced_accuracy:.4f} "
            f"| AUC {result.auc:.4f}",
            flush=True,
        )
        rows.append(
            {
                "arm": name,
                "order": order,
                "accuracy": round(result.accuracy, 4),
                "balanced_accuracy": round(result.balanced_accuracy, 4),
                "auc": round(result.auc, 4),
                "TN": int(result.matrix[0, 0]),
                "FP": int(result.matrix[0, 1]),
                "FN": int(result.matrix[1, 0]),
                "TP": int(result.matrix[1, 1]),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "four_matrices.csv", index=False)
    _plot(results, outdir / "confusion_matrices_four_arms.png")
    return table


def _plot(results: dict[str, Evaluation], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for ax, (name, result) in zip(axes.ravel(), results.items()):
        matrix = result.matrix
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                colour = "white" if value > matrix.max() / 2 else "black"
                ax.text(col, row, str(value), ha="center", va="center", color=colour, fontsize=15)
        ax.set_xticks([0, 1], ["non-permeable", "permeable"])
        ax.set_yticks([0, 1], ["non-permeable", "permeable"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"{name}\n"
            f"AUC {result.auc:.3f} | accuracy {result.accuracy:.3f} | "
            f"bal.acc {result.balanced_accuracy:.3f}",
            fontsize=11,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "10+2 quantum circuit on cleaned BBBP (1955 molecules)\n"
        "Only the atom-to-qubit selection changes; identical folds and SVM",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print(run().to_string(index=False))
