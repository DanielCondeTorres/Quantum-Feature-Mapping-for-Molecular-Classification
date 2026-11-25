"""Audit: does the 10+2 circuit distinguish stereoisomers?

Run from the repository root:

    python -m experiments.audit_stereo_collisions

RDKit keeps stereochemistry in canonical isomeric SMILES (the @/@@ tags are
present), but the feature map uses Pauling electronegativity and bond order,
which are identical for enantiomers. This script measures, on the cleaned full
dataset, how many stereoisomer forms collapse to the same feature vector and
how often those colliding forms carry opposite BBBP labels.

Outputs: analysis/stereo_collision/ (one CSV table, one PNG figure, README).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from molecular_encoding.chemistry.atom_selection import assign_qubits, Strategy
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")
CLEAN = Path("data/BBBP_clean_full.csv")
OUT = Path("analysis/stereo_collision")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
CSV = TABLES / "stereo_collisions.csv"


def _flatten(smiles: str) -> str:
    return Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CLEAN)
    labels = dict(zip(frame["canonical_smiles"], frame["label"]))

    groups = defaultdict(list)
    for smiles in frame["canonical_smiles"]:
        groups[_flatten(smiles)].append(smiles)

    print("Computing feature hashes per stereoisomer...")
    rows = []
    for flat, members in groups.items():
        if len(members) < 2 or not any("@" in s for s in members):
            continue
        old = [statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in members]
        chem = [statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib") for s in members]
        for smiles, old_x, chem_x in zip(members, old, chem):
            backbone = list(assign_qubits(smiles, Strategy.CHEMICAL_COVERAGE).backbone_indices)
            backbone_flat = list(assign_qubits(flat, Strategy.CHEMICAL_COVERAGE).backbone_indices)
            rows.append({
                "flat_smiles": flat,
                "canonical_smiles": smiles,
                "label": int(labels[smiles]),
                "n_forms": len(members),
                "old_features_identical": bool(np.allclose(old[0], old_x)),
                "chem_features_identical": bool(np.allclose(chem[0], chem_x)),
                "selector_changes_without_stereo": backbone != backbone_flat,
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(CSV, index=False)

    summary = {
        "stereo_forms_analysed": len(audit),
        "old_features_identical": int(audit["old_features_identical"].sum()),
        "chem_features_identical": int(audit["chem_features_identical"].sum()),
        "selector_changes_without_stereo": int(audit["selector_changes_without_stereo"].sum()),
        "forms_in_skeletons_with_opposite_labels": int(
            audit.groupby("flat_smiles")["label"].transform("nunique").gt(1).sum()
        ),
    }
    (OUT / "settings.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["old_features_identical", "chem_features_identical", "selector_changes_without_stereo"]
    vals = [summary[k] for k in keys]
    ax.bar(keys, vals, color=["#C44E52", "#C44E52", "#55A868"])
    for bar, value in zip(ax.containers[0], vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(value), ha="center")
    ax.set_ylabel("number of forms")
    ax.set_title(f"Feature-map blindness to stereo ({len(audit)} forms)")
    fig.tight_layout()
    fig.savefig(FIGURES / "stereo_collisions.png", dpi=160)
    plt.close(fig)

    print("\n".join(f"  {k}: {v}" for k, v in summary.items()))
    print(f"\nSaved: {CSV} and figure")


if __name__ == "__main__":
    main()
