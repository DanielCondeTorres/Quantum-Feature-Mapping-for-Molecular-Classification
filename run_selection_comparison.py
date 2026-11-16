"""Paired BBBP comparison: SMILES-order versus chemistry-coverage selection.

The quantum feature map is evaluated with exact NumPy statevector evolution.
Its gates (Ry and Rxx) follow the same rotation conventions as Qiskit. This
keeps the experiment runnable in CPU environments without a working Qiskit
installation while preserving the 12-qubit probability features.

Run from the repository root:

    python run_selection_comparison.py

Defaults reproduce the repository's balanced 250-molecule, 10-split protocol.
The script stores paired per-split and aggregate metrics under
``molecular_encoding/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from molecular_encoding.atom_selection import Strategy, assign_qubits
from molecular_encoding.properties import get_atom_properties
from molecular_encoding.tail_summaries import summarize_tail


N_BACKBONE = 10
N_TAIL = 2
N_QUBITS = N_BACKBONE + N_TAIL
ALPHA = 0.1


def _apply_ry(state: np.ndarray, theta: float, qubit: int) -> None:
    """Apply Qiskit's real Ry(theta) matrix in little-endian qubit order."""
    indices = np.arange(state.size)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    a0 = state[zero].copy()
    a1 = state[one].copy()
    cosine, sine = np.cos(theta / 2.0), np.sin(theta / 2.0)
    state[zero] = cosine * a0 - sine * a1
    state[one] = sine * a0 + cosine * a1


def _apply_rxx(state: np.ndarray, theta: float, q1: int, q2: int) -> None:
    """Apply ``exp(-i theta XX / 2)`` by paired bit flips."""
    partner = np.arange(state.size) ^ (1 << q1) ^ (1 << q2)
    state[:] = np.cos(theta / 2.0) * state - 1j * np.sin(theta / 2.0) * state[partner]


def _apply_single_axis(state: np.ndarray, theta: float, qubit: int, axis: str) -> None:
    indices = np.arange(state.size)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    if axis == "rz":
        state[zero] *= np.exp(-0.5j * theta)
        state[one] *= np.exp(0.5j * theta)
        return
    a0, a1 = state[zero].copy(), state[one].copy()
    cosine, sine = np.cos(theta / 2.0), np.sin(theta / 2.0)
    if axis == "rx":
        state[zero] = cosine * a0 - 1j * sine * a1
        state[one] = -1j * sine * a0 + cosine * a1
    else:
        state[zero] = cosine * a0 - sine * a1
        state[one] = sine * a0 + cosine * a1


def _apply_rzz(state: np.ndarray, theta: float, q1: int, q2: int) -> None:
    indices = np.arange(state.size)
    bit1 = (indices >> q1) & 1
    bit2 = (indices >> q2) & 1
    parity = 1 - 2 * (bit1 ^ bit2)
    state *= np.exp(-0.5j * theta * parity)


def statevector_probabilities(
    smiles: str,
    strategy: Strategy,
    prop: str = "pauling_en",
    tail_mode: str = "global",
) -> np.ndarray:
    """Build and exactly simulate the shared 10+2 feature map for one SMILES.

    ``SMILES_ORDER`` retains RDKit's atom traversal from the source string.
    ``CHEMICAL_COVERAGE`` canonicalizes internally before selecting atoms.
    ``prop`` and ``tail_mode`` mirror the existing molecular_encoding options.
    """
    assignment = assign_qubits(
        smiles, strategy=strategy, n_backbone=N_BACKBONE, n_tail=N_TAIL
    )
    if assignment is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
    state[0] = 1.0
    if prop == "multi_physchem":
        from molecular_encoding.multifeature import multifeature_gate_operations
        for operation in multifeature_gate_operations(assignment):
            name, angle, q1, *other = operation
            if name in {"rx", "ry", "rz"}:
                _apply_single_axis(state, angle, q1, name)
            elif name == "rxx":
                _apply_rxx(state, angle, q1, other[0])
            elif name == "rzz":
                _apply_rzz(state, angle, q1, other[0])
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()
    atom_properties = get_atom_properties(assignment, prop=prop)

    for qubit, atom_idx in enumerate(assignment.backbone_indices):
        value = atom_properties[atom_idx]
        angle = ALPHA * value**2 if prop == "pauling_en" else value
        _apply_ry(state, angle, qubit)

    if assignment.tail_indices and tail_mode == "global":
        tail = summarize_tail(assignment.mol, assignment.backbone_indices)
        _apply_ry(state, tail.chemical_angle, N_BACKBONE)
        _apply_ry(state, tail.structural_angle, N_BACKBONE + 1)
    elif assignment.tail_indices and tail_mode == "sum":
        for indices, qubit in ((assignment.tail_indices[0::2], N_BACKBONE),
                               (assignment.tail_indices[1::2], N_BACKBONE + 1)):
            values = [atom_properties.get(idx, 0.0) for idx in indices]
            if values:
                angle = (ALPHA / 2.0) * sum(values) if prop == "pauling_en" else float(np.mean(values))
                _apply_ry(state, angle, qubit)
    elif assignment.tail_indices and tail_mode == "stats":
        values = np.asarray([atom_properties.get(idx, 0.0) for idx in assignment.tail_indices])
        _apply_ry(state, float(values.mean()), N_BACKBONE)
        _apply_ry(state, min(float(values.std()), np.pi / 2.0), N_BACKBONE + 1)
    elif assignment.tail_indices and tail_mode == "pharm":
        donor_count = 0
        acceptor_count = 0
        for idx in assignment.tail_indices:
            atom = assignment.mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            if symbol in ("N", "O") and atom.GetTotalNumHs() > 0:
                donor_count += 1
            if symbol in ("N", "O", "F"):
                acceptor_count += 1
        _apply_ry(state, min(donor_count / 10.0, 1.0) * np.pi, N_BACKBONE)
        _apply_ry(state, min(acceptor_count / 10.0, 1.0) * np.pi, N_BACKBONE + 1)
    elif tail_mode not in {"global", "sum", "stats", "pharm"}:
        raise ValueError(f"Unknown tail mode: {tail_mode}")

    for bond in assignment.mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        q_left = assignment.atom_to_qubit.get(left)
        q_right = assignment.atom_to_qubit.get(right)
        theta = bond.GetBondTypeAsDouble() / 2.0
        if q_left is not None and q_right is not None:
            _apply_rxx(state, theta, q_left, q_right)
        elif q_left is not None and right in assignment.tail_indices:
            tail_qubit = N_BACKBONE + 1 if tail_mode == "global" else (
                N_BACKBONE if assignment.tail_indices.index(right) % 2 == 0 else N_BACKBONE + 1
            )
            _apply_rxx(state, theta, q_left, tail_qubit)
        elif q_right is not None and left in assignment.tail_indices:
            tail_qubit = N_BACKBONE + 1 if tail_mode == "global" else (
                N_BACKBONE if assignment.tail_indices.index(left) % 2 == 0 else N_BACKBONE + 1
            )
            _apply_rxx(state, theta, q_right, tail_qubit)

    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def _prepare_unique_balanced_data(path: Path, per_class: int, seed: int):
    """Parse, canonical-deduplicate, remove label conflicts, and balance BBBP."""
    frame = pd.read_csv(path).dropna(subset=["p_np", "smiles"]).copy()
    frame["p_np"] = frame["p_np"].astype(int)
    frame = frame[frame["p_np"].isin((0, 1))]

    canonical = []
    valid_rows = []
    for row_idx, row in frame.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None or len(str(row["smiles"])) >= 70:
            continue
        valid_rows.append(row_idx)
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    frame = frame.loc[valid_rows].copy()
    frame["canonical_smiles"] = canonical

    # Exclude canonical structures with contradictory labels; collapse exact
    # duplicates before splitting to prevent train/test identity leakage.
    labels_per_structure = frame.groupby("canonical_smiles")["p_np"].nunique()
    conflicts = set(labels_per_structure[labels_per_structure > 1].index)
    frame = frame[~frame["canonical_smiles"].isin(conflicts)]
    frame = frame.drop_duplicates("canonical_smiles", keep="first")

    class_counts = frame["p_np"].value_counts()
    if any(class_counts.get(label, 0) < per_class for label in (0, 1)):
        raise ValueError(
            f"Need {per_class} unique molecules per class; available: {class_counts.to_dict()}"
        )
    balanced = pd.concat([
        frame[frame["p_np"] == label].sample(per_class, random_state=seed)
        for label in (0, 1)
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    info = {
        "n_valid_rows_before_deduplication": len(valid_rows),
        "n_conflicting_structures_removed": len(conflicts),
        "n_unique_structures": len(frame),
        "n_balanced_molecules": len(balanced),
    }
    return balanced, info


def _ecfp4_features(smiles_list: list[str]) -> np.ndarray:
    return np.asarray([
        np.asarray(AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles), 2, nBits=256
        ), dtype=np.int8)
        for smiles in smiles_list
    ])


def _evaluate_svm(X: np.ndarray, y: np.ndarray, splits, model_name: str):
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        classifier = SVC(kernel="rbf", C=10.0)
        classifier.fit(X_train, y[train_idx])
        predictions = classifier.predict(X_test)
        decision = classifier.decision_function(X_test)
        rows.append({
            "split": split_id,
            "method": model_name,
            "accuracy": accuracy_score(y[test_idx], predictions),
            "f1": f1_score(y[test_idx], predictions, zero_division=0),
            "auc_roc": roc_auc_score(y[test_idx], decision),
        })
    return rows


def run_comparison(
    dataset: Path,
    per_class: int = 125,
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run paired SVM evaluation with identical folds and model parameters."""
    data, data_info = _prepare_unique_balanced_data(dataset, per_class, seed)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed
    )
    splits = list(splitter.split(np.zeros(len(y)), y))

    print(f"Dataset: {len(data)} unique balanced molecules ({per_class}/class)")
    print("Generating exact statevector features for SMILES_ORDER...")
    X_smiles = np.stack([
        statevector_probabilities(s, Strategy.SMILES_ORDER) for s in smiles
    ])
    print("Generating exact statevector features for CHEMICAL_COVERAGE...")
    X_coverage = np.stack([
        statevector_probabilities(s, Strategy.CHEMICAL_COVERAGE) for s in smiles
    ])

    records = []
    records.extend(_evaluate_svm(X_smiles, y, splits, "SMILES_ORDER"))
    records.extend(_evaluate_svm(X_coverage, y, splits, "CHEMICAL_COVERAGE"))
    fold_metrics = pd.DataFrame(records)

    paired = fold_metrics.pivot(index="split", columns="method", values=["accuracy", "f1", "auc_roc"])
    for metric in ("accuracy", "f1", "auc_roc"):
        paired[(metric, "delta_coverage_minus_smiles")] = (
            paired[(metric, "CHEMICAL_COVERAGE")] - paired[(metric, "SMILES_ORDER")]
        )
    paired.columns = ["_".join(col).strip("_") for col in paired.columns]
    paired = paired.reset_index()

    summary_rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        row = {"method": method}
        for metric in ("accuracy", "f1", "auc_roc"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    delta_row = {"method": "paired_delta_coverage_minus_smiles"}
    for metric in ("accuracy", "f1", "auc_roc"):
        delta = paired[f"{metric}_delta_coverage_minus_smiles"]
        delta_row[f"{metric}_mean"] = delta.mean()
        delta_row[f"{metric}_std"] = delta.std(ddof=1)
        delta_row[f"{metric}_coverage_wins"] = int((delta > 0).sum())
    summary = pd.concat([summary, pd.DataFrame([delta_row])], ignore_index=True)

    settings = {
        **data_info,
        "dataset": str(dataset),
        "per_class": per_class,
        "n_splits": n_splits,
        "test_size": test_size,
        "seed": seed,
        "classifier": "StandardScaler + SVC(RBF, C=10), AUC from decision score",
        "tail_mode": "global two-channel",
        "readout": "exact 4096-state computational-basis probabilities",
    }
    return fold_metrics, summary, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("BBBP.csv"))
    parser.add_argument("--per-class", type=int, default=125)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("molecular_encoding")
    )
    args = parser.parse_args()

    fold_metrics, summary, settings = run_comparison(
        args.dataset, args.per_class, args.splits, args.test_size, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "selection_vs_smiles_per_split.csv"
    summary_path = args.output_dir / "selection_vs_smiles_summary.csv"
    settings_path = args.output_dir / "selection_vs_smiles_settings.json"
    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    settings_path.write_text(pd.Series(settings).to_json(indent=2) + "\n")

    print("\nPer-method metrics across paired splits (mean ± sample SD):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {fold_path}, {summary_path}, {settings_path}")


if __name__ == "__main__":
    main()
