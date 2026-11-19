"""Shared data, simulation, and scoring utilities for benchmark scripts.

Experiment modules should contain study-specific orchestration only. This
module centralizes the benchmark sample definition, the exact NumPy simulator,
and the fixed/nested RBF-SVM scorers used across the comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from molecular_encoding.chemistry.atom_selection import Strategy, assign_qubits
from molecular_encoding.chemistry.properties import (
    get_atom_properties,
    molecular_size_angle,
)
from molecular_encoding.chemistry.tail_summaries import summarize_tail
from molecular_encoding.quantum.readout import DEFAULT_READOUT, features_from_probabilities


N_BACKBONE = 10
N_TAIL = 2
N_QUBITS = N_BACKBONE + N_TAIL
ALPHA = 0.1


def _apply_ry(state: np.ndarray, theta: float, qubit: int) -> None:
    """Apply Qiskit's real Ry(theta) matrix in little-endian qubit order."""
    indices = np.arange(state.size)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    a0, a1 = state[zero].copy(), state[one].copy()
    cosine, sine = np.cos(theta / 2.0), np.sin(theta / 2.0)
    state[zero] = cosine * a0 - sine * a1
    state[one] = sine * a0 + cosine * a1


def _apply_rxx(state: np.ndarray, theta: float, q1: int, q2: int) -> None:
    """Apply ``exp(-i theta XX / 2)`` by paired bit flips."""
    indices = np.arange(state.size)
    partner = indices ^ (1 << q1) ^ (1 << q2)
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
    bit1, bit2 = (indices >> q1) & 1, (indices >> q2) & 1
    parity = 1 - 2 * (bit1 ^ bit2)
    state *= np.exp(-0.5j * theta * parity)


def statevector_probabilities(
    smiles: str,
    strategy: Strategy,
    prop: str = "pauling_en",
    tail_mode: str = "global",
    tail_angles: tuple[float, float] | None = None,
    size_channel: str = "none",
    size_weight: float = 0.4,
) -> np.ndarray:
    """Return exact NumPy statevector probabilities for the 10+2 circuit.

    ``size_channel`` optionally blends a bounded molecular-length angle into
    the q10 tail rotation; it is supported only with ``tail_mode='global'``.
    The default ``'none'`` leaves the established embedding untouched.
    """
    assignment = assign_qubits(smiles, strategy=strategy, n_backbone=N_BACKBONE, n_tail=N_TAIL)
    if assignment is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
    state[0] = 1.0
    if prop == "multi_physchem":
        from molecular_encoding.quantum.multifeature import multifeature_gate_operations

        for name, angle, q1, *other in multifeature_gate_operations(assignment):
            if name in {"rx", "ry", "rz"}:
                _apply_single_axis(state, angle, q1, name)
            elif name == "rxx":
                _apply_rxx(state, angle, q1, other[0])
            elif name == "rzz":
                _apply_rzz(state, angle, q1, other[0])
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()

    if prop.startswith("angle_mix:"):
        from molecular_encoding.quantum.angle_embedding import mixed_atom_angles

        channels = tuple(prop.removeprefix("angle_mix:").split("+"))
        atom_properties = mixed_atom_angles(assignment, channels)
    else:
        atom_properties = get_atom_properties(assignment, prop=prop)
    for qubit, atom_idx in enumerate(assignment.backbone_indices):
        value = atom_properties[atom_idx]
        angle = ALPHA * value**2 if prop == "pauling_en" else value
        _apply_ry(state, angle, qubit)

    tail = assignment.tail_indices
    if size_channel != "none" and tail_mode != "global":
        raise ValueError("size_channel is only supported with tail_mode='global'")
    size_angle = 0.0 if size_channel == "none" else molecular_size_angle(assignment.mol, size_channel)

    if tail and tail_mode == "global":
        if tail_angles is None:
            summary = summarize_tail(assignment.mol, assignment.backbone_indices)
            tail_angles = summary.chemical_angle, summary.structural_angle
        chemical_angle, structural_angle = float(tail_angles[0]), float(tail_angles[1])
        if size_channel != "none":
            # q10 keeps its chemical meaning; q11 keeps topology/interface.
            chemical_angle = (1.0 - size_weight) * chemical_angle + size_weight * size_angle
        _apply_ry(state, chemical_angle, N_BACKBONE)
        _apply_ry(state, structural_angle, N_BACKBONE + 1)
    elif size_channel != "none" and tail_mode == "global":
        # No tail atoms: q10 would be idle, so it now carries the length signal.
        _apply_ry(state, size_weight * size_angle, N_BACKBONE)
    elif tail and tail_mode == "sum":
        for indices, qubit in ((tail[0::2], N_BACKBONE), (tail[1::2], N_BACKBONE + 1)):
            values = [atom_properties.get(idx, 0.0) for idx in indices]
            if values:
                angle = (ALPHA / 2.0) * sum(values) if prop == "pauling_en" else float(np.mean(values))
                _apply_ry(state, angle, qubit)
    elif tail and tail_mode == "stats":
        values = np.asarray([atom_properties.get(idx, 0.0) for idx in tail])
        _apply_ry(state, float(values.mean()), N_BACKBONE)
        _apply_ry(state, min(float(values.std()), np.pi / 2.0), N_BACKBONE + 1)
    elif tail and tail_mode == "pharm":
        donors = acceptors = 0
        for idx in tail:
            atom = assignment.mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            donors += int(symbol in ("N", "O") and atom.GetTotalNumHs() > 0)
            acceptors += int(symbol in ("N", "O", "F"))
        _apply_ry(state, min(donors / 10.0, 1.0) * np.pi, N_BACKBONE)
        _apply_ry(state, min(acceptors / 10.0, 1.0) * np.pi, N_BACKBONE + 1)
    elif tail_mode not in {"global", "sum", "stats", "pharm"}:
        raise ValueError(f"Unknown tail mode: {tail_mode}")

    tail_set = set(tail)
    for bond in assignment.mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        q_left, q_right = assignment.atom_to_qubit.get(left), assignment.atom_to_qubit.get(right)
        theta = bond.GetBondTypeAsDouble() / 2.0
        if q_left is not None and q_right is not None:
            _apply_rxx(state, theta, q_left, q_right)
        elif q_left is not None and right in tail_set:
            tail_qubit = N_BACKBONE + 1 if tail_mode == "global" else (
                N_BACKBONE if tail.index(right) % 2 == 0 else N_BACKBONE + 1
            )
            _apply_rxx(state, theta, q_left, tail_qubit)
        elif q_right is not None and left in tail_set:
            tail_qubit = N_BACKBONE + 1 if tail_mode == "global" else (
                N_BACKBONE if tail.index(left) % 2 == 0 else N_BACKBONE + 1
            )
            _apply_rxx(state, theta, q_right, tail_qubit)

    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def statevector_features(
    smiles: str,
    strategy: Strategy,
    prop: str = "pauling_en",
    tail_mode: str = "global",
    *,
    readout: str = DEFAULT_READOUT,
    tail_angles: tuple[float, float] | None = None,
    size_channel: str = "none",
    size_weight: float = 0.4,
) -> np.ndarray:
    """Simulate one molecule and extract a selected measurement readout."""
    probabilities = statevector_probabilities(
        smiles, strategy, prop=prop, tail_mode=tail_mode, tail_angles=tail_angles,
        size_channel=size_channel, size_weight=size_weight,
    )
    return features_from_probabilities(probabilities, readout=readout)[0]


def prepare_unique_balanced_data(path: Path, per_class: int, seed: int):
    """Canonical-deduplicate BBBP, remove label conflicts, and sample balanced data."""
    frame = pd.read_csv(path).dropna(subset=["p_np", "smiles"]).copy()
    frame["p_np"] = frame["p_np"].astype(int)
    frame = frame[frame["p_np"].isin((0, 1))]
    canonical, valid_rows = [], []
    for row_idx, row in frame.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None or len(str(row["smiles"])) >= 70:
            continue
        valid_rows.append(row_idx)
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    frame = frame.loc[valid_rows].copy()
    frame["canonical_smiles"] = canonical
    label_counts = frame.groupby("canonical_smiles")["p_np"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    frame = frame[~frame["canonical_smiles"].isin(conflicts)]
    frame = frame.drop_duplicates("canonical_smiles", keep="first")
    class_counts = frame["p_np"].value_counts()
    if any(class_counts.get(label, 0) < per_class for label in (0, 1)):
        raise ValueError(f"Need {per_class} unique molecules per class; available: {class_counts.to_dict()}")
    balanced = pd.concat([
        frame[frame["p_np"] == label].sample(per_class, random_state=seed)
        for label in (0, 1)
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced, {
        "n_valid_rows_before_deduplication": len(valid_rows),
        "n_conflicting_structures_removed": len(conflicts),
        "n_unique_structures": len(frame),
        "n_balanced_molecules": len(balanced),
    }


def evaluate_svm(features: np.ndarray, y: np.ndarray, splits, method: str):
    """Score standardized RBF-SVM with fixed ``C=10`` on precomputed folds."""
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_idx])
        test_features = scaler.transform(features[test_idx])
        classifier = SVC(kernel="rbf", C=10.0).fit(train_features, y[train_idx])
        predictions = classifier.predict(test_features)
        decision = classifier.decision_function(test_features)
        rows.append({
            "split": split_id,
            "method": method,
            "accuracy": accuracy_score(y[test_idx], predictions),
            "f1": f1_score(y[test_idx], predictions, zero_division=0),
            "auc_roc": roc_auc_score(y[test_idx], decision),
        })
    return rows


def evaluate_nested_c(features: np.ndarray, y: np.ndarray, splits, seed: int, method: str):
    """Tune SVM ``C`` only within each training fold, then score held-out data."""
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + split_id)
        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf")),
            {"svc__C": (0.1, 1.0, 10.0, 100.0, 1000.0)},
            scoring="roc_auc", cv=inner_cv, refit=True, n_jobs=1,
        ).fit(features[train_idx], y[train_idx])
        predictions = search.predict(features[test_idx])
        decision = search.decision_function(features[test_idx])
        rows.append({
            "split": split_id,
            "method": method,
            "accuracy": accuracy_score(y[test_idx], predictions),
            "f1": f1_score(y[test_idx], predictions, zero_division=0),
            "auc_roc": roc_auc_score(y[test_idx], decision),
            "selected_c": search.best_params_["svc__C"],
        })
    return rows
