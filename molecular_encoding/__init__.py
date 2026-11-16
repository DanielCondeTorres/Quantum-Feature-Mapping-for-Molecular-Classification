"""
molecular_encoding
==================
Molecular representation module for QAMP 2027.

Implements atom-to-qubit assignment strategies independent of SMILES ordering,
for the 10+2 quantum feature map (QuantumCircuit with N_BACKBONE + N_TAIL qubits).
"""

from .atom_selection import assign_qubits, Strategy
from .preprocessing import preprocess_smiles
from .composite_descriptor import compute_composite_bbb_angles
from .capacity_analysis import backbone_retention_profile, dataset_backbone_summary

def encode_molecule(smiles: str, strategy: "Strategy" = None, **kwargs):
    """
    Full pipeline: SMILES → cleaned molecule → qubit assignment → QuantumCircuit.

    Parameters
    ----------
    smiles   : str      Raw SMILES string.
    strategy : Strategy Atom ordering strategy (default: SCAFFOLD_CENTRALITY).
    **kwargs           Passed to build_circuit (n_backbone, n_tail, alpha, prop).

    Returns
    -------
    qc       : QuantumCircuit  Ready to transpile and run.
    meta     : dict            Assignment metadata for analysis/visualization.
    """
    from .atom_selection import Strategy as S
    if strategy is None:
        strategy = S.CANONICAL_RANK

    clean_smiles = preprocess_smiles(smiles)
    if clean_smiles is None:
        return None, None

    assignment = assign_qubits(clean_smiles, strategy=strategy)
    if assignment is None:
        return None, None

    # Keep the chemistry-only part of the package usable on machines where
    # Qiskit is intentionally not installed (e.g. data-preparation workers).
    from .circuit_builder import build_circuit
    qc = build_circuit(assignment, **kwargs)
    return qc, assignment


def __getattr__(name):
    """Lazily expose the optional Qiskit circuit builder."""
    if name == "build_circuit":
        from .circuit_builder import build_circuit
        return build_circuit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["encode_molecule", "assign_qubits", "Strategy", "build_circuit", "preprocess_smiles", "compute_composite_bbb_angles", "backbone_retention_profile", "dataset_backbone_summary"]
