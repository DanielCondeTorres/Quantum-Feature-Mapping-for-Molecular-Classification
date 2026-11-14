"""
molecular_encoding
==================
Molecular representation module for QAMP 2027.

Implements atom-to-qubit assignment strategies independent of SMILES ordering,
for the 10+2 quantum feature map (QuantumCircuit with N_BACKBONE + N_TAIL qubits).
"""

from .atom_selection import assign_qubits, Strategy
from .circuit_builder import build_circuit
from .preprocessing import preprocess_smiles

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
        strategy = S.SCAFFOLD_CENTRALITY

    clean_smiles = preprocess_smiles(smiles)
    if clean_smiles is None:
        return None, None

    assignment = assign_qubits(clean_smiles, strategy=strategy)
    if assignment is None:
        return None, None

    qc = build_circuit(assignment, **kwargs)
    return qc, assignment


__all__ = ["encode_molecule", "assign_qubits", "Strategy", "build_circuit", "preprocess_smiles"]
