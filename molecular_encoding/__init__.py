"""
molecular_encoding
==================
Molecular representation module for QAMP 2027.

Implements chemistry-aware atom selection and global tail summaries for the
10+2 quantum feature map. ``encode_molecule`` defaults to both new policies.
"""

from .atom_selection import assign_qubits, Strategy
from .preprocessing import preprocess_smiles
from .composite_descriptor import compute_composite_bbb_angles
from .capacity_analysis import backbone_retention_profile, dataset_backbone_summary
from .backbone_selection import BackboneSelection, select_backbone_atoms
from .tail_summaries import TailSummary, summarize_tail
from .graph_pooling import PooledMolecule, gate_operations, pool_molecule

def encode_molecule(
    smiles: str,
    strategy: "Strategy" = Strategy.CHEMICAL_COVERAGE,
    *,
    n_backbone: int = 10,
    n_tail: int = 2,
    random_seed: int = 42,
    **circuit_kwargs,
):
    """
    Full pipeline: SMILES → cleaned molecule → qubit assignment → QuantumCircuit.

    Parameters
    ----------
    smiles   : str      Raw SMILES string.
    strategy : Strategy Atom ordering strategy (default: CHEMICAL_COVERAGE).
    n_backbone : int    Maximum individually encoded atoms (default 10).
    n_tail : int        Number of summary qubits (default 2).
    random_seed : int   Seed for the RANDOM negative-control strategy.
    **circuit_kwargs    Passed to build_circuit (for example ``prop`` or
                        ``tail_mode``).

    Returns
    -------
    qc       : QuantumCircuit  Ready to transpile and run.
    meta     : QubitAssignment Assignment metadata for analysis/visualization.
    """
    clean_smiles = preprocess_smiles(smiles)
    if clean_smiles is None:
        return None, None

    assignment = assign_qubits(
        clean_smiles,
        strategy=strategy,
        n_backbone=n_backbone,
        n_tail=n_tail,
        random_seed=random_seed,
    )
    if assignment is None:
        return None, None

    # Keep the chemistry-only part of the package usable on machines where
    # Qiskit is intentionally not installed (e.g. data-preparation workers).
    from .circuit_builder import build_circuit
    qc = build_circuit(assignment, **circuit_kwargs)
    return qc, assignment


def encode_graph_pooled_molecule(smiles: str):
    """Build the experimental connected-fragment 10+2 molecular embedding.

    Unlike :func:`encode_molecule`, backbone wires represent pooled connected
    atom clusters rather than individual selected atoms. The original encoder
    remains the default and is unchanged.
    """
    from .graph_pooling import build_graph_pooled_circuit

    return build_graph_pooled_circuit(smiles)


def __getattr__(name):
    """Lazily expose the optional Qiskit circuit builder."""
    if name == "build_circuit":
        from .circuit_builder import build_circuit
        return build_circuit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "encode_molecule", "assign_qubits", "Strategy", "build_circuit",
    "preprocess_smiles", "compute_composite_bbb_angles",
    "backbone_retention_profile", "dataset_backbone_summary",
    "BackboneSelection", "select_backbone_atoms", "TailSummary", "summarize_tail",
    "PooledMolecule", "pool_molecule", "gate_operations",
    "encode_graph_pooled_molecule",
]
