"""Molecule preparation, atom selection, and chemical descriptors."""

from .atom_selection import QubitAssignment, Strategy, assign_qubits, describe_assignment
from .backbone_selection import BackboneSelection, select_backbone_atoms
from .composite_descriptor import compute_composite_bbb_angles
from .preprocessing import preprocess_dataset, preprocess_smiles
from .properties import get_atom_properties
from .tail_summaries import TailSummary, summarize_tail

__all__ = [
    "BackboneSelection", "QubitAssignment", "Strategy", "TailSummary",
    "assign_qubits", "compute_composite_bbb_angles", "describe_assignment",
    "get_atom_properties", "preprocess_dataset", "preprocess_smiles",
    "select_backbone_atoms", "summarize_tail",
]
