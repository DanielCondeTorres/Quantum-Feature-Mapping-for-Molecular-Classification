"""Conservative parent-fragment selection and auditable fragment reporting.

This is a deterministic heuristic, not a replacement for curated parent
selection (e.g. when two active organic components are present). Carbon-
containing fragments are preferred, then heavy-atom count, heteroatom count,
and canonical isomeric SMILES break ties. Every fragment is returned for audit.
"""
from __future__ import annotations

from rdkit import Chem


def fragment_rank(fragment: Chem.Mol) -> tuple[int, int, int, str]:
    """Rank a fragment by organic character, size, heteroatoms, stable string."""
    heavy = sum(atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms())
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
    hetero = sum(atom.GetAtomicNum() > 1 and atom.GetAtomicNum() != 6
                 for atom in fragment.GetAtoms())
    canonical = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
    return int(has_carbon), heavy, hetero, canonical


def choose_parent_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, list[dict]]:
    """Return selected parent and per-fragment audit rows.

    ``manual_review`` is true when multiple fragments tie on organic status,
    heavy-atom count and heteroatom count, i.e. the final lexical tie-break
    selected one but chemistry alone did not identify a unique parent.
    """
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        raise ValueError("Molecule has no fragments")
    ranks = [fragment_rank(fragment) for fragment in fragments]
    winner = max(range(len(fragments)), key=lambda idx: ranks[idx])
    rank_prefix = ranks[winner][:3]
    manual_review = sum(rank[:3] == rank_prefix for rank in ranks) > 1
    audit = []
    for idx, fragment in enumerate(fragments):
        audit.append({
            "fragment_smiles": ranks[idx][3],
            "heavy_atom_count": ranks[idx][1],
            "heteroatom_count": ranks[idx][2],
            "formal_charge": Chem.GetFormalCharge(fragment),
            "has_carbon": bool(ranks[idx][0]),
            "kept_as_parent": idx == winner,
            "manual_review": manual_review,
        })
    return fragments[winner], audit
