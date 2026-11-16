"""Quantify the structural information retained by a fixed atom backbone.

The 10+2 mapping is lossy only when a molecule has more than ten heavy atoms.
This module makes that loss reportable rather than implicit: tail--tail bonds
are omitted, while backbone--tail bonds remain represented through a summary
qubit.  Use it as an ablation companion when comparing backbone sizes.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .atom_selection import Strategy, assign_qubits


def backbone_retention_profile(
    smiles: str,
    strategy: Strategy = Strategy.CANONICAL_RANK,
    backbone_sizes: Iterable[int] = (6, 8, 10, 12),
) -> pd.DataFrame:
    """Return one structural-retention row per requested backbone size.

    ``represented_bond_fraction`` counts backbone--backbone and
    backbone--tail bonds, which are explicitly encoded as Rxx gates.  Bonds
    wholly inside the tail are necessarily omitted by the current 10+2 design.
    """
    rows = []
    for size in backbone_sizes:
        if size <= 0:
            raise ValueError("backbone sizes must be positive integers")
        assignment = assign_qubits(smiles, strategy=strategy, n_backbone=size)
        if assignment is None:
            continue

        backbone = set(assignment.backbone_indices)
        n_bb = n_bt = n_tt = 0
        for bond in assignment.mol.GetBonds():
            left = bond.GetBeginAtomIdx() in backbone
            right = bond.GetEndAtomIdx() in backbone
            if left and right:
                n_bb += 1
            elif left or right:
                n_bt += 1
            else:
                n_tt += 1
        n_bonds = n_bb + n_bt + n_tt
        n_atoms = assignment.mol.GetNumHeavyAtoms()
        rows.append({
            "smiles": assignment.smiles,
            "strategy": strategy.value,
            "n_backbone": size,
            "n_heavy_atoms": n_atoms,
            "n_tail_atoms": len(assignment.tail_indices),
            "atom_retention_fraction": min(size, n_atoms) / n_atoms,
            "backbone_backbone_bonds": n_bb,
            "backbone_tail_bonds": n_bt,
            "tail_tail_bonds_dropped": n_tt,
            "represented_bond_fraction": (n_bb + n_bt) / n_bonds if n_bonds else 1.0,
        })
    return pd.DataFrame(rows)


def dataset_backbone_summary(
    smiles_list: Iterable[str],
    strategy: Strategy = Strategy.CANONICAL_RANK,
    backbone_sizes: Iterable[int] = (6, 8, 10, 12),
) -> pd.DataFrame:
    """Average capacity-loss metrics across a dataset for each backbone size."""
    profiles = [backbone_retention_profile(s, strategy, backbone_sizes) for s in smiles_list]
    profiles = [profile for profile in profiles if not profile.empty]
    if not profiles:
        return pd.DataFrame()
    combined = pd.concat(profiles, ignore_index=True)
    return combined.groupby("n_backbone", as_index=False).agg(
        n_molecules=("smiles", "size"),
        mean_heavy_atoms=("n_heavy_atoms", "mean"),
        mean_tail_atoms=("n_tail_atoms", "mean"),
        mean_atom_retention=("atom_retention_fraction", "mean"),
        mean_bond_retention=("represented_bond_fraction", "mean"),
        mean_dropped_tail_bonds=("tail_tail_bonds_dropped", "mean"),
    )
