"""Chemistry-aware selection of atoms for a fixed-width circuit backbone.

This module selects a *set* of atoms for individual encoding. It does not use
RDKit canonical atom ranks as an importance score. Instead, a deterministic
greedy coverage objective favors atoms that add new local Morgan environments,
chemical roles, and useful graph connectivity.

The score is intentionally task-agnostic and uses only one molecule. For a
task-specific selector (for example BBB permeability), its fixed weights must
be justified and compared through a held-out ablation; they must not be tuned
on test labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@dataclass(frozen=True)
class BackboneSelection:
    """Auditable result of the greedy backbone selection.

    ``selected_atoms`` is ordered by selection step and maps directly to
    backbone qubits. ``tail_atoms`` contains all remaining heavy atoms in
    canonical-rank order. ``reasons`` records the feature tokens each selected
    atom contributed at the time it was selected.
    """

    selected_atoms: tuple[int, ...]
    tail_atoms: tuple[int, ...]
    scores: Mapping[int, float]
    reasons: Mapping[int, tuple[str, ...]]


# Fixed, transparent weights. They are not fitted to BBBP labels.
_ROLE_WEIGHTS = {
    "charged": 3.0,
    "donor": 2.5,
    "acceptor": 2.0,
    "heteroatom": 1.25,
    "aromatic": 0.75,
    "ring": 0.50,
}
_CONNECTIVITY_WEIGHT = 0.20
_CENTRALITY_WEIGHT = 0.15


def select_backbone_atoms(
    mol: Chem.Mol,
    n_backbone: int = 10,
    centrality: Mapping[int, float] | None = None,
    *,
    physicochemical_coverage: bool = False,
    distance_diversity: bool = False,
) -> BackboneSelection:
    """Select atoms by marginal chemical-environment coverage.

    The objective is deterministic after canonical parsing and has three
    parts:

    1. Cover distinct radius-0/1/2 Morgan environments, weighted by their
       within-molecule rarity so a repeated environment has diminishing value.
    2. Cover explicit chemical-role tokens (charge, donor/acceptor, heteroatom,
       aromaticity, and ring membership).
    3. Give small bonuses to graph-central atoms and atoms adjacent to an
       already selected atom, which helps retain useful connectivity.

    Canonical ranks are used only as a stable tie-break. If the molecule has
    fewer heavy atoms than the requested width, every heavy atom is selected.
    Hydrogen atoms are not assigned their own qubits.

    Parameters
    ----------
    mol:
        Sanitized RDKit molecule. The caller should canonicalize/reparse it
        when SMILES-independent indices are required.
    n_backbone:
        Maximum number of atoms to select; must be positive.
    centrality:
        Optional node-centrality values keyed by RDKit atom index. If omitted,
        no external graph package is needed and a simple degree proxy is used.

    Returns
    -------
    BackboneSelection
        Selected atoms in qubit order, remaining atoms, marginal scores, and
        the features responsible for each selection.
    """
    if n_backbone <= 0:
        raise ValueError("n_backbone must be a positive integer")

    heavy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy_atoms:
        return BackboneSelection((), (), {}, {})

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    roles = atom_role_tokens(mol)
    environments = _morgan_environment_tokens(mol)
    # Add coarse, fixed-boundary physicochemical tokens to Morgan/role coverage.
    # This encourages the ten wires to retain atoms with distinct polarity,
    # charge, and lipophilicity rather than spending the whole budget on
    # repeated carbon environments. These bins are structural priors, not
    # fitted to BBBP labels.
    if physicochemical_coverage:
        environments = _add_physchem_tokens(mol, environments)

    frequencies = Counter(
        token for atom_idx in heavy_atoms for token in environments[atom_idx]
    )
    token_weights: dict[str, float] = {}
    for token in _all_tokens(roles, environments, heavy_atoms):
        if token.startswith("morgan:"):
            # Diminishing value for common/repeated local environments.
            token_weights[token] = 1.0 / sqrt(frequencies[token])
        elif token.startswith("role:"):
            token_weights[token] = _ROLE_WEIGHTS[token.split(":", 1)[1]]
        else:
            token_weights[token] = 0.6

    max_centrality = max((centrality or {}).values(), default=0.0)
    if max_centrality <= 0:
        max_centrality = max(
            (mol.GetAtomWithIdx(i).GetDegree() for i in heavy_atoms), default=1
        ) or 1

    selected: list[int] = []
    selected_set: set[int] = set()
    covered: set[str] = set()
    scores: dict[int, float] = {}
    reasons: dict[int, tuple[str, ...]] = {}

    while len(selected) < min(n_backbone, len(heavy_atoms)):
        best_atom = None
        best_score = float("-inf")
        best_new_tokens: tuple[str, ...] = ()

        for atom_idx in heavy_atoms:
            if atom_idx in selected_set:
                continue
            tokens = roles[atom_idx] | environments[atom_idx]
            new_tokens = tuple(sorted(tokens - covered))
            coverage_gain = sum(token_weights[token] for token in new_tokens)

            if centrality is None:
                centrality_value = mol.GetAtomWithIdx(atom_idx).GetDegree() / max_centrality
            else:
                centrality_value = centrality.get(atom_idx, 0.0) / max_centrality

            neighbors_selected = sum(
                neighbor.GetIdx() in selected_set
                for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                if neighbor.GetAtomicNum() > 1
            )
            degree = max(1, mol.GetAtomWithIdx(atom_idx).GetDegree())
            connectivity_value = neighbors_selected / degree

            if distance_diversity and selected_set:
                distances = [
                    nx_distance(mol, atom_idx, selected_idx)
                    for selected_idx in selected_set
                ]
                distance_novelty = min(min(distances), 4) / 4.0
            elif distance_diversity:
                distance_novelty = 1.0
            if distance_diversity:
                score = (coverage_gain + _CENTRALITY_WEIGHT * centrality_value
                         + 0.10 * connectivity_value + 0.20 * distance_novelty)
            else:
                # Preserve the established selector as the unchanged default.
                score = (coverage_gain + _CENTRALITY_WEIGHT * centrality_value
                         + _CONNECTIVITY_WEIGHT * connectivity_value)
            # Exact deterministic tie-break; canonical ranks do not contribute
            # to the score and therefore are not mistaken for importance.
            if score > best_score or (
                score == best_score and (best_atom is None or ranks[atom_idx] < ranks[best_atom])
            ):
                best_atom = atom_idx
                best_score = score
                best_new_tokens = new_tokens

        assert best_atom is not None  # guarded by loop bounds
        selected.append(best_atom)
        selected_set.add(best_atom)
        covered.update(roles[best_atom] | environments[best_atom])
        scores[best_atom] = float(best_score)
        reasons[best_atom] = best_new_tokens

    tail = tuple(sorted((i for i in heavy_atoms if i not in selected_set), key=ranks.__getitem__))
    return BackboneSelection(tuple(selected), tail, scores, reasons)


def atom_role_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Return simple, inspectable chemistry roles for each heavy atom."""
    donor_atoms: set[int] = set()
    acceptor_atoms: set[int] = set()
    # RDKit's feature definitions handle common charged and aromatic cases.
    # Add explicit hydrogens for feature perception; retain only heavy atom IDs.
    factory = _feature_factory()
    feature_mol = Chem.AddHs(mol)
    for feature in factory.GetFeaturesForMol(feature_mol):
        atom_ids = {i for i in feature.GetAtomIds() if i < mol.GetNumAtoms()}
        if feature.GetFamily() == "Donor":
            donor_atoms.update(atom_ids)
        elif feature.GetFamily() == "Acceptor":
            acceptor_atoms.update(atom_ids)

    result: dict[int, set[str]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        tokens = set()
        if atom.GetFormalCharge() != 0:
            tokens.add("role:charged")
        if idx in donor_atoms:
            tokens.add("role:donor")
        if idx in acceptor_atoms:
            tokens.add("role:acceptor")
        if atom.GetAtomicNum() not in (6, 1):
            tokens.add("role:heteroatom")
        if atom.GetIsAromatic():
            tokens.add("role:aromatic")
        if atom.IsInRing():
            tokens.add("role:ring")
        result[idx] = tokens
    return result


@lru_cache(maxsize=1)
def _feature_factory():
    """Build RDKit's standard donor/acceptor feature definitions once."""
    from pathlib import Path
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    feature_file = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(feature_file))


def _morgan_environment_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Map each atom to its radius 0, 1, and 2 Morgan environment tokens."""
    bit_info: dict[int, list[tuple[int, int]]] = {}
    # The sparse Morgan fingerprint avoids fixed-size bit-vector collisions;
    # bitInfo associates each environment identifier with its center and radius.
    rdMolDescriptors.GetMorganFingerprint(
        mol, 2, bitInfo=bit_info, useChirality=True, useBondTypes=True
    )
    result: dict[int, set[str]] = defaultdict(set)
    for environment_id, occurrences in bit_info.items():
        for atom_idx, radius in occurrences:
            if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() > 1:
                result[atom_idx].add(f"morgan:r{radius}:{environment_id}")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            result[atom.GetIdx()]  # ensure a set exists, even for unusual atoms
    return dict(result)


def _add_physchem_tokens(mol: Chem.Mol, environments: dict[int, set[str]]):
    """Augment atom environments with interpretable, fixed-scale features."""
    from rdkit.Chem import AllChem, Crippen, rdMolDescriptors

    rw = Chem.RWMol(mol)
    AllChem.ComputeGasteigerCharges(rw)
    charges = []
    for atom in rw.GetAtoms():
        try:
            value = float(atom.GetDoubleProp("_GasteigerCharge"))
        except (KeyError, ValueError):
            value = 0.0
        charges.append(value if np.isfinite(value) else 0.0)
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    logp = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    result = {idx: set(tokens) for idx, tokens in environments.items()}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() <= 1:
            continue
        # Quantized bins avoid over-weighting tiny descriptor differences.
        result[idx].add(f"tpsa_bin:{min(int(max(tpsa[idx], 0) // 10), 4)}")
        result[idx].add(f"logp_bin:{int(np.clip(np.floor(logp[idx][0] + 2), 0, 4))}")
        result[idx].add(f"charge_bin:{int(np.clip(np.floor(charges[idx] + 1.5), 0, 2))}")
        result[idx].add(f"degree_bin:{min(atom.GetDegree(), 4)}")
    return result


def nx_distance(mol: Chem.Mol, left: int, right: int) -> int:
    """Small unweighted shortest-path distance, capped by caller."""
    from collections import deque

    queue = deque([(left, 0)])
    visited = {left}
    while queue:
        current, distance = queue.popleft()
        if current == right:
            return distance
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if neighbor.GetAtomicNum() > 1 and idx not in visited:
                visited.add(idx)
                queue.append((idx, distance + 1))
    return 99


def _all_tokens(roles, environments, atom_indices):
    for atom_idx in atom_indices:
        yield from roles[atom_idx]
        yield from environments[atom_idx]
