"""Atom-ordering strategies for the ten backbone qubits.

The cleaning pipeline is order-agnostic; the circuit is not. This module makes
the ordering choice explicit and reproducible:

``canonical``
    First ten heavy atoms of RDKit's canonical SMILES.
``source``
    First ten heavy atoms of the string as written (the published baseline).
``chemical``
    Greedy coverage of Morgan environments and chemical roles, ported from the
    ``dct_rdkit`` selector so the two branches can be compared directly.
``chemical_canonical_order``
    The chemical-coverage *set*, but ordered by canonical atom index. Used as
    a diagnostic: it separates "which atoms" from "which qubit".
``random``
    The canonical heavy-atom order rotated to a random starting atom.
``source_random``
    The source heavy-atom order rotated to a random starting atom (keeps the
    original string's traversal, including any counter-ions).
``canonical_shuffled``
    The first ten canonical atoms in a random qubit order.

No strategy looks at the label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Mapping

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdMolDescriptors

from .config import N_BACKBONE


@dataclass(frozen=True)
class BackboneSelection:
    """Selected atoms in qubit order, plus the remaining tail atoms."""

    selected_atoms: tuple[int, ...]
    tail_atoms: tuple[int, ...]
    scores: Mapping[int, float]
    reasons: Mapping[int, tuple[str, ...]]


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


def heavy_indices(mol: Chem.Mol) -> list[int]:
    """Non-hydrogen atom indices in the molecule's current order."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


@lru_cache(maxsize=1)
def _feature_factory():
    from pathlib import Path

    feature_file = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(feature_file))


def atom_role_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Inspectable chemical roles, with the same tokens as ``dct_rdkit``."""
    donors: set[int] = set()
    acceptors: set[int] = set()
    factory = _feature_factory()
    with_hydrogens = Chem.AddHs(mol)
    for feature in factory.GetFeaturesForMol(with_hydrogens):
        atom_ids = {i for i in feature.GetAtomIds() if i < mol.GetNumAtoms()}
        if feature.GetFamily() == "Donor":
            donors.update(atom_ids)
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(atom_ids)

    result: dict[int, set[str]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        tokens: set[str] = set()
        if atom.GetFormalCharge() != 0:
            tokens.add("role:charged")
        if idx in donors:
            tokens.add("role:donor")
        if idx in acceptors:
            tokens.add("role:acceptor")
        if atom.GetAtomicNum() != 6:
            tokens.add("role:heteroatom")
        if atom.GetIsAromatic():
            tokens.add("role:aromatic")
        if atom.IsInRing():
            tokens.add("role:ring")
        result[idx] = tokens
    return result


def _morgan_environment_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Radius 0/1/2 Morgan environments keyed by atom index."""
    bit_info: dict[int, list[tuple[int, int]]] = {}
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
            result[atom.GetIdx()]
    return dict(result)


def _token_weights(roles, environments, atom_indices) -> dict[str, float]:
    frequencies = Counter(
        token for atom_idx in atom_indices for token in environments[atom_idx]
    )
    weights: dict[str, float] = {}
    for token in _all_tokens(roles, environments, atom_indices):
        if token.startswith("morgan:"):
            weights[token] = 1.0 / sqrt(frequencies[token])
        elif token.startswith("role:"):
            weights[token] = _ROLE_WEIGHTS[token.split(":", 1)[1]]
        else:
            weights[token] = 0.6
    return weights


def _all_tokens(roles, environments, atom_indices):
    for atom_idx in atom_indices:
        yield from roles[atom_idx]
        yield from environments[atom_idx]


def betweenness_centrality(mol: Chem.Mol) -> dict[int, float]:
    """Unweighted betweenness on the heavy-atom graph (NetworkX definition)."""
    import networkx as nx

    graph = nx.Graph()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            graph.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetAtomicNum() > 1 and right.GetAtomicNum() > 1:
            graph.add_edge(left.GetIdx(), right.GetIdx())

    if graph.number_of_nodes() > 1:
        return nx.betweenness_centrality(graph, normalized=True, weight=None)
    return {node: 0.0 for node in graph.nodes()}


def _centrality_scale(mol, centrality, heavy_atoms) -> float:
    supplied_max = max((centrality or {}).values(), default=0.0)
    if supplied_max > 0:
        return supplied_max
    return max(
        (mol.GetAtomWithIdx(i).GetDegree() for i in heavy_atoms), default=1
    ) or 1


def _normalized_centrality(mol, centrality, scale, atom_idx) -> float:
    if centrality is None:
        return mol.GetAtomWithIdx(atom_idx).GetDegree() / scale
    return centrality.get(atom_idx, 0.0) / scale


def select_backbone_atoms(
    mol: Chem.Mol,
    n_backbone: int = N_BACKBONE,
    centrality: Mapping[int, float] | None = None,
) -> BackboneSelection:
    """Greedy coverage of Morgan environments and chemical roles.

    This is the ``CHEMICAL_COVERAGE`` rule from the ``dct_rdkit`` branch,
    reproduced here so the comparison does not depend on that package. The
    weights are fixed structural priors, never fitted to BBBP labels.
    """
    if n_backbone <= 0:
        raise ValueError("n_backbone must be a positive integer")

    heavy_atoms = heavy_indices(mol)
    if not heavy_atoms:
        return BackboneSelection((), (), {}, {})

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    roles = atom_role_tokens(mol)
    environments = _morgan_environment_tokens(mol)
    token_weights = _token_weights(roles, environments, heavy_atoms)
    max_centrality = _centrality_scale(mol, centrality, heavy_atoms)

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
            centrality_value = _normalized_centrality(
                mol, centrality, max_centrality, atom_idx
            )
            neighbors_selected = sum(
                neighbor.GetIdx() in selected_set
                for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                if neighbor.GetAtomicNum() > 1
            )
            degree = max(1, mol.GetAtomWithIdx(atom_idx).GetDegree())
            connectivity_value = neighbors_selected / degree

            score = (
                coverage_gain
                + _CENTRALITY_WEIGHT * centrality_value
                + _CONNECTIVITY_WEIGHT * connectivity_value
            )
            if score > best_score or (
                score == best_score
                and (best_atom is None or ranks[atom_idx] < ranks[best_atom])
            ):
                best_atom = atom_idx
                best_score = score
                best_new_tokens = new_tokens

        assert best_atom is not None
        selected.append(best_atom)
        selected_set.add(best_atom)
        covered.update(roles[best_atom] | environments[best_atom])
        scores[best_atom] = float(best_score)
        reasons[best_atom] = best_new_tokens

    tail = tuple(
        sorted(
            (idx for idx in heavy_atoms if idx not in selected_set),
            key=ranks.__getitem__,
        )
    )
    return BackboneSelection(tuple(selected), tail, scores, reasons)


def chemical_coverage_backbone(
    mol: Chem.Mol, n_backbone: int = N_BACKBONE
) -> BackboneSelection:
    """Chemical-coverage selection with the same centrality as ``dct_rdkit``."""
    return select_backbone_atoms(
        mol, n_backbone, betweenness_centrality(mol)
    )


def random_start_backbone(
    mol: Chem.Mol, seed: int, n_backbone: int = N_BACKBONE
) -> tuple[int, ...]:
    """Random heavy atom as q0, then the cyclic atom order of the molecule."""
    heavy = heavy_indices(mol)
    if not heavy:
        return ()
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(heavy)))
    order = [heavy[(start + offset) % len(heavy)] for offset in range(len(heavy))]
    return tuple(order[:n_backbone])


def shuffled_backbone(
    mol: Chem.Mol, seed: int, n_backbone: int = N_BACKBONE
) -> tuple[int, ...]:
    """First ten canonical atoms, with their qubit order randomly permuted."""
    heavy = heavy_indices(mol)
    backbone = list(heavy[:n_backbone])
    rng = np.random.default_rng(seed)
    rng.shuffle(backbone)
    return tuple(backbone)
