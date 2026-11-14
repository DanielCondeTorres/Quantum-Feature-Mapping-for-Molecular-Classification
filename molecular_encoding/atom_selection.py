"""
atom_selection.py
-----------------
Atom-to-qubit assignment strategies for the 10+2 quantum feature map.

The key design decision in this module:
  Given a molecule with N atoms, which 10 atoms go to the backbone qubits
  (0–9, individually resolved) and which go to the tail (compressed into
  qubits 10–11)?

Four strategies are implemented, plus the original SMILES-order baseline.

SMILES-INVARIANCE NOTE
----------------------
All strategies except SMILES_ORDER produce the same assignment regardless of
how the SMILES string is written for a given molecule. This is enforced by
working from the molecule object (after canonical parsing) and using
graph-theoretic or RDKit-canonical rankings, never string indices.

GRAPH REPRESENTATION
--------------------
Internally, each molecule is converted to a NetworkX graph:

    G = (V, E)
    V = {atom_idx: {symbol, atomic_num, ...}}
    E = {(i, j): {bond_order, is_aromatic, is_in_ring}}

Centrality measures (betweenness, degree) are computed on this graph to
rank atoms by their structural importance. See _build_mol_graph().
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Strategy(enum.Enum):
    """Atom-to-qubit assignment strategies."""
    SMILES_ORDER         = "smiles_order"         # Original / baseline (SMILES-dependent)
    CANONICAL_RANK       = "canonical_rank"        # RDKit Morgan rank (SMILES-invariant, highest accuracy)
    PHARMACOPHORE_FIRST  = "pharmacophore_first"   # Heteroatoms & H-bonding prioritized
    SCAFFOLD_CENTRALITY  = "scaffold_centrality"   # Murcko scaffold first, then betweenness
    RANDOM               = "random"                # Negative control (shuffled)


@dataclass
class QubitAssignment:
    """
    Result of assigning a molecule's atoms to circuit qubits.

    Attributes
    ----------
    smiles          : canonical SMILES used for assignment
    n_backbone      : number of backbone qubits (default 10)
    n_tail          : number of tail / summary qubits (default 2)
    backbone_indices: list of RDKit atom indices mapped to qubits 0 .. n_backbone-1
                      backbone_indices[i] = atom index assigned to qubit i
    tail_indices    : list of RDKit atom indices compressed into tail qubits
    atom_to_qubit   : dict {atom_idx: qubit_id} for all backbone atoms
    mol             : rdkit.Chem.Mol object (used by downstream modules)
    strategy        : Strategy used to produce this assignment
    graph           : NetworkX DiGraph of the molecule (for visualization)
    centrality      : dict {atom_idx: betweenness_centrality} (all atoms)
    scaffold_atoms  : set of atom indices that are part of the Murcko scaffold
    """
    smiles          : str
    n_backbone      : int
    n_tail          : int
    backbone_indices: List[int]
    tail_indices    : List[int]
    atom_to_qubit   : dict
    mol             : object          # rdkit.Chem.Mol
    strategy        : Strategy
    graph           : object = None   # networkx.Graph
    centrality      : dict = field(default_factory=dict)
    scaffold_atoms  : set  = field(default_factory=set)


def assign_qubits(
    smiles: str,
    strategy: Strategy = Strategy.SCAFFOLD_CENTRALITY,
    n_backbone: int = 10,
    n_tail: int = 2,
    random_seed: int = 42,
) -> QubitAssignment | None:
    """
    Assign molecule atoms to qubit indices for the (n_backbone + n_tail) circuit.

    Parameters
    ----------
    smiles      : str       Canonical SMILES string (run preprocess_smiles first).
    strategy    : Strategy  One of the four ordering strategies.
    n_backbone  : int       Number of individually-resolved qubits (default 10).
    n_tail      : int       Number of tail / summary qubits (default 2).
    random_seed : int       Used only for Strategy.RANDOM.

    Returns
    -------
    QubitAssignment | None  None if the molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumHeavyAtoms()

    # Build molecular graph (always, for visualization and centrality)
    G, centrality = _build_mol_graph(mol)

    # Identify Murcko scaffold atoms (always, for SCAFFOLD strategy and metadata)
    scaffold_atom_set = _get_scaffold_atoms(mol)

    # --- Select backbone atom indices by strategy ---
    if strategy == Strategy.SMILES_ORDER:
        ordered = [atom.GetIdx() for atom in mol.GetAtoms()]

    elif strategy == Strategy.CANONICAL_RANK:
        ranks = list(Chem.CanonicalRankAtoms(mol))
        ordered = sorted(range(n_atoms), key=lambda i: ranks[i])

    elif strategy == Strategy.PHARMACOPHORE_FIRST:
        ordered = _pharmacophore_order(mol)

    elif strategy == Strategy.SCAFFOLD_CENTRALITY:
        ordered = _scaffold_centrality_order(mol, scaffold_atom_set, centrality)

    elif strategy == Strategy.RANDOM:
        ordered = list(range(n_atoms))
        rng = random.Random(random_seed)
        rng.shuffle(ordered)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Split into backbone (first n_backbone) and tail (the rest)
    backbone_indices = ordered[:n_backbone]
    tail_indices     = ordered[n_backbone:]

    atom_to_qubit = {atom_idx: qubit for qubit, atom_idx in enumerate(backbone_indices)}

    return QubitAssignment(
        smiles           = smiles,
        n_backbone       = n_backbone,
        n_tail           = n_tail,
        backbone_indices = backbone_indices,
        tail_indices     = tail_indices,
        atom_to_qubit    = atom_to_qubit,
        mol              = mol,
        strategy         = strategy,
        graph            = G,
        centrality       = centrality,
        scaffold_atoms   = scaffold_atom_set,
    )


# ---------------------------------------------------------------------------
# Internal: Molecular Graph
# ---------------------------------------------------------------------------

def _build_mol_graph(mol) -> Tuple[nx.Graph, dict]:
    """
    Build a NetworkX Graph from an RDKit molecule.

    Nodes carry: symbol, atomic_num, is_in_ring, is_aromatic.
    Edges carry: bond_order (double counts as 2.0, aromatic as 1.5), is_aromatic.

    Returns the graph and betweenness centrality dict keyed by atom index.
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            symbol      = atom.GetSymbol(),
            atomic_num  = atom.GetAtomicNum(),
            is_in_ring  = atom.IsInRing(),
            is_aromatic = atom.GetIsAromatic(),
        )

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_order  = bond.GetBondTypeAsDouble(),
            is_aromatic = bond.GetIsAromatic(),
        )

    # Betweenness centrality: fraction of shortest paths passing through each node.
    # Atoms that "bridge" parts of the molecule score highest.
    # weight=None → topological (unweighted); weight='bond_order' → chemical.
    if G.number_of_nodes() > 1:
        centrality = nx.betweenness_centrality(G, normalized=True, weight=None)
    else:
        centrality = {n: 0.0 for n in G.nodes()}

    return G, centrality


def _get_scaffold_atoms(mol) -> set:
    """
    Return the set of RDKit atom indices that belong to the Murcko scaffold.
    Returns an empty set if the molecule has no rings (acyclic molecules).
    """
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        match = mol.GetSubstructMatch(scaffold_mol)
        return set(match)
    except Exception:
        return set()


def _pharmacophore_order(mol) -> List[int]:
    """
    Order atoms by pharmacophoric relevance for membrane/BBB permeability:
      - Heteroatoms (N, O, S, P, halogens) get top priority (Lipinski polar groups)
      - Hydrogen bond donors (HBD) and acceptors (HBA) prioritized
      - Graph degree used as secondary tie-breaker
    """
    scores = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        is_hetero = 1 if sym not in ("C", "H") else 0
        is_donor = 1 if (sym in ("N", "O") and atom.GetTotalNumHs() > 0) else 0
        is_acceptor = 1 if sym in ("N", "O", "F") else 0
        degree = atom.GetDegree()
        score = is_hetero * 10 + is_donor * 5 + is_acceptor * 3 + degree * 1
        scores.append((score, atom.GetIdx()))

    scores.sort(key=lambda x: -x[0])
    return [idx for _, idx in scores]


def _scaffold_centrality_order(mol, scaffold_atoms: set, centrality: dict) -> List[int]:
    """
    Order atoms for SCAFFOLD_CENTRALITY strategy:

      1. Scaffold atoms, sorted by betweenness centrality (descending).
         → These go into the first N_scaffold backbone qubits.
      2. Non-scaffold atoms, sorted by betweenness centrality (descending).
         → These fill the remaining backbone slots and become the tail.

    WHY THIS ORDER?
    - The Murcko scaffold is the rigid, ring-containing core of a drug molecule.
      For BBB permeability, the scaffold typically contains the aromatic rings
      and heteroatoms that drive membrane partitioning.
    - Within both groups, betweenness centrality ranks atoms by their structural
      importance: bridgehead atoms, atoms connecting rings to chains, etc.
    - This ensures the most topologically and chemically important atoms always
      receive individually resolved backbone qubits, regardless of SMILES order.
    """
    all_indices   = [atom.GetIdx() for atom in mol.GetAtoms()]
    scaffold_part = sorted(
        [i for i in all_indices if i in scaffold_atoms],
        key=lambda i: -centrality.get(i, 0.0),
    )
    non_scaffold  = sorted(
        [i for i in all_indices if i not in scaffold_atoms],
        key=lambda i: -centrality.get(i, 0.0),
    )
    return scaffold_part + non_scaffold


# ---------------------------------------------------------------------------
# Utility: summary for debugging / logging
# ---------------------------------------------------------------------------

def describe_assignment(assignment: QubitAssignment) -> str:
    """
    Return a human-readable summary of a QubitAssignment.
    Useful for debugging and for the collision analysis.
    """
    mol = assignment.mol
    lines = [
        f"Strategy : {assignment.strategy.value}",
        f"SMILES   : {assignment.smiles}",
        f"Atoms    : {mol.GetNumHeavyAtoms()} heavy atoms",
        f"Backbone : {assignment.n_backbone} qubits  "
        f"({len(assignment.backbone_indices)} assigned)",
        f"Tail     : {len(assignment.tail_indices)} atoms → "
        f"{assignment.n_tail} summary qubits",
        "",
        "Qubit  Atom  Symbol  Centrality  In-Scaffold?",
        "-----  ----  ------  ----------  ------------",
    ]
    for qubit, atom_idx in enumerate(assignment.backbone_indices):
        atom   = mol.GetAtomWithIdx(atom_idx)
        c      = assignment.centrality.get(atom_idx, 0.0)
        in_sc  = "✓" if atom_idx in assignment.scaffold_atoms else "·"
        lines.append(
            f"  {qubit:2d}    {atom_idx:3d}   {atom.GetSymbol():2s}      "
            f"{c:.4f}      {in_sc}"
        )

    if assignment.tail_indices:
        tail_syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in assignment.tail_indices]
        lines.append(f"\nTail atoms: {tail_syms} → qubits {assignment.n_backbone}–"
                     f"{assignment.n_backbone + assignment.n_tail - 1}")
    return "\n".join(lines)
