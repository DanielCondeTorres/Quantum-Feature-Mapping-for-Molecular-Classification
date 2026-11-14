"""
tests/test_atom_selection.py
-----------------------------
Unit tests for the atom_selection module.

Run with: pytest molecular_encoding/tests/
"""

import pytest
from rdkit import Chem

from molecular_encoding.preprocessing import preprocess_smiles
from molecular_encoding.atom_selection import assign_qubits, Strategy, describe_assignment


# --- Test molecules ---
IBUPROFEN   = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"      # 18 heavy atoms, has scaffold
CAFFEINE    = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"       # 14 heavy atoms, 2 fused rings
BENZENE     = "c1ccccc1"                           # 6 heavy atoms, all scaffold
WATER       = "O"                                  # 1 heavy atom — edge case
ETHANOL     = "CCO"                                # 3 heavy atoms, no scaffold


class TestPreprocessing:
    def test_canonical_smiles_is_deterministic(self):
        s1 = preprocess_smiles("OC(=O)c1ccccc1")   # benzoic acid, variant 1
        s2 = preprocess_smiles("c1ccc(cc1)C(=O)O") # same molecule, different SMILES
        assert s1 == s2, "Canonical SMILES must be identical for the same molecule"

    def test_salt_removal(self):
        s = preprocess_smiles("CC(=O)O.[Na+]")    # sodium acetate
        assert s is not None
        assert "Na" not in s, "Salt (Na+) should be removed"

    def test_too_large_filtered(self):
        # A very large molecule (should fail MW filter at default 600 Da)
        big = "C" * 60
        s = preprocess_smiles(big, max_mw=600)
        # linear alkane C60 MW ≈ 844 Da → should be None
        assert s is None


class TestAssignment:
    def test_all_strategies_produce_12_qubits_circuit(self):
        for strat in Strategy:
            clean = preprocess_smiles(IBUPROFEN)
            a = assign_qubits(clean, strategy=strat)
            assert a is not None
            assert len(a.backbone_indices) == 10
            assert a.n_tail == 2

    def test_canonical_rank_is_smiles_invariant(self):
        """The canonical rank strategy must produce the same assignment
        regardless of which SMILES representation is used."""
        s1 = "OC(=O)c1ccccc1"
        s2 = "c1ccc(cc1)C(=O)O"
        c1 = preprocess_smiles(s1)
        c2 = preprocess_smiles(s2)
        a1 = assign_qubits(c1, strategy=Strategy.CANONICAL_RANK)
        a2 = assign_qubits(c2, strategy=Strategy.CANONICAL_RANK)
        assert a1 is not None and a2 is not None
        # After canonicalization, c1 == c2, so assignments should be equal
        assert a1.backbone_indices == a2.backbone_indices

    def test_scaffold_atoms_in_backbone_first(self):
        """For SCAFFOLD_CENTRALITY, all scaffold atoms should appear in the
        backbone (if the scaffold has ≤ n_backbone atoms)."""
        clean = preprocess_smiles(IBUPROFEN)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
        assert a is not None
        # All scaffold atoms should be in the backbone (Ibuprofen scaffold = benzene, 6 atoms)
        scaffold_in_backbone = a.scaffold_atoms.issubset(set(a.backbone_indices))
        assert scaffold_in_backbone, (
            f"Scaffold atoms {a.scaffold_atoms} not all in backbone {a.backbone_indices}"
        )

    def test_random_strategy_uses_seed(self):
        """Random strategy with the same seed must be reproducible."""
        clean = preprocess_smiles(CAFFEINE)
        a1 = assign_qubits(clean, strategy=Strategy.RANDOM, random_seed=0)
        a2 = assign_qubits(clean, strategy=Strategy.RANDOM, random_seed=0)
        a3 = assign_qubits(clean, strategy=Strategy.RANDOM, random_seed=99)
        assert a1.backbone_indices == a2.backbone_indices
        assert a1.backbone_indices != a3.backbone_indices

    def test_small_molecule_no_tail(self):
        """Benzene has only 6 heavy atoms — no tail atoms should exist."""
        clean = preprocess_smiles(BENZENE)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY, n_backbone=10)
        assert a is not None
        assert len(a.tail_indices) == 0
        assert len(a.backbone_indices) == 6  # only 6 atoms, not padded

    def test_describe_assignment_runs(self):
        clean = preprocess_smiles(CAFFEINE)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
        desc = describe_assignment(a)
        assert "Strategy" in desc
        assert "qubit" in desc.lower() or "Qubit" in desc


class TestCircuitBuilder:
    def test_circuit_has_correct_qubit_count(self):
        from molecular_encoding.circuit_builder import build_circuit
        clean = preprocess_smiles(IBUPROFEN)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
        qc = build_circuit(a)
        assert qc.num_qubits == 12

    def test_circuit_has_measure_all(self):
        from molecular_encoding.circuit_builder import build_circuit
        clean = preprocess_smiles(CAFFEINE)
        a = assign_qubits(clean, strategy=Strategy.CANONICAL_RANK)
        qc = build_circuit(a)
        # measure_all adds a barrier + measurements
        assert qc.num_clbits == 12

    def test_all_tail_modes(self):
        from molecular_encoding.circuit_builder import build_circuit
        clean = preprocess_smiles(IBUPROFEN)
        for mode in ["sum", "stats", "pharm"]:
            a  = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
            qc = build_circuit(a, tail_mode=mode)
            assert qc.num_qubits == 12
