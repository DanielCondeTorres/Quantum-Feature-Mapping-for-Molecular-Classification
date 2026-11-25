"""Tests for the explicit 10+2 embedding."""

from __future__ import annotations

import numpy as np
import pytest

from analysis_features.config import N_QUBITS
from analysis_features.embedding import embed

PROPRANOLOL_SALT = "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"


def test_full_readout_is_a_normalised_distribution() -> None:
    result = embed("CCO", readout="full")
    assert result.features.size == 1 << N_QUBITS
    assert result.features.sum() == pytest.approx(1.0)


def test_default_readout_has_78_pauli_moments() -> None:
    assert embed("CCO").features.size == N_QUBITS + N_QUBITS * (N_QUBITS - 1) // 2


def test_propranolol_qubit_zero_is_carbon_after_salt_stripping() -> None:
    result = embed(PROPRANOLOL_SALT)
    assert result.salt_stripped is True
    assert "Cl" not in result.parent_smiles
    assert result.qubits[0].element == "C"


def test_keeping_salts_would_put_chlorine_on_qubit_zero() -> None:
    # This is the failure mode the cleaning pipeline removes. Canonical
    # re-numbering happens to move Cl, but the published baseline keeps the
    # source atom order, where the counter-ion is literally atom 0.
    result = embed(PROPRANOLOL_SALT, strip_salts=False, order="source")
    assert result.qubits[0].element == "Cl"


def test_canonical_order_is_independent_of_input_spelling() -> None:
    first = embed("CCO", order="canonical").features
    second = embed("OCC", order="canonical").features
    assert np.allclose(first, second)


def test_source_order_is_representation_dependent() -> None:
    first = embed("CCO", order="source").features
    second = embed("OCC", order="source").features
    assert not np.allclose(first, second)


def test_source_order_matches_the_paper_baseline() -> None:
    result = embed("CCO", order="source")
    assert [q.element for q in result.qubits] == ["C", "C", "O"]


def test_enantiomers_collide_with_identical_atom_order() -> None:
    left = embed("C[C@H](N)C(=O)O", order="source").features
    right = embed("C[C@@H](N)C(=O)O", order="source").features
    assert np.allclose(left, right)


def test_paper_and_global_tail_modes_differ() -> None:
    paper = embed("CCCCCCCCCCCC", tail_mode="paper").features
    summary = embed("CCCCCCCCCCCC", tail_mode="global").features
    assert not np.allclose(paper, summary)


def test_interface_bonds_reach_both_tail_qubits_in_paper_mode() -> None:
    result = embed("CCCCCCCCCCCC", tail_mode="paper")
    targets = {right for left, right, _ in result.bond_gates if left == 9}
    assert {10, 11} <= targets


def test_tpsa_is_an_optional_property_that_changes_the_angles() -> None:
    pauling = embed("CCO", property="pauling").features
    tpsa = embed("CCO", property="tpsa").features
    assert not np.allclose(pauling, tpsa)


def test_invalid_smiles_raise() -> None:
    with pytest.raises(ValueError):
        embed("not-a-smiles")
