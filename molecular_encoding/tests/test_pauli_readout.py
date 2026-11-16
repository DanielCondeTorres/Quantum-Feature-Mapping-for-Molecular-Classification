"""Tests for estimating Pauli-Z observables from basis probabilities."""
import numpy as np

from run_pauli_readout_comparison import pauli_z_features


def test_uniform_distribution_has_zero_nontrivial_z_moments():
    features = pauli_z_features(np.full(1 << 12, 1 / (1 << 12)))
    assert features["Z_only"].shape == (1, 12)
    assert features["ZZ_only"].shape == (1, 66)
    assert features["Z_plus_ZZ"].shape == (1, 78)
    np.testing.assert_allclose(features["Z_plus_ZZ"], 0.0, atol=1e-12)


def test_basis_state_moments_are_bit_parities():
    probabilities = np.zeros(1 << 12)
    probabilities[5] = 1.0  # bits q0 and q2 are one
    features = pauli_z_features(probabilities)
    np.testing.assert_allclose(features["Z_only"][0, :4], [-1, 1, -1, 1])
    # Pair ordering starts (0,1), (0,2), (0,3).
    np.testing.assert_allclose(features["ZZ_only"][0, :3], [-1, 1, -1])
