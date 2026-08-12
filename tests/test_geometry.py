"""Tests for the nanorod geometry functions.
This module tests the functions inside the nanorods/geometry.py module.
"""

import numpy as np
import pytest

from nanorods.geometry import sub2ind, capsule_surface_offsets, capsule_radius_at_x, nanorod_neighbors


def test_sub2ind():
    """Check the conversion from 2D lattice indices to linear indices."""
    assert sub2ind((3, 4), 0, 0) == 0
    assert sub2ind((3, 4), 1, 0) == 4
    assert sub2ind((3, 4), 2, 3) == 11


def test_capsule_invalid_length():
    """Check that a nanorod with L < d raises an error."""
    with pytest.raises(ValueError):
        capsule_surface_offsets(10e-9, 20e-9, 2)


def test_capsule_surface_symmetry():
    """Check the shape and reflection symmetry of the surface sites."""
    offsets, n_sites = capsule_surface_offsets(50e-9, 25e-9, 2)

    assert offsets.shape == (n_sites, 2)
    assert n_sites%6 == 0
    assert np.allclose(offsets[:n_sites//2, 0], offsets[n_sites//2:, 0][::-1])
    assert np.allclose(offsets[:n_sites//2, 1], -offsets[n_sites//2:, 1][::-1])


def test_capsule_radius_at_center():
    """Check that the local radius at the nanorod center is d/2."""
    d = 25e-9
    assert np.isclose(capsule_radius_at_x(0, 50e-9, d), d/2)


def test_nanorod_neighbors():
    """Check the neighbor pattern for even and odd lattice columns."""
    assert nanorod_neighbors(0, 0) == [(1, 0), (0, 1), (-1, 1)]
    assert nanorod_neighbors(0, 1) == [(1, 0), (1, 1), (0, 1)]