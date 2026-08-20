"""Tests for the nanorod geometry functions.
This module tests the functions inside the nanorods/geometry.py module.
"""

import numpy as np
import pytest

from nanorods.geometry import sub2ind, capsule_surface_offsets, capsule_radius_at_x, nanorod_neighbors, minimum_image, build_springs_by_distance, build_nanorod_geometry


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
    
def test_minimum_image():
    """Check that periodic corrections are applied only along periodic directions."""
    sep = np.array([[9.0, 9.0]])

    sep_x = minimum_image(sep, 10.0, 10.0, periodic_x=True)
    sep_y = minimum_image(sep, 10.0, 10.0, periodic_y=True)
    sep_xy = minimum_image(sep, 10.0, 10.0, periodic_x=True, periodic_y=True)

    assert np.allclose(sep_x, [[-1.0, 9.0]])
    assert np.allclose(sep_y, [[9.0, -1.0]])
    assert np.allclose(sep_xy, [[-1.0, -1.0]])


def test_periodic_x_adds_boundary_connection():
    """Check that periodic x boundaries connect the first and last columns."""
    Nx, Ny = 4, 1

    pos0 = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [6.0, 0.0],
        [9.0, 0.0]
    ])

    site_offsets = np.zeros((4, 1, 2))

    open_springs = build_springs_by_distance(
        Nx, Ny, pos0, site_offsets, 3.1,
        boxLx=12.0, boxLy=1.0
    )

    periodic_springs = build_springs_by_distance(
        Nx, Ny, pos0, site_offsets, 3.1,
        boxLx=12.0, boxLy=1.0,
        periodic_x=True
    )

    assert len(open_springs[0]) == 3
    assert len(periodic_springs[0]) == 4


def test_periodic_y_adds_boundary_connection():
    """Check that periodic y boundaries connect the first and last rows."""
    Nx, Ny = 1, 4

    pos0 = np.array([
        [0.0, 0.0],
        [0.0, 3.0],
        [0.0, 6.0],
        [0.0, 9.0]
    ])

    site_offsets = np.zeros((4, 1, 2))

    open_springs = build_springs_by_distance(
        Nx, Ny, pos0, site_offsets, 3.1,
        boxLx=1.0, boxLy=12.0
    )

    periodic_springs = build_springs_by_distance(
        Nx, Ny, pos0, site_offsets, 3.1,
        boxLx=1.0, boxLy=12.0,
        periodic_y=True
    )

    assert len(open_springs[0]) == 3
    assert len(periodic_springs[0]) == 4


def test_periodic_x_requires_even_Nx():
    """Check that periodic x boundaries require an even number of columns."""
    with pytest.raises(ValueError, match="Nx must be even"):
        build_nanorod_geometry(
            Nx=3,
            Ny=4,
            L=25e-9,
            d=25e-9,
            s=3.4e-9,
            smax=4.4e-9,
            ligand_surface_density=4,
            k_lig=2.0,
            periodic_x=True
        )