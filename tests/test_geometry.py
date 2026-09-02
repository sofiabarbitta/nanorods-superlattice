"""Tests for the nanorod geometry functions.

This module tests the functions inside the
'nanorods/geometry.py' module.
"""

import numpy as np
import pytest

from nanorods.geometry import (
    sub2ind,
    capsule_surface_offsets,
    capsule_radius_at_x,
    nanorod_neighbors,
    minimum_image,
    build_springs_by_distance,
    build_nanorod_geometry
)


def test_sub2ind():
    """
    This function tests the conversion from 2D lattice indices to
    linear indices.

    Given a lattice composed of 3 rows and 4 columns, this test converts
    different row and column indices using the `sub2ind` function. It
    then checks that the returned linear indices correspond to the
    expected positions in the flattened lattice.
    """
    assert sub2ind((3, 4), 0, 0) == 0
    assert sub2ind((3, 4), 1, 0) == 4
    assert sub2ind((3, 4), 2, 3) == 11


def test_capsule_invalid_length():
    """
    This function tests that an invalid nanorod geometry is rejected.

    Given a nanorod length smaller than its diameter, this test calls
    the `capsule_surface_offsets` function and checks that a ValueError
    is raised, since this geometry cannot represent a capsule-shaped
    nanorod.
    """
    with pytest.raises(ValueError):
        capsule_surface_offsets(10e-9, 20e-9, 2)


def test_capsule_invalid_site_density():
    """
    This function tests that the interaction-site density must be positive.

    Given a non-positive value of the interaction-site density, this
    test calls the `capsule_surface_offsets` function and checks that a
    ValueError is raised.
    """
    with pytest.raises(ValueError):
        capsule_surface_offsets(50e-9, 25e-9, 0)


def test_capsule_surface_shape():
    """
    This function tests the shape of the generated nanorod surface sites.

    Given a capsule-shaped nanorod and a positive interaction-site
    density, this test generates the surface-site positions using the
    `capsule_surface_offsets` function. It then checks that each site
    has two coordinates and that the number of generated positions
    agrees with the returned number of sites.
    """
    offsets, n_sites = capsule_surface_offsets(50e-9, 25e-9, 2)

    assert offsets.shape == (n_sites, 2)


def test_capsule_surface_number_of_sites():
    """
    This function tests that the number of interaction sites preserves
    the required lattice symmetry.

    Given a capsule-shaped nanorod, this test generates its interaction
    sites using the `capsule_surface_offsets` function. It then checks
    that the total number of sites is a multiple of six, as required
    to preserve the hexagonal symmetry of the spherical limit.
    """
    _, n_sites = capsule_surface_offsets(50e-9, 25e-9, 2)

    assert n_sites%6 == 0


def test_capsule_surface_symmetry():
    """
    This function tests the reflection symmetry of the nanorod surface.

    Given the interaction sites generated on a capsule-shaped nanorod,
    this test compares the upper and lower halves of the surface. It
    checks that corresponding sites have the same x coordinate and
    opposite y coordinates.
    """
    offsets, n_sites = capsule_surface_offsets(50e-9, 25e-9, 2)

    upper = offsets[:n_sites//2]
    lower = offsets[n_sites//2:][::-1]

    assert np.allclose(upper[:, 0], lower[:, 0])
    assert np.allclose(upper[:, 1], -lower[:, 1])


def test_capsule_radius_at_center():
    """
    This function tests the nanorod radius at its center.

    Given a capsule-shaped nanorod with diameter 25 nm, this test
    evaluates the local radius at the center of the nanorod using the
    `capsule_radius_at_x` function. It then checks that the returned
    radius is equal to half the nanorod diameter.
    """
    d = 25e-9

    radius = capsule_radius_at_x(0, 50e-9, d)

    assert np.isclose(radius, d/2)


def test_nanorod_neighbors_even_column():
    """
    This function tests the nearest-neighbor pattern for an even column.

    Given a nanorod located in an even column of the staggered lattice,
    this test determines its forward nearest neighbors using the
    `nanorod_neighbors` function. It then checks that the returned
    relative indices correspond to the expected staggered geometry.
    """
    neighbors = nanorod_neighbors(0)

    assert neighbors == [(1, 0), (0, 1), (-1, 1)]


def test_nanorod_neighbors_odd_column():
    """
    This function tests the nearest-neighbor pattern for an odd column.

    Given a nanorod located in an odd column of the staggered lattice,
    this test determines its forward nearest neighbors using the
    `nanorod_neighbors` function. It then checks that the returned
    relative indices correspond to the expected staggered geometry.
    """
    neighbors = nanorod_neighbors(1)

    assert neighbors == [(1, 0), (1, 1), (0, 1)]


def test_minimum_image_periodic_x():
    """
    This function tests the minimum-image correction along x.

    Given a separation vector with an x component larger than half the
    simulation-box length, this test applies periodic boundary conditions
    only along x. It then checks that the x component is replaced by the
    shortest equivalent separation while the y component is unchanged.
    """
    sep = np.array([[9.0, 9.0]])

    corrected = minimum_image(
        sep,
        10.0,
        10.0,
        periodic_x=True
    )

    assert np.allclose(corrected, [[-1.0, 9.0]])


def test_minimum_image_periodic_y():
    """
    This function tests the minimum-image correction along y.

    Given a separation vector with a y component larger than half the
    simulation-box length, this test applies periodic boundary conditions
    only along y. It then checks that the y component is replaced by the
    shortest equivalent separation while the x component is unchanged.
    """
    sep = np.array([[9.0, 9.0]])

    corrected = minimum_image(
        sep,
        10.0,
        10.0,
        periodic_y=True
    )

    assert np.allclose(corrected, [[9.0, -1.0]])


def test_minimum_image_periodic_xy():
    """
    This function tests the minimum-image correction along both directions.

    Given a separation vector whose x and y components are larger than
    half the corresponding simulation-box lengths, this test applies
    periodic boundary conditions along both directions. It then checks
    that both components are replaced by their shortest equivalent
    separations.
    """
    sep = np.array([[9.0, 9.0]])

    corrected = minimum_image(
        sep,
        10.0,
        10.0,
        periodic_x=True,
        periodic_y=True
    )

    assert np.allclose(corrected, [[-1.0, -1.0]])


def test_periodic_x_adds_boundary_connection():
    """
    This function tests the spring connections across a periodic x boundary.

    Given four nanorods equally spaced along x, this test constructs the
    spring network first with open boundaries and then with periodic
    boundaries along x. It checks that periodicity adds the connection
    between the first and last nanorods.
    """
    Nx, Ny = 4, 1

    pos0 = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [6.0, 0.0],
        [9.0, 0.0]
    ])

    site_offsets = np.zeros((4, 1, 2))

    open_springs = build_springs_by_distance(
        Nx, Ny,
        pos0,
        site_offsets,
        3.1,
        boxLx=12.0,
        boxLy=1.0
    )

    periodic_springs = build_springs_by_distance(
        Nx, Ny,
        pos0,
        site_offsets,
        3.1,
        boxLx=12.0,
        boxLy=1.0,
        periodic_x=True
    )

    assert len(open_springs[0]) == 3
    assert len(periodic_springs[0]) == 4


def test_periodic_y_adds_boundary_connection():
    """
    This function tests the spring connections across a periodic y boundary.

    Given four nanorods equally spaced along y, this test constructs the
    spring network first with open boundaries and then with periodic
    boundaries along y. It checks that periodicity adds the connection
    between the first and last nanorods.
    """
    Nx, Ny = 1, 4

    pos0 = np.array([
        [0.0, 0.0],
        [0.0, 3.0],
        [0.0, 6.0],
        [0.0, 9.0]
    ])

    site_offsets = np.zeros((4, 1, 2))

    open_springs = build_springs_by_distance(
        Nx, Ny,
        pos0,
        site_offsets,
        3.1,
        boxLx=1.0,
        boxLy=12.0
    )

    periodic_springs = build_springs_by_distance(
        Nx, Ny,
        pos0,
        site_offsets,
        3.1,
        boxLx=1.0,
        boxLy=12.0,
        periodic_y=True
    )

    assert len(open_springs[0]) == 3
    assert len(periodic_springs[0]) == 4


def test_periodic_x_requires_even_Nx():
    """
    This function tests the lattice requirement for periodic x boundaries.

    Given a staggered lattice with an odd number of columns, this test
    attempts to build the geometry with periodic boundary conditions
    along x. It checks that a ValueError is raised because the column
    staggering is not compatible with this periodic boundary.
    """
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