"""Tests for the nanorod simulation functions."""

import numpy as np
import pytest

from nanorods.simulation import (
    nanorod_moment_of_inertia,
    line_displacement,
    compute_accelerations,
    run_simulation
)


def test_moment_of_inertia():
    """
    Given the masses and dimensions of a nanorod, this test calculates
    its moment of inertia and checks that the expected value is returned.
    """
    result = nanorod_moment_of_inertia(
        m_cylinder=2.0,
        m_sphere=1.0,
        r=3.0,
        l=4.0
    )

    assert np.isclose(result, 19.266666666666666)


@pytest.mark.parametrize(
    "direction, mode, expected_indices, axis",
    [
        ("parallel", "longitudinal", [2, 6, 10], 0),
        ("parallel", "transverse", [2, 6, 10], 1),
        ("perpendicular", "longitudinal", [4, 5, 6, 7], 1),
        ("perpendicular", "transverse", [4, 5, 6, 7], 0),
    ]
)
def test_line_displacement(direction, mode, expected_indices, axis):
    """
    Given a small lattice and a selected propagation direction and mode,
    this test applies a line displacement and checks that only the
    expected central row or column is displaced along the correct axis.
    """
    nx = 4
    ny = 3
    amplitude = 0.5

    pos0 = np.zeros((nx*ny, 2))

    pos, _, _, _ = line_displacement(
        pos0,
        nx,
        ny,
        amplitude,
        direction,
        mode
    )

    expected = pos0.copy()
    expected[expected_indices, axis] += amplitude

    assert np.allclose(pos, expected)


@pytest.mark.parametrize(
    "direction, mode",
    [
        ("invalid", "longitudinal"),
        ("parallel", "invalid"),
    ]
)
def test_line_displacement_invalid_input(direction, mode):
    """
    Given an invalid propagation direction or excitation mode, this test
    checks that the line displacement function raises a ValueError.
    """
    pos0 = np.zeros((12, 2))

    with pytest.raises(ValueError):
        line_displacement(
            pos0,
            4,
            3,
            0.5,
            direction,
            mode
        )


def test_equilibrium_acceleration():
    """
    Given two nanorods connected by a spring at its equilibrium length,
    this test calculates the accelerations and checks that both
    translational and angular accelerations are zero.
    """
    pos = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ])

    theta = np.zeros(2)

    geometry = {
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([1.0]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2)),
        "periodic_x": False,
        "periodic_y": False
    }

    acc, alpha = compute_accelerations(
        pos,
        theta,
        geometry,
        mass=1.0,
        inertia_z=1.0
    )

    assert np.allclose(acc, 0)
    assert np.allclose(alpha, 0)


def test_stretched_spring_acceleration():
    """
    Given two nanorods connected by a stretched spring, this test
    calculates the acceleration and checks that the resulting forces
    pull the two nanorods toward each other with equal magnitude.
    """
    pos = np.array([
        [0.0, 0.0],
        [1.2, 0.0]
    ])

    theta = np.zeros(2)

    geometry = {
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([1.0]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2)),
        "periodic_x": False,
        "periodic_y": False
    }

    acc, alpha = compute_accelerations(
        pos,
        theta,
        geometry,
        mass=1.0,
        inertia_z=1.0
    )

    expected = np.array([
        [0.2, 0.0],
        [-0.2, 0.0]
    ])

    assert np.allclose(acc, expected)
    assert np.allclose(alpha, 0)


def test_spring_torque():
    """
    Given a stretched spring connected away from the nanorod centers,
    this test calculates the angular accelerations and checks that the
    spring produces opposite torques on the two nanorods.
    """
    pos = np.array([
        [0.0, 0.0],
        [1.2, 0.0]
    ])

    theta = np.zeros(2)

    site_offsets = np.array([
        [[0.0, 0.5]],
        [[0.0, 0.5]]
    ])

    geometry = {
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([1.0]),
        "k_eff": np.array([1.0]),
        "site_offsets": site_offsets,
        "periodic_x": False,
        "periodic_y": False
    }

    _, alpha = compute_accelerations(
        pos,
        theta,
        geometry,
        mass=1.0,
        inertia_z=1.0
    )

    assert np.allclose(alpha, [-0.1, 0.1])


def test_run_simulation_equilibrium():
    """
    Given a system initially at equilibrium with zero velocity, this test
    runs the simulation and checks that the nanorods remain at their
    equilibrium positions.
    """
    pos0 = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ])

    geometry = {
        "pos0": pos0,
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([1.0]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2)),
        "periodic_x": False,
        "periodic_y": False
    }

    result = run_simulation(
        geometry,
        pos0,
        np.zeros_like(pos0),
        np.zeros(2),
        np.zeros(2),
        mass=1.0,
        inertia_z=1.0,
        dt=0.1,
        nt=3
    )

    assert np.allclose(result["pos_profiles"], pos0)


def test_run_simulation_output_shape():
    """
    Given a two-nanorod system and three integration steps, this test
    runs the simulation and checks that all output arrays contain the
    initial state followed by the three calculated states.
    """
    pos0 = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ])

    geometry = {
        "pos0": pos0,
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([1.0]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2)),
        "periodic_x": False,
        "periodic_y": False
    }

    result = run_simulation(
        geometry,
        pos0,
        np.zeros_like(pos0),
        np.zeros(2),
        np.zeros(2),
        mass=1.0,
        inertia_z=1.0,
        dt=0.1,
        nt=3
    )

    assert result["pos_profiles"].shape == (4, 2, 2)
    assert result["theta_profiles"].shape == (4, 2)
    assert result["u_profiles"].shape == (4, 2, 2)
    assert result["time_profiles"].shape == (4,)


@pytest.mark.parametrize(
    "mass, inertia_z, dt, nt",
    [
        (0.0, 1.0, 0.1, 1),
        (1.0, 0.0, 0.1, 1),
        (1.0, 1.0, 0.0, 1),
        (1.0, 1.0, 0.1, 0),
    ]
)
def test_run_simulation_invalid_parameters(mass, inertia_z, dt, nt):
    """
    Given a simulation with a non-positive physical or integration
    parameter, this test checks that run_simulation raises a ValueError.
    """
    pos0 = np.zeros((1, 2))

    geometry = {
        "pos0": pos0,
        "ci": np.array([], dtype=int),
        "cj": np.array([], dtype=int),
        "si": np.array([], dtype=int),
        "sj": np.array([], dtype=int),
        "springL": np.array([]),
        "k_eff": np.array([]),
        "site_offsets": np.zeros((1, 1, 2)),
        "periodic_x": False,
        "periodic_y": False
    }

    with pytest.raises(ValueError):
        run_simulation(
            geometry,
            pos0,
            np.zeros_like(pos0),
            np.zeros(1),
            np.zeros(1),
            mass,
            inertia_z,
            dt,
            nt
        )


def test_periodic_x_equilibrium():
    """
    Given two nanorods connected across the periodic x boundary at their
    equilibrium separation, this test checks that the minimum-image
    correction gives zero acceleration.
    """
    pos = np.array([
        [0.1, 0.0],
        [0.9, 0.0]
    ])

    theta = np.zeros(2)

    geometry = {
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([0.2]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2)),
        "boxLx": 1.0,
        "boxLy": 1.0,
        "periodic_x": True,
        "periodic_y": False
    }

    acc, alpha = compute_accelerations(
        pos,
        theta,
        geometry,
        mass=1.0,
        inertia_z=1.0
    )

    assert np.allclose(acc, 0)
    assert np.allclose(alpha, 0)


def test_periodic_y_equilibrium():
    """
    Given two nanorods connected across the periodic y boundary at their
    equilibrium separation, this test checks that the minimum-image
    correction gives zero acceleration.
    """
    pos = np.array([
        [0.0, 0.1],
        [0.0, 0.9]
    ])

    theta = np.zeros(2)

    geometry = {
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([0.2]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2)),
        "boxLx": 1.0,
        "boxLy": 1.0,
        "periodic_x": False,
        "periodic_y": True
    }

    acc, alpha = compute_accelerations(
        pos,
        theta,
        geometry,
        mass=1.0,
        inertia_z=1.0
    )

    assert np.allclose(acc, 0)
    assert np.allclose(alpha, 0)


def test_periodic_x_requires_box_length():
    """
    Given periodic boundary conditions along x without a box length,
    this test checks that compute_accelerations raises a ValueError.
    """
    pos = np.zeros((2, 2))
    theta = np.zeros(2)

    geometry = {
        "ci": np.array([], dtype=int),
        "cj": np.array([], dtype=int),
        "si": np.array([], dtype=int),
        "sj": np.array([], dtype=int),
        "springL": np.array([]),
        "k_eff": np.array([]),
        "site_offsets": np.zeros((2, 1, 2)),
        "periodic_x": True,
        "periodic_y": False
    }

    with pytest.raises(ValueError):
        compute_accelerations(
            pos,
            theta,
            geometry,
            mass=1.0,
            inertia_z=1.0
        )


def test_periodic_y_requires_box_length():
    """
    Given periodic boundary conditions along y without a box length,
    this test checks that compute_accelerations raises a ValueError.
    """
    pos = np.zeros((2, 2))
    theta = np.zeros(2)

    geometry = {
        "ci": np.array([], dtype=int),
        "cj": np.array([], dtype=int),
        "si": np.array([], dtype=int),
        "sj": np.array([], dtype=int),
        "springL": np.array([]),
        "k_eff": np.array([]),
        "site_offsets": np.zeros((2, 1, 2)),
        "periodic_x": False,
        "periodic_y": True
    }

    with pytest.raises(ValueError):
        compute_accelerations(
            pos,
            theta,
            geometry,
            mass=1.0,
            inertia_z=1.0
        )