"""Tests for the nanorod simulation functions.
This module tests the functions inside the nanorods/simulation.py module.
"""

import numpy as np
import pytest

from nanorods.simulation import (
    nanorod_moment_of_inertia,
    line_displacement,
    compute_accelerations,
    run_simulation
)


def test_moment_of_inertia():
    """Check the nanorod moment of inertia for a simple case."""
    Iz = nanorod_moment_of_inertia(1.0, 0.0, 2.0, 4.0)
    expected = (1/12)*(3*2**2+4**2)

    assert np.isclose(Iz, expected)


@pytest.mark.parametrize(
    "direction, mode, displaced, axis",
    [
        ("parallel", "longitudinal", [1, 4, 7], 0),
        ("parallel", "transverse", [1, 4, 7], 1),
        ("perpendicular", "longitudinal", [3, 4, 5], 1),
        ("perpendicular", "transverse", [3, 4, 5], 0)
    ]
)
def test_line_displacement(direction, mode, displaced, axis):
    """Check the four possible line displacement configurations."""
    Nx, Ny = 3, 3
    pos0 = np.zeros((Nx*Ny, 2))
    amplitude = 1e-9

    pos, vel, theta, omega = line_displacement(
        pos0, Nx, Ny, amplitude, direction, mode
    )

    expected = pos0.copy()
    expected[displaced, axis] = amplitude

    assert np.allclose(pos, expected)


@pytest.mark.parametrize(
    "direction, mode",
    [
        ("wrong", "longitudinal"),
        ("parallel", "wrong")
    ]
)
def test_line_displacement_invalid_input(direction, mode):
    """Check that invalid excitation parameters raise an error."""
    pos0 = np.zeros((9, 2))

    with pytest.raises(ValueError):
        line_displacement(pos0, 3, 3, 1e-9, direction, mode)


def test_equilibrium_acceleration():
    """Check that a spring at equilibrium produces no acceleration."""
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    theta = np.zeros(2)

    ci = np.array([0])
    cj = np.array([1])
    si = np.array([0])
    sj = np.array([0])

    springL = np.array([1.0])
    k_eff = np.array([1.0])
    site_offsets = np.zeros((2, 1, 2))

    acc, alpha = compute_accelerations(
        pos, theta, ci, cj, si, sj,
        springL, k_eff, 1.0, 1.0, site_offsets
    )

    assert np.allclose(acc, 0)
    assert np.allclose(alpha, 0)


def test_stretched_spring_acceleration():
    """Check the acceleration produced by a stretched spring."""
    pos = np.array([[0.0, 0.0], [1.2, 0.0]])
    theta = np.zeros(2)

    ci = np.array([0])
    cj = np.array([1])
    si = np.array([0])
    sj = np.array([0])

    springL = np.array([1.0])
    k_eff = np.array([2.0])
    site_offsets = np.zeros((2, 1, 2))

    acc, alpha = compute_accelerations(
        pos, theta, ci, cj, si, sj,
        springL, k_eff, 1.0, 1.0, site_offsets
    )

    expected = np.array([
        [0.4, 0.0],
        [-0.4, 0.0]
    ])

    assert np.allclose(acc, expected)


def test_spring_torque():
    """Check that an off-center spring produces angular acceleration."""
    pos = np.array([[0.0, 0.0], [1.2, 0.0]])
    theta = np.zeros(2)

    ci = np.array([0])
    cj = np.array([1])
    si = np.array([0])
    sj = np.array([0])

    springL = np.array([1.0])
    k_eff = np.array([2.0])

    site_offsets = np.array([
        [[0.0, 0.5]],
        [[0.0, 0.5]]
    ])

    acc, alpha = compute_accelerations(
        pos, theta, ci, cj, si, sj,
        springL, k_eff, 1.0, 1.0, site_offsets
    )

    assert alpha[0] < 0
    assert alpha[1] > 0
    assert np.isclose(alpha[0], -alpha[1])


def test_run_simulation_equilibrium():
    """Check that an equilibrium system remains at equilibrium."""
    pos0 = np.array([[0.0, 0.0], [1.0, 0.0]])

    geometry = {
        "pos0": pos0,
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([1.0]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2))
    }

    pos = pos0.copy()
    vel = np.zeros_like(pos)
    theta = np.zeros(2)
    omega = np.zeros(2)

    results = run_simulation(
        geometry, pos, vel, theta, omega,
        mass=1.0, Iz=1.0, dt=0.1, Nt=5
    )

    assert np.allclose(results["u_profiles"], 0)


def test_run_simulation_output_shape():
    """Check the number and shape of the stored simulation states."""
    pos0 = np.array([[0.0, 0.0], [1.0, 0.0]])

    geometry = {
        "pos0": pos0,
        "ci": np.array([0]),
        "cj": np.array([1]),
        "si": np.array([0]),
        "sj": np.array([0]),
        "springL": np.array([1.0]),
        "k_eff": np.array([1.0]),
        "site_offsets": np.zeros((2, 1, 2))
    }

    results = run_simulation(
        geometry,
        pos0.copy(),
        np.zeros_like(pos0),
        np.zeros(2),
        np.zeros(2),
        mass=1.0,
        Iz=1.0,
        dt=0.1,
        Nt=5
    )

    assert results["pos_profiles"].shape == (6, 2, 2)
    assert results["theta_profiles"].shape == (6, 2)
    assert results["u_profiles"].shape == (6, 2, 2)
    assert results["time_profiles"].shape == (6,)


@pytest.mark.parametrize(
    "mass, Iz, dt, Nt",
    [
        (0.0, 1.0, 0.1, 5),
        (1.0, 0.0, 0.1, 5),
        (1.0, 1.0, 0.0, 5),
        (1.0, 1.0, 0.1, 0)
    ]
)
def test_run_simulation_invalid_input(mass, Iz, dt, Nt):
    """Check that invalid simulation parameters raise an error."""
    geometry = {
        "pos0": np.zeros((2, 2)),
        "ci": np.array([], dtype=int),
        "cj": np.array([], dtype=int),
        "si": np.array([], dtype=int),
        "sj": np.array([], dtype=int),
        "springL": np.array([]),
        "k_eff": np.array([]),
        "site_offsets": np.zeros((2, 1, 2))
    }

    with pytest.raises(ValueError):
        run_simulation(
            geometry,
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros(2),
            np.zeros(2),
            mass, Iz, dt, Nt
        )
        
def test_acceleration_periodic_x_equilibrium():
    """Check equilibrium for a spring crossing the periodic x boundary."""
    pos = np.array([[0.0, 0.0], [9.0, 0.0]])
    theta = np.zeros(2)

    acc, alpha = compute_accelerations(
        pos,
        theta,
        ci=np.array([0]),
        cj=np.array([1]),
        si=np.array([0]),
        sj=np.array([0]),
        springL=np.array([1.0]),
        k_eff=np.array([1.0]),
        mass=1.0,
        Iz=1.0,
        site_offsets=np.zeros((2, 1, 2)),
        boxLx=10.0,
        boxLy=10.0,
        periodic_x=True
    )

    assert np.allclose(acc, 0)
    assert np.allclose(alpha, 0)


def test_acceleration_periodic_y_equilibrium():
    """Check equilibrium for a spring crossing the periodic y boundary."""
    pos = np.array([[0.0, 0.0], [0.0, 9.0]])
    theta = np.zeros(2)

    acc, alpha = compute_accelerations(
        pos,
        theta,
        ci=np.array([0]),
        cj=np.array([1]),
        si=np.array([0]),
        sj=np.array([0]),
        springL=np.array([1.0]),
        k_eff=np.array([1.0]),
        mass=1.0,
        Iz=1.0,
        site_offsets=np.zeros((2, 1, 2)),
        boxLx=10.0,
        boxLy=10.0,
        periodic_y=True
    )

    assert np.allclose(acc, 0)
    assert np.allclose(alpha, 0)


def test_periodic_boundary_requires_box_length():
    """Check that periodic directions require the corresponding box length."""
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    theta = np.zeros(2)

    with pytest.raises(ValueError, match="boxLx is required"):
        compute_accelerations(
            pos,
            theta,
            ci=np.array([0]),
            cj=np.array([1]),
            si=np.array([0]),
            sj=np.array([0]),
            springL=np.array([1.0]),
            k_eff=np.array([1.0]),
            mass=1.0,
            Iz=1.0,
            site_offsets=np.zeros((2, 1, 2)),
            periodic_x=True
        )