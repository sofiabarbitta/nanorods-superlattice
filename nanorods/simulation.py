"""Dynamics functions for the nanorod superlattice simulation."""

import numpy as np

from .geometry import get_sites, sub2ind, minimum_image


def nanorod_moment_of_inertia(m_cylinder, m_sphere, r, l):
    """
    Calculate the nanorod moment of inertia around the z axis.

    --- Parameters ---
    m_cylinder: float
        Mass of the cylindrical part (in kg).
    m_sphere: float
        Total mass of the two hemispherical caps (in kg).
    r: float
        Nanorod radius (in meters).
    l: float
        Length of the cylindrical part (in meters).

    --- Returns ---
    Iz: float
        Moment of inertia around the z axis (in kg m^2).
    """
    return (1/12)*m_cylinder*(3*r**2+l**2)+m_sphere*((83/320)*r**2+(l/2+3*r/8)**2)


def line_displacement(pos0, Nx, Ny, amplitude, direction, mode):
    """
    Apply a line displacement to the center row or column of the lattice.

    --- Parameters ---
    pos0: ndarray, shape (N, 2)
        Equilibrium nanorod positions.
    Nx, Ny: int
        Number of nanorods along x and y.
    amplitude: float
        Initial displacement amplitude (in meters).
    direction: str
        Propagation direction: "parallel" or "perpendicular".
    mode: str
        Wave mode: "longitudinal" or "transverse".

    --- Returns ---
    pos, vel, theta, omega: ndarray
        Initial translational and rotational variables.
    """
    if direction not in ("parallel", "perpendicular"):
        raise ValueError("direction must be 'parallel' or 'perpendicular'")
    if mode not in ("longitudinal", "transverse"):
        raise ValueError("mode must be 'longitudinal' or 'transverse'")

    pos = pos0.copy()
    vel = np.zeros_like(pos)
    theta = np.zeros(len(pos))
    omega = np.zeros(len(pos))

    if direction == "parallel":
        col = Nx//2

        for row in range(Ny):
            idx = sub2ind((Ny, Nx), row, col)

            if mode == "longitudinal":
                pos[idx, 0] += amplitude
            else:
                pos[idx, 1] += amplitude

    else:
        row = Ny//2

        for col in range(Nx):
            idx = sub2ind((Ny, Nx), row, col)

            if mode == "longitudinal":
                pos[idx, 1] += amplitude
            else:
                pos[idx, 0] += amplitude

    return pos, vel, theta, omega


def compute_accelerations(
    pos, theta, ci, cj, si, sj, springL, k_eff, mass, Iz, site_offsets,
    boxLx=None, boxLy=None, periodic_x=False, periodic_y=False
):
    """
    Calculate translational and angular accelerations.

    Spring forces follow Hooke's law and torques are calculated with
    respect to the center of each nanorod.

    --- Parameters ---
    pos: ndarray, shape (N, 2)
        Nanorod center positions.
    theta: ndarray, shape (N,)
        Nanorod rotation angles (in radians).
    ci, cj: ndarray
        Indices of the connected nanorods.
    si, sj: ndarray
        Indices of the connected interaction sites.
    springL: ndarray
        Spring rest lengths (in meters).
    k_eff: ndarray
        Effective spring constants (in N/m).
    mass: float
        Nanorod mass (in kg).
    Iz: float
        Nanorod moment of inertia (in kg m^2).
    site_offsets: ndarray
        Interaction site positions relative to each nanorod center.
    boxLx, boxLy: float or None
        Simulation box dimensions (in meters).
    periodic_x, periodic_y: bool
        Periodic boundary conditions along x and y.

    --- Returns ---
    acc: ndarray, shape (N, 2)
        Translational accelerations (in m/s^2).
    alpha: ndarray, shape (N,)
        Angular accelerations (in rad/s^2).
    """
    acc = np.zeros_like(pos)
    torque = np.zeros(len(pos))

    sites = get_sites(pos, theta, site_offsets)

    p_i = sites[ci, si]
    p_j = sites[cj, sj]

    sep_vec = p_i-p_j
    
    if periodic_x and boxLx is None:
        raise ValueError("boxLx is required when periodic_x is True")
    if periodic_y and boxLy is None:
        raise ValueError("boxLy is required when periodic_y is True")

    if periodic_x or periodic_y:
        sep_vec = minimum_image(
            sep_vec, boxLx, boxLy,
            periodic_x, periodic_y
        )
    
    sep = np.linalg.norm(sep_vec, axis=1)
    sep_safe = np.where(sep > 1e-12, sep, 1e-12)

    dL = sep-springL

    Fx = -k_eff*dL*sep_vec[:, 0]/sep_safe
    Fy = -k_eff*dL*sep_vec[:, 1]/sep_safe

    np.add.at(acc[:, 0], ci, Fx/mass)
    np.add.at(acc[:, 1], ci, Fy/mass)
    np.add.at(acc[:, 0], cj, -Fx/mass)
    np.add.at(acc[:, 1], cj, -Fy/mass)

    r_i = p_i-pos[ci]
    r_j = p_j-pos[cj]

    tau_i = r_i[:, 0]*Fy-r_i[:, 1]*Fx
    tau_j = r_j[:, 0]*(-Fy)-r_j[:, 1]*(-Fx)

    np.add.at(torque, ci, tau_i)
    np.add.at(torque, cj, tau_j)

    alpha = torque/Iz

    return acc, alpha


def run_simulation(geometry, pos, vel, theta, omega, mass, Iz, dt, Nt):
    """
    Run the nanorod dynamics using the kick-drift-kick algorithm.

    --- Parameters ---
    geometry: dict
        Nanorod geometry and spring network.
    pos, vel: ndarray
        Initial positions and velocities.
    theta, omega: ndarray
        Initial orientations and angular velocities.
    mass: float
        Nanorod mass (in kg).
    Iz: float
        Nanorod moment of inertia (in kg m^2).
    dt: float
        Integration time step (in seconds).
    Nt: int
        Number of integration steps.

    --- Returns ---
    results: dict
        Position, rotation, displacement and time profiles.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if Nt <= 0:
        raise ValueError("Nt must be positive")
    if mass <= 0:
        raise ValueError("mass must be positive")
    if Iz <= 0:
        raise ValueError("Iz must be positive")

    pos = pos.copy()
    vel = vel.copy()
    theta = theta.copy()
    omega = omega.copy()

    pos0 = geometry["pos0"]
    ci, cj = geometry["ci"], geometry["cj"]
    si, sj = geometry["si"], geometry["sj"]
    springL = geometry["springL"]
    k_eff = geometry["k_eff"]
    site_offsets = geometry["site_offsets"]
    boxLx = geometry.get("boxLx")
    boxLy = geometry.get("boxLy")
    periodic_x = geometry.get("periodic_x", False)
    periodic_y = geometry.get("periodic_y", False)

    pos_profiles = [pos.copy()]
    theta_profiles = [theta.copy()]
    u_profiles = [(pos-pos0).copy()]
    time_profiles = [0.0]

    acc, alpha = compute_accelerations(
        pos, theta, ci, cj, si, sj,
        springL, k_eff, mass, Iz, site_offsets,
        boxLx, boxLy, periodic_x, periodic_y
    )

    t = 0.0

    for i in range(Nt):

        # First half kick.
        vel += acc*dt/2
        omega += alpha*dt/2

        # Drift.
        pos += vel*dt
        theta += omega*dt

        acc, alpha = compute_accelerations(
            pos, theta, ci, cj, si, sj,
            springL, k_eff, mass, Iz, site_offsets,
            boxLx, boxLy, periodic_x, periodic_y
        )

        # Second half kick.
        vel += acc*dt/2
        omega += alpha*dt/2

        t += dt

        pos_profiles.append(pos.copy())
        theta_profiles.append(theta.copy())
        u_profiles.append((pos-pos0).copy())
        time_profiles.append(t)

    return {
        "pos_profiles": np.asarray(pos_profiles),
        "theta_profiles": np.asarray(theta_profiles),
        "u_profiles": np.asarray(u_profiles),
        "time_profiles": np.asarray(time_profiles)
    }