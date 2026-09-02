"""Dynamics functions for the nanorod superlattice simulation."""

import numpy as np

from .geometry import get_sites, minimum_image


def nanorod_moment_of_inertia(
    m_cylinder: float,
    m_sphere: float,
    r: float,
    l: float
) -> float:
    """
    Calculate the nanorod moment of inertia around the z axis.

    Parameters
    ----------
    m_cylinder : float
        Mass of the cylindrical part, in kg.
    m_sphere : float
        Total mass of the two hemispherical caps, in kg.
    r : float
        Nanorod radius, in meters.
    l : float
        Length of the cylindrical part, in meters.

    Returns
    -------
    float
        Moment of inertia around the z axis, in kg m^2.
    """
    return (1/12)*m_cylinder*(3*r**2+l**2)+m_sphere*((83/320)*r**2+(l/2+3*r/8)**2)


def line_displacement(
    pos0: np.ndarray,
    nx: int,
    ny: int,
    amplitude: float,
    direction: str,
    mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a line displacement to the central row or column of the lattice.

    Parameters
    ----------
    pos0 : ndarray, shape (N, 2)
        Equilibrium nanorod positions, in meters.
    nx, ny : int
        Number of nanorods along x and y.
    amplitude : float
        Initial displacement amplitude, in meters.
    direction : str
        Propagation direction: "parallel" or "perpendicular".
    mode : str
        Wave mode: "longitudinal" or "transverse".

    Returns
    -------
    pos : ndarray, shape (N, 2)
        Initial nanorod positions.
    vel : ndarray, shape (N, 2)
        Initial translational velocities.
    theta : ndarray, shape (N,)
        Initial rotation angles.
    omega : ndarray, shape (N,)
        Initial angular velocities.

    Raises
    ------
    ValueError
        If direction or mode is not valid.
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
        col = nx//2
        indices = np.arange(ny)*nx+col
        axis = 0 if mode == "longitudinal" else 1

    else:
        row = ny//2
        indices = row*nx+np.arange(nx)
        axis = 1 if mode == "longitudinal" else 0

    pos[indices, axis] += amplitude

    return pos, vel, theta, omega


def compute_accelerations(
    pos: np.ndarray,
    theta: np.ndarray,
    geometry: dict,
    mass: float,
    inertia_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the translational and angular accelerations.

    Spring forces follow Hooke's law. The corresponding torques are
    calculated with respect to the center of each nanorod.

    Parameters
    ----------
    pos : ndarray, shape (N, 2)
        Nanorod center positions, in meters.
    theta : ndarray, shape (N,)
        Nanorod rotation angles, in radians.
    geometry : dict
        Nanorod geometry and spring network.
    mass : float
        Nanorod mass, in kg.
    inertia_z : float
        Nanorod moment of inertia around the z axis, in kg m^2.

    Returns
    -------
    acc : ndarray, shape (N, 2)
        Translational accelerations, in m/s^2.
    alpha : ndarray, shape (N,)
        Angular accelerations, in rad/s^2.

    Raises
    ------
    ValueError
        If a periodic boundary is used without the corresponding
        simulation box length.
    """
    ci = geometry["ci"]
    cj = geometry["cj"]
    si = geometry["si"]
    sj = geometry["sj"]

    spring_length = geometry["springL"]
    k_eff = geometry["k_eff"]
    site_offsets = geometry["site_offsets"]

    box_lx = geometry.get("boxLx")
    box_ly = geometry.get("boxLy")

    periodic_x = geometry.get("periodic_x", False)
    periodic_y = geometry.get("periodic_y", False)

    if periodic_x and box_lx is None:
        raise ValueError("boxLx is required when periodic_x is True")
    if periodic_y and box_ly is None:
        raise ValueError("boxLy is required when periodic_y is True")

    sites = get_sites(pos, theta, site_offsets)

    p_i = sites[ci, si]
    p_j = sites[cj, sj]

    sep_vec = p_i-p_j

    if periodic_x or periodic_y:
        sep_vec = minimum_image(
            sep_vec,
            box_lx,
            box_ly,
            periodic_x,
            periodic_y
        )

    sep = np.linalg.norm(sep_vec, axis=1)
    sep_safe = np.where(sep > 1e-12, sep, 1e-12)

    delta_length = sep-spring_length

    force = -k_eff[:, None]*delta_length[:, None]*sep_vec/sep_safe[:, None]

    acc = np.zeros_like(pos)

    np.add.at(acc, ci, force/mass)
    np.add.at(acc, cj, -force/mass)

    r_i = p_i-pos[ci]
    r_j = p_j-pos[cj]

    tau_i = r_i[:, 0]*force[:, 1]-r_i[:, 1]*force[:, 0]
    tau_j = r_j[:, 0]*(-force[:, 1])-r_j[:, 1]*(-force[:, 0])

    torque = np.zeros(len(pos))

    np.add.at(torque, ci, tau_i)
    np.add.at(torque, cj, tau_j)

    alpha = torque/inertia_z

    return acc, alpha


def run_simulation(
    geometry: dict,
    pos: np.ndarray,
    vel: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    mass: float,
    inertia_z: float,
    dt: float,
    nt: int
) -> dict:
    """
    Run the nanorod dynamics using the kick-drift-kick algorithm.

    Parameters
    ----------
    geometry : dict
        Nanorod geometry and spring network.
    pos, vel : ndarray
        Initial positions and translational velocities.
    theta, omega : ndarray
        Initial rotation angles and angular velocities.
    mass : float
        Nanorod mass, in kg.
    inertia_z : float
        Nanorod moment of inertia around the z axis, in kg m^2.
    dt : float
        Integration time step, in seconds.
    nt : int
        Number of integration steps.

    Returns
    -------
    dict
        Position, rotation, displacement and time profiles.

    Raises
    ------
    ValueError
        If dt, nt, mass or inertia_z is not positive.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if nt <= 0:
        raise ValueError("nt must be positive")
    if mass <= 0:
        raise ValueError("mass must be positive")
    if inertia_z <= 0:
        raise ValueError("inertia_z must be positive")

    pos = pos.copy()
    vel = vel.copy()
    theta = theta.copy()
    omega = omega.copy()

    pos0 = geometry["pos0"]

    pos_profiles = [pos.copy()]
    theta_profiles = [theta.copy()]
    u_profiles = [(pos-pos0).copy()]
    time_profiles = [0.0]

    acc, alpha = compute_accelerations(
        pos,
        theta,
        geometry,
        mass,
        inertia_z
    )

    t = 0.0

    for _ in range(nt):

        # First half kick.
        vel += acc*dt/2
        omega += alpha*dt/2

        # Drift.
        pos += vel*dt
        theta += omega*dt

        acc, alpha = compute_accelerations(
            pos,
            theta,
            geometry,
            mass,
            inertia_z
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