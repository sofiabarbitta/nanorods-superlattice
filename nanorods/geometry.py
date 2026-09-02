"""Geometry functions for the nanorod superlattice simulation."""

import numpy as np


def sub2ind(array_shape: tuple[int, int], row: int, col: int) -> int:
    """
    Convert 2D lattice indices to a linear index.

    Parameters
    ----------
    array_shape : tuple
        Number of rows and columns of the lattice.
    row, col : int
        Row and column indices.

    Returns
    -------
    int
        Corresponding linear index.
    """
    return row*array_shape[1]+col


def minimum_image(
    sep: np.ndarray,
    boxLx: float,
    boxLy: float,
    periodic_x: bool = False,
    periodic_y: bool = False
) -> np.ndarray:
    """
    Apply the minimum-image convention along periodic directions.

    Parameters
    ----------
    sep : ndarray
        Separation vectors, in meters.
    boxLx, boxLy : float
        Simulation box lengths along x and y, in meters.
    periodic_x, periodic_y : bool
        Periodic boundary conditions along x and y.

    Returns
    -------
    ndarray
        Corrected separation vectors.
    """
    sep = sep.copy()

    if periodic_x:
        sep[..., 0] -= np.round(sep[..., 0]/boxLx)*boxLx

    if periodic_y:
        sep[..., 1] -= np.round(sep[..., 1]/boxLy)*boxLy

    return sep


def get_sites(
    pos: np.ndarray,
    theta: np.ndarray,
    site_offsets: np.ndarray
) -> np.ndarray:
    """
    Return the positions of the interaction sites in the laboratory frame.

    Parameters
    ----------
    pos : ndarray, shape (N, 2)
        Nanorod center positions, in meters.
    theta : ndarray, shape (N,)
        Nanorod rotation angles, in radians.
    site_offsets : ndarray, shape (N, n_sites, 2)
        Interaction-site positions relative to the nanorod centers,
        in meters.

    Returns
    -------
    ndarray, shape (N, n_sites, 2)
        Interaction-site positions in the laboratory frame.
    """
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]

    xloc = site_offsets[..., 0]
    yloc = site_offsets[..., 1]

    sites = np.empty_like(site_offsets)

    sites[..., 0] = pos[:, 0, None]+c*xloc-s*yloc
    sites[..., 1] = pos[:, 1, None]+s*xloc+c*yloc

    return sites


def capsule_surface_offsets(
    L: float,
    d: float,
    sites_per_nm: float
) -> tuple[np.ndarray, int]:
    """
    Generate interaction sites along the 2D nanorod surface.

    The nanorod consists of a straight section of length L-d and two
    semicircular ends of radius d/2.

    Parameters
    ----------
    L : float
        Total nanorod length, in meters.
    d : float
        Nanorod diameter, in meters.
    sites_per_nm : float
        Linear density of interaction sites, in nm^-1.

    Returns
    -------
    offsets : ndarray, shape (n_sites, 2)
        Interaction-site positions relative to the nanorod center.
    n_sites : int
        Total number of interaction sites.

    Raises
    ------
    ValueError
        If L is smaller than d or sites_per_nm is not positive.
    """
    if L < d:
        raise ValueError("L must be greater than or equal to d")
    if sites_per_nm <= 0:
        raise ValueError("sites_per_nm must be positive")

    r = d/2
    l = L-d

    half_perimeter = l+np.pi*r
    ds_target = 1e-9/sites_per_nm

    # Multiple of 6 to preserve the hexagonal symmetry when L = d.
    n_sites = max(6, 6*int(np.round(2*half_perimeter/(6*ds_target))))
    n_half = n_sites//2

    # Midpoint sampling avoids sites exactly at the capsule corners.
    u = (np.arange(n_half)+0.5)*half_perimeter/n_half
    quarter_arc = np.pi*r/2

    x = np.empty(n_half)
    y = np.empty(n_half)

    left = u < quarter_arc
    line = (u >= quarter_arc) & (u < quarter_arc+l)
    right = ~(left | line)

    angle = np.pi-u[left]/r
    x[left] = -l/2+r*np.cos(angle)
    y[left] = r*np.sin(angle)

    x[line] = -l/2+u[line]-quarter_arc
    y[line] = r

    angle = np.pi/2-(u[right]-quarter_arc-l)/r
    x[right] = l/2+r*np.cos(angle)
    y[right] = r*np.sin(angle)

    upper = np.column_stack((x, y))
    lower = upper[::-1]*np.array([1, -1])

    return np.vstack((upper, lower)), n_sites


def capsule_radius_at_x(
    x: float | np.ndarray,
    L: float,
    d: float
) -> float | np.ndarray:
    """
    Return the local nanorod radius at coordinate x.

    Parameters
    ----------
    x : float or ndarray
        Position along the nanorod long axis, in meters.
    L : float
        Total nanorod length, in meters.
    d : float
        Nanorod diameter, in meters.

    Returns
    -------
    float or ndarray
        Local radius, in meters.
    """
    r = d/2
    xc = (L-d)/2
    q = np.maximum(np.abs(x)-xc, 0)

    return np.sqrt(np.maximum(r*r-q*q, 0))


def hidden_points_at_site(
    site: np.ndarray,
    L: float,
    d: float,
    n_sites: int
) -> np.ndarray:
    """
    Reconstruct the off-plane points associated with a 2D interaction site.

    Parameters
    ----------
    site : ndarray, shape (2,)
        Position of the interaction site relative to the nanorod center.
    L : float
        Total nanorod length, in meters.
    d : float
        Nanorod diameter, in meters.
    n_sites : int
        Number of sites along the 2D perimeter.

    Returns
    -------
    ndarray, shape (N, 3)
        Corresponding points on the 3D nanorod surface.
    """
    x, y = site

    r = d/2
    perimeter = 2*(L-d)+2*np.pi*r
    ds_arc = perimeter/n_sites

    radius = capsule_radius_at_x(x, L, d)

    if radius <= 0:
        return np.array([[x, 0.0, 0.0]])

    n_phi = max(1, int(np.round(np.pi*radius/ds_arc)))
    phi = (np.arange(n_phi)+0.5)*np.pi/n_phi-np.pi/2

    sign = 1 if y >= 0 else -1

    return np.column_stack((
        np.full(n_phi, x),
        sign*radius*np.cos(phi),
        radius*np.sin(phi)
    ))


def match_unique_pairs(
    dist: np.ndarray,
    smax: float
) -> np.ndarray:
    """
    Match interaction sites separated by less than smax.

    Each site can belong to only one connection. The closest pairs are
    selected first.

    Parameters
    ----------
    dist : ndarray
        Distance matrix between two sets of sites, in meters.
    smax : float
        Maximum interaction distance, in meters.

    Returns
    -------
    ndarray, shape (n_pairs, 2)
        Indices of the matched sites.
    """
    candidates = np.argwhere(dist <= smax)

    if len(candidates) == 0:
        return np.empty((0, 2), dtype=int)

    distances = dist[candidates[:, 0], candidates[:, 1]]
    candidates = candidates[np.argsort(distances)]

    used_i = set()
    used_j = set()
    pairs = []

    for a, b in candidates:
        if a in used_i or b in used_j:
            continue

        used_i.add(a)
        used_j.add(b)
        pairs.append((a, b))

    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def nanorod_neighbors(col: int) -> list[tuple[int, int]]:
    """
    Return the forward nearest neighbors in the staggered lattice.

    Parameters
    ----------
    col : int
        Column index of the nanorod.

    Returns
    -------
    list of tuple
        Relative row and column indices of the neighboring nanorods.
    """
    if col%2 == 0:
        return [(1, 0), (0, 1), (-1, 1)]

    return [(1, 0), (1, 1), (0, 1)]


def neighbor_pairs(
    Nx: int,
    Ny: int,
    periodic_x: bool = False,
    periodic_y: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the nearest-neighbor nanorod pairs.

    Parameters
    ----------
    Nx, Ny : int
        Number of nanorods along x and y.
    periodic_x, periodic_y : bool
        Periodic boundary conditions along x and y.

    Returns
    -------
    i, j : ndarray
        Linear indices of neighboring nanorod pairs.
    """
    rows, cols = np.indices((Ny, Nx))
    rows = rows.ravel()
    cols = cols.ravel()

    even = cols%2 == 0

    drow = np.where(
        even[:, None],
        np.array([[1, 0, -1]]),
        np.array([[1, 1, 0]])
    )

    dcol = np.array([0, 1, 1])

    rr = rows[:, None]+drow
    cc = cols[:, None]+dcol

    valid = np.ones(rr.shape, dtype=bool)

    if periodic_y:
        rr %= Ny
    else:
        valid &= (rr >= 0) & (rr < Ny)

    if periodic_x:
        cc %= Nx
    else:
        valid &= (cc >= 0) & (cc < Nx)

    i = np.broadcast_to(
        (rows*Nx+cols)[:, None],
        rr.shape
    )

    j = rr*Nx+cc

    return i[valid], j[valid]


def build_springs_by_distance(
    Nx: int,
    Ny: int,
    pos0: np.ndarray,
    site_offsets: np.ndarray,
    smax: float,
    boxLx: float,
    boxLy: float,
    periodic_x: bool = False,
    periodic_y: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the spring connections between neighboring nanorods.

    Parameters
    ----------
    Nx, Ny : int
        Number of nanorods along x and y.
    pos0 : ndarray, shape (N, 2)
        Equilibrium nanorod positions.
    site_offsets : ndarray
        Interaction-site positions relative to each nanorod center.
    smax : float
        Maximum interaction distance, in meters.
    boxLx, boxLy : float
        Simulation box dimensions, in meters.
    periodic_x, periodic_y : bool
        Periodic boundary conditions along x and y.

    Returns
    -------
    ci, cj : ndarray
        Indices of the connected nanorods.
    si, sj : ndarray
        Indices of the connected interaction sites.
    """
    sites0 = get_sites(pos0, np.zeros(len(pos0)), site_offsets)

    rod_i, rod_j = neighbor_pairs(
        Nx, Ny,
        periodic_x,
        periodic_y
    )

    ci = []
    cj = []
    si = []
    sj = []

    for i, j in zip(rod_i, rod_j):
        sep = sites0[i, :, None]-sites0[j, None]
        sep = minimum_image(
            sep,
            boxLx,
            boxLy,
            periodic_x,
            periodic_y
        )

        dist = np.linalg.norm(sep, axis=-1)
        pairs = match_unique_pairs(dist, smax)

        if len(pairs) == 0:
            continue

        a, b = pairs.T

        ci.extend([i]*len(a))
        cj.extend([j]*len(a))
        si.extend(a)
        sj.extend(b)

    return tuple(
        np.asarray(values, dtype=int)
        for values in (ci, cj, si, sj)
    )


def compute_effective_spring_constants(
    pos0: np.ndarray,
    site_offsets: np.ndarray,
    ci: np.ndarray,
    cj: np.ndarray,
    si: np.ndarray,
    sj: np.ndarray,
    L: float,
    d: float,
    smax: float,
    k_lig: float,
    boxLx: float,
    boxLy: float,
    periodic_x: bool = False,
    periodic_y: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the effective spring constant of each connection.

    Off-plane ligand connections are treated as parallel springs.

    Parameters
    ----------
    pos0 : ndarray
        Equilibrium nanorod positions.
    site_offsets : ndarray
        Interaction-site positions relative to each nanorod center.
    ci, cj : ndarray
        Indices of the connected nanorods.
    si, sj : ndarray
        Indices of the connected interaction sites.
    L : float
        Total nanorod length, in meters.
    d : float
        Nanorod diameter, in meters.
    smax : float
        Maximum ligand interaction distance, in meters.
    k_lig : float
        Spring constant of a single ligand, in N/m.
    boxLx, boxLy : float
        Simulation box dimensions, in meters.
    periodic_x, periodic_y : bool
        Periodic boundary conditions along x and y.

    Returns
    -------
    k_eff : ndarray
        Effective spring constants, in N/m.
    n_count : ndarray
        Number of ligands contributing to each connection.
    """
    n_count = np.zeros(len(ci), dtype=int)
    n_sites = site_offsets.shape[1]

    for q, (i, j, a, b) in enumerate(zip(ci, cj, si, sj)):
        pts1 = hidden_points_at_site(
            site_offsets[i, a],
            L, d, n_sites
        )

        pts2 = hidden_points_at_site(
            site_offsets[j, b],
            L, d, n_sites
        )

        pts1 += np.array([pos0[i, 0], pos0[i, 1], 0])
        pts2 += np.array([pos0[j, 0], pos0[j, 1], 0])

        sep = pts1[:, None]-pts2[None]
        sep = minimum_image(
            sep,
            boxLx,
            boxLy,
            periodic_x,
            periodic_y
        )

        dist = np.linalg.norm(sep, axis=-1)

        n_count[q] = len(match_unique_pairs(dist, smax))

    return k_lig*n_count, n_count


def build_nanorod_geometry(
    Nx: int,
    Ny: int,
    L: float,
    d: float,
    s: float,
    smax: float,
    ligand_surface_density: float,
    k_lig: float,
    periodic_x: bool = False,
    periodic_y: bool = False
) -> dict:
    """
    Build the nanorod lattice and its spring network.

    Parameters
    ----------
    Nx, Ny : int
        Number of nanorods along x and y.
    L : float
        Total nanorod length, in meters.
    d : float
        Nanorod diameter, in meters.
    s : float
        Nanorod-nanorod spacing, in meters.
    smax : float
        Maximum ligand interaction distance, in meters.
    ligand_surface_density : float
        Ligand surface density, in nm^-2.
    k_lig : float
        Spring constant of a single ligand, in N/m.
    periodic_x, periodic_y : bool
        Periodic boundary conditions along x and y.

    Returns
    -------
    dict
        Nanorod positions, spring connections and lattice parameters.

    Raises
    ------
    ValueError
        If the input geometry or interaction parameters are not valid.
    """
    if Nx <= 0 or Ny <= 0:
        raise ValueError("Nx and Ny must be positive")
    if L < d:
        raise ValueError("L must be greater than or equal to d")
    if s < 0:
        raise ValueError("s must be positive or zero")
    if smax <= 0:
        raise ValueError("smax must be positive")
    if ligand_surface_density <= 0:
        raise ValueError("ligand_surface_density must be positive")
    if k_lig <= 0:
        raise ValueError("k_lig must be positive")
    if periodic_x and Nx%2 != 0:
        raise ValueError("Nx must be even when periodic_x is True")

    l = L-d

    dx = l+np.sqrt(3)/2*(d+s)
    dy = d+s

    rows, cols = np.indices((Ny, Nx))

    pos0 = np.column_stack((
        cols.ravel()*dx,
        dy/2*(cols.ravel()%2+2*rows.ravel())
    ))

    boxLx = Nx*dx
    boxLy = Ny*dy

    sites_per_nm = np.sqrt(ligand_surface_density)

    offsets, n_sites = capsule_surface_offsets(
        L,
        d,
        sites_per_nm
    )

    site_offsets = np.repeat(
        offsets[None],
        len(pos0),
        axis=0
    )

    ci, cj, si, sj = build_springs_by_distance(
        Nx, Ny,
        pos0,
        site_offsets,
        smax,
        boxLx,
        boxLy,
        periodic_x,
        periodic_y
    )

    sites0 = get_sites(
        pos0,
        np.zeros(len(pos0)),
        site_offsets
    )

    sep0 = sites0[ci, si]-sites0[cj, sj]
    sep0 = minimum_image(
        sep0,
        boxLx,
        boxLy,
        periodic_x,
        periodic_y
    )

    springL = np.linalg.norm(sep0, axis=1)

    k_eff, n_count = compute_effective_spring_constants(
        pos0,
        site_offsets,
        ci, cj, si, sj,
        L, d,
        smax,
        k_lig,
        boxLx,
        boxLy,
        periodic_x,
        periodic_y
    )

    return {
        "pos0": pos0,
        "ci": ci,
        "cj": cj,
        "si": si,
        "sj": sj,
        "springL": springL,
        "k_eff": k_eff,
        "n_count": n_count,
        "site_offsets": site_offsets,
        "boxLx": boxLx,
        "boxLy": boxLy,
        "dx": dx,
        "dy": dy,
        "n_sites": n_sites,
        "periodic_x": periodic_x,
        "periodic_y": periodic_y
    }
    