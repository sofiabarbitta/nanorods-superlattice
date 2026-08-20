"""Geometry functions for the nanorod superlattice simulation."""

import numpy as np


def sub2ind(array_shape, i, j):
    """Convert 2D lattice indices (i, j) to a linear index."""
    return i*array_shape[1]+j

def minimum_image(sep, boxLx, boxLy, periodic_x=False, periodic_y=False):
    """Apply the minimum-image convention along periodic directions."""
    sep = sep.copy()

    if periodic_x:
        sep[..., 0] -= np.round(sep[..., 0]/boxLx)*boxLx

    if periodic_y:
        sep[..., 1] -= np.round(sep[..., 1]/boxLy)*boxLy

    return sep

def get_sites(pos, theta, site_offsets):
    """
    Get the position of the interaction sites for each nanorod.

    --- Parameters ---
    pos: ndarray, shape (N, 2)
        Nanorod center positions.
    theta: ndarray, shape (N,)
        Nanorod rotation angles (in radians).
    site_offsets: ndarray, shape (N, n_sites, 2)
        Interaction site positions relative to the nanorod center.

    --- Returns ---
    sites: ndarray, shape (N, n_sites, 2)
        Interaction site positions in the laboratory frame.
    """
    N = len(pos)
    n_sites = site_offsets.shape[1]
    sites = np.zeros((N, n_sites, 2))

    c = np.cos(theta)
    s = np.sin(theta)

    for i in range(N):
        xloc = site_offsets[i, :, 0]
        yloc = site_offsets[i, :, 1]

        sites[i, :, 0] = pos[i, 0]+c[i]*xloc-s[i]*yloc
        sites[i, :, 1] = pos[i, 1]+s[i]*xloc+c[i]*yloc

    return sites


def capsule_surface_offsets(L, d, sites_per_nm):
    """
    Generate interaction sites along the 2D nanorod surface.

    The nanorod profile has a straight section of length L-d and two
    semicircular ends of radius d/2.

    --- Parameters ---
    L: float
        Total nanorod length (in meters).
    d: float
        Nanorod diameter (in meters).
    sites_per_nm: float
        Linear density of interaction sites (in nm^-1).

    --- Returns ---
    offsets: ndarray, shape (n_sites, 2)
        Interaction site positions relative to the nanorod center.
    n_sites: int
        Total number of interaction sites.
    """
    if L < d:
        raise ValueError("L must be greater than or equal to d")
    if sites_per_nm <= 0:
        raise ValueError("sites_per_nm must be positive")

    r, l = d/2, L-d
    half_perimeter = l+np.pi*r
    ds_target = 1e-9/sites_per_nm

    # Multiple of 6 to preserve the hexagonal symmetry when L = d.
    n_sites = max(6, 6*int(np.round(2*half_perimeter/(6*ds_target))))
    n_half = n_sites//2

    # Midpoint sampling avoids placing sites exactly at the capsule corners.
    u = (np.arange(n_half)+0.5)*half_perimeter/n_half
    quarter_arc = np.pi*r/2

    x, y = np.empty(n_half), np.empty(n_half)

    left = u < quarter_arc
    line = (u >= quarter_arc) & (u < quarter_arc+l)
    right = ~(left | line)

    theta = np.pi-u[left]/r
    x[left] = -l/2+r*np.cos(theta)
    y[left] = r*np.sin(theta)

    x[line] = -l/2+u[line]-quarter_arc
    y[line] = r

    theta = np.pi/2-(u[right]-quarter_arc-l)/r
    x[right] = l/2+r*np.cos(theta)
    y[right] = r*np.sin(theta)

    upper = np.column_stack((x, y))

    # Exact reflection of the upper half.
    lower = upper[::-1]*np.array([1, -1])

    return np.vstack((upper, lower)), n_sites


def capsule_radius_at_x(x, L, d):
    """
    Return the local radius of the nanorod cross section at coordinate x.

    --- Parameters ---
    x: float or ndarray
        Position along the nanorod long axis (in meters).
    L: float
        Total nanorod length (in meters).
    d: float
        Nanorod diameter (in meters).

    --- Returns ---
    float or ndarray
        Local radius (in meters).
    """
    r, xc = d/2, (L-d)/2
    q = np.maximum(np.abs(x)-xc, 0)

    return np.sqrt(np.maximum(r*r-q*q, 0))


def hidden_points_at_site(site, L, d, n_sites):
    """
    Reconstruct the off-plane points associated with a 2D interaction site.

    These points are used to estimate the number of ligand connections
    contributing to the effective spring constant.

    --- Parameters ---
    site: array_like, shape (2,)
        Position of the 2D interaction site.
    L: float
        Total nanorod length (in meters).
    d: float
        Nanorod diameter (in meters).
    n_sites: int
        Number of sites used along the 2D perimeter.

    --- Returns ---
    ndarray, shape (N, 3)
        Corresponding points on the 3D nanorod surface.
    """
    x, y = site
    r = d/2
    perimeter = 2*(L-d)+2*np.pi*r
    ds_arc = perimeter/n_sites
    R = capsule_radius_at_x(x, L, d)

    if R <= 0:
        return np.array([[x, 0.0, 0.0]])

    n_phi = max(1, int(np.round(np.pi*R/ds_arc)))
    phi = (np.arange(n_phi)+0.5)*np.pi/n_phi-np.pi/2
    sign = 1 if y >= 0 else -1

    return np.column_stack((np.full(n_phi, x), sign*R*np.cos(phi), R*np.sin(phi)))


def match_unique_pairs(dist, smax):
    """
    Match sites separated by less than smax.

    Each site can belong to only one connection and the closest pairs
    are selected first.

    --- Parameters ---
    dist: ndarray
        Distance matrix between two sets of sites (in meters).
    smax: float
        Maximum interaction distance (in meters).

    --- Returns ---
    pairs: ndarray, shape (n_pairs, 2)
        Indices of the matched sites.
    """
    candidates = sorted(np.argwhere(dist <= smax), key=lambda ab: dist[tuple(ab)])
    used_i, used_j, pairs = set(), set(), []

    for a, b in candidates:
        a, b = int(a), int(b)

        if a in used_i or b in used_j:
            continue

        used_i.add(a)
        used_j.add(b)
        pairs.append((a, b))

    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def nanorod_neighbors(row, col):
    """
    Return the forward nearest neighbors in the staggered lattice.

    --- Returns ---
    list of tuple
        Relative (row, column) indices of the neighboring nanorods.
    """
    if col%2 == 0:
        return [(1, 0), (0, 1), (-1, 1)]

    return [(1, 0), (1, 1), (0, 1)]


def build_springs_by_distance(
    Nx, Ny, pos0, site_offsets, smax, boxLx, boxLy,
    periodic_x=False, periodic_y=False
):
    """
    Build the spring connections between neighboring nanorods.

    --- Parameters ---
    Nx, Ny: int
        Number of nanorods along x and y.
    pos0: ndarray, shape (N, 2)
        Equilibrium nanorod positions.
    site_offsets: ndarray, shape (N, n_sites, 2)
        Interaction site positions relative to each nanorod center.
    smax: float
        Maximum interaction distance (in meters).
    boxLx, boxLy: float
        Simulation box dimensions (in meters).
    periodic_x, periodic_y: bool
        Periodic boundary conditions along x and y.

    --- Returns ---
    ci, cj: ndarray
        Indices of the connected nanorods.
    si, sj: ndarray
        Indices of the connected interaction sites.
    """
    sites0 = get_sites(pos0, np.zeros(len(pos0)), site_offsets)
    ci, cj, si, sj = [], [], [], []

    for row in range(Ny):
        for col in range(Nx):
            i = sub2ind((Ny, Nx), row, col)

            for drow, dcol in nanorod_neighbors(row, col):
                rr, cc = row+drow, col+dcol

                if periodic_y:
                    rr %= Ny
                elif rr < 0 or rr >= Ny:
                    continue

                if periodic_x:
                    cc %= Nx
                elif cc < 0 or cc >= Nx:
                    continue

                j = sub2ind((Ny, Nx), rr, cc)

                sep = sites0[i, :, None]-sites0[j, None]
                sep = minimum_image(
                    sep, boxLx, boxLy,
                    periodic_x, periodic_y
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

    return tuple(np.asarray(values, dtype=int) for values in (ci, cj, si, sj))


def compute_effective_spring_constants(
    pos0, site_offsets, ci, cj, si, sj, L, d, smax, k_lig,
    boxLx, boxLy, periodic_x=False, periodic_y=False
):
    """
    Compute the effective spring constant of each connection.

    Off-plane ligand connections are treated as parallel springs.

    --- Parameters ---
    pos0: ndarray, shape (N, 2)
        Equilibrium nanorod positions.
    site_offsets: ndarray, shape (N, n_sites, 2)
        Interaction site positions relative to each nanorod center.
    ci, cj: ndarray
        Indices of the connected nanorods.
    si, sj: ndarray
        Indices of the connected interaction sites.
    L: float
        Total nanorod length (in meters).
    d: float
        Nanorod diameter (in meters).
    smax: float
        Maximum ligand interaction distance (in meters).
    k_lig: float
        Spring constant of a single ligand (in N/m).
    boxLx, boxLy: float
        Simulation box dimensions (in meters).
    periodic_x, periodic_y: bool
        Periodic boundary conditions along x and y.

    --- Returns ---
    k_eff: ndarray
        Effective spring constants (in N/m).
    n_count: ndarray
        Number of ligands contributing to each connection.
    """
    n_count = np.zeros(len(ci), dtype=int)
    n_sites = site_offsets.shape[1]

    for q, (i, j, a, b) in enumerate(zip(ci, cj, si, sj)):
        pts1 = hidden_points_at_site(site_offsets[i, a], L, d, n_sites)
        pts2 = hidden_points_at_site(site_offsets[j, b], L, d, n_sites)

        pts1 += np.array([pos0[i, 0], pos0[i, 1], 0])
        pts2 += np.array([pos0[j, 0], pos0[j, 1], 0])

        sep = pts1[:, None]-pts2[None]
        sep = minimum_image(
            sep, boxLx, boxLy,
            periodic_x, periodic_y
        )

        dist = np.linalg.norm(sep, axis=-1)

        n_count[q] = len(match_unique_pairs(dist, smax))

    return k_lig*n_count, n_count

def build_nanorod_geometry(
    Nx, Ny, L, d, s, smax, ligand_surface_density, k_lig,
    periodic_x=False, periodic_y=False
):
    """
    Build the nanorod lattice and its spring network.

    --- Parameters ---
    Nx, Ny: int
        Number of nanorods along x and y.
    L: float
        Total nanorod length (in meters).
    d: float
        Nanorod diameter (in meters).
    s: float
        Nanorod-nanorod spacing (in meters).
    smax: float
        Maximum ligand interaction distance (in meters).
    ligand_surface_density: float
        Ligand surface density (in nm^-2).
    k_lig: float
        Spring constant of a single ligand (in N/m).
    periodic_x, periodic_y: bool
        Periodic boundary conditions along x and y.

    --- Returns ---
    geometry: dict
        Nanorod positions, spring connections and lattice parameters.
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

    pos0 = np.array([
        [col*dx, dy/2*(col%2+2*row)]
        for row in range(Ny)
        for col in range(Nx)
    ])

    boxLx = Nx*dx
    boxLy = Ny*dy

    sites_per_nm = np.sqrt(ligand_surface_density)

    offsets, n_sites = capsule_surface_offsets(L, d, sites_per_nm)
    site_offsets = np.repeat(offsets[None], len(pos0), axis=0)

    ci, cj, si, sj = build_springs_by_distance(
        Nx, Ny, pos0, site_offsets, smax,
        boxLx, boxLy, periodic_x, periodic_y
    )

    sites0 = get_sites(pos0, np.zeros(len(pos0)), site_offsets)

    sep0 = sites0[ci, si]-sites0[cj, sj]
    sep0 = minimum_image(
        sep0, boxLx, boxLy,
        periodic_x, periodic_y
    )

    springL = np.linalg.norm(sep0, axis=1)

    k_eff, n_count = compute_effective_spring_constants(
        pos0, site_offsets, ci, cj, si, sj,
        L, d, smax, k_lig,
        boxLx, boxLy, periodic_x, periodic_y
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