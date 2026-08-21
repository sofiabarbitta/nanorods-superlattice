"""Visualization functions for the nanorod superlattice simulation."""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def rod_patch(x, y, theta, L, d, facecolor="#D4AF37", n=80):
    """
    Draw a capsule-shaped nanorod.

    --- Parameters ---
    x, y: float
        Nanorod center position.
    theta: float
        Nanorod rotation angle (in radians).
    L: float
        Total nanorod length.
    d: float
        Nanorod diameter.
    facecolor:
        Nanorod color.
    n: int
        Number of points used for each rounded end.

    --- Returns ---
    patch: PathPatch
        Matplotlib patch representing the nanorod.
    """
    r = d/2
    xc = (L-d)/2

    t1 = np.linspace(-np.pi/2, np.pi/2, n)
    t2 = np.linspace(np.pi/2, 3*np.pi/2, n)

    pts = np.c_[
        np.r_[xc+r*np.cos(t1), -xc+r*np.cos(t2)],
        np.r_[r*np.sin(t1), r*np.sin(t2)]
    ]

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])

    pts = pts@R.T

    pts[:, 0] += x
    pts[:, 1] += y

    return PathPatch(
        Path(
            np.r_[pts, pts[:1]],
            [Path.MOVETO]+[Path.LINETO]*(len(pts)-1)+[Path.CLOSEPOLY]
        ),
        facecolor=facecolor,
        edgecolor="black",
        lw=0.5
    )


def plot_lattice(pos, theta, u, L, d, vmax=None, title=None):
    """
    Plot the nanorod lattice with color proportional to displacement.

    --- Parameters ---
    pos: ndarray, shape (N, 2)
        Nanorod center positions (in meters).
    theta: ndarray, shape (N,)
        Nanorod rotation angles (in radians).
    u: ndarray, shape (N, 2)
        Nanorod center-of-mass displacements (in meters).
    L: float
        Total nanorod length (in meters).
    d: float
        Nanorod diameter (in meters).
    vmax: float or None
        Maximum displacement used for color normalization (in meters).
    title: str or None
        Optional figure title.

    --- Returns ---
    fig, ax
        Matplotlib figure and axes.
    """
    pos_nm = pos/1e-9
    L_nm = L/1e-9
    d_nm = d/1e-9

    displacement = np.linalg.norm(u, axis=1)

    if vmax is None:
        vmax = np.max(displacement)

    if vmax == 0:
        vmax = 1

    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap("turbo")

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(pos)):
        color = cmap(norm(displacement[i]))

        ax.add_patch(
            rod_patch(
                pos_nm[i, 0],
                pos_nm[i, 1],
                theta[i],
                L_nm,
                d_nm,
                facecolor=color
            )
        )

    ax.autoscale()
    ax.margins(0.02)
    ax.set_aspect("equal")

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    if title is not None:
        ax.set_title(title)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$|\mathbf{r}-\mathbf{r}_0|$ (m)")

    fig.tight_layout()

    return fig, ax