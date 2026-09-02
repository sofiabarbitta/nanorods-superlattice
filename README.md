# Nanorod Superlattice Simulation

Python simulation of mechanical wave propagation in a two-dimensional nanorod superlattice.

Nanorods are modeled as rigid bodies with translational and rotational degrees of freedom and interact through ligand-mediated effective springs. The model can generate longitudinal and transverse excitations propagating either parallel or perpendicular to the nanorod long axis.

## Project structure

```text
nanorod-superlattice/
├── setup.py
├── configuration.json
├── requirements.txt
├── README.md
├── nanorods/
│   ├── __init__.py
│   ├── geometry.py
│   ├── simulation.py
│   └── visualization.py
├── tests/
│   ├── test_geometry.py
│   └── test_simulation.py
└── run_examples/
    └── nanorod_simulation.ipynb
```

## Physical model

The nanorods are represented as rigid capsule-shaped particles arranged in a staggered two-dimensional lattice.

Each nanorod has two translational degrees of freedom along the $x$ and $y$ directions and one rotational degree of freedom around the $z$ axis.

Neighboring nanorods interact through ligand-mediated springs connecting interaction sites on their surfaces. Multiple ligand connections acting between the same surface regions are treated as springs acting in parallel, giving an effective spring constant

```math
k_{\mathrm{eff}} = N_{\mathrm{lig}} k_{\mathrm{lig}}.
```

where $N_{\mathrm{lig}}$ is the number of ligand connections and $k_{\mathrm{lig}}$ is the spring constant of a single ligand.

The translational and rotational dynamics are obtained from Newton's equations of motion and integrated using a kick-drift-kick scheme.

## Excitation modes

The simulation supports two propagation directions:

- `parallel`: propagation along the nanorod long axis, corresponding to the $x$ direction;
- `perpendicular`: propagation perpendicular to the nanorod long axis, corresponding to the $y$ direction.

For each propagation direction, two excitation modes are available:

- `longitudinal`;
- `transverse`.

The initial excitation is produced by applying a displacement to the central row or column of the lattice.

For parallel propagation, the central column is displaced. For perpendicular propagation, the central row is displaced. The direction of the displacement determines whether the excitation is longitudinal or transverse.

## Boundary conditions

Periodic boundary conditions are applied along the direction perpendicular to wave propagation.

For `parallel` propagation, the wave travels along $x$. Therefore:

- the $x$ direction is open;
- the $y$ direction is periodic.

For `perpendicular` propagation, the wave travels along $y$. Therefore:

- the $x$ direction is periodic;
- the $y$ direction is open.

The boundary conditions are selected automatically from the propagation direction.

Spring separations across periodic boundaries are evaluated using the minimum-image convention.

Because of the staggered lattice geometry, an even number of columns `Nx` is required when periodic boundary conditions are applied along $x$.

## Configuration

The main simulation parameters are defined in `configuration.json`.

Example:

```json
{
    "lattice": {
        "Nx": 36,
        "Ny": 80
    },
    "nanorod": {
        "aspect_ratio": 3.5,
        "diameter_nm": 25.0,
        "spacing_nm": 3.4
    },
    "ligands": {
        "maximum_interaction_distance_nm": 4.4,
        "surface_density_nm2": 4.0,
        "spring_constant_N_m": 2.0
    },
    "excitation": {
        "amplitude_nm": 0.5,
        "direction": "parallel",
        "mode": "longitudinal"
    },
    "simulation": {
        "time_step_ps": 1.0,
        "total_time_ns": 1.5
    }
}
```

The notebook reads these parameters automatically and converts the quantities to SI units before running the simulation.

The excitation parameters can be changed using:

```text
direction = "parallel" or "perpendicular"
mode = "longitudinal" or "transverse"
```

The periodic boundary conditions are then determined automatically from the selected propagation direction.

## Installation

Clone the repository:

```bash
git clone https://github.com/sofiabarbitta/nanorods-superlattice.git
cd nanorods-superlattice
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Install the `nanorods` package in editable mode:

```bash
python -m pip install -e .
```

The package can then be imported directly from Python, for example:

```python
from nanorods.geometry import build_nanorod_geometry
from nanorods.simulation import run_simulation
```

The editable installation allows changes to the source code to be immediately available without reinstalling the package.

## Running the simulation

After installing the package, open the example notebook:

```text
run_examples/nanorod_simulation.ipynb
```

and run the cells from top to bottom.

The notebook performs the following steps:

1. reads the parameters from `configuration.json`;
2. calculates the nanorod mass and moment of inertia;
3. constructs the staggered nanorod lattice;
4. generates the ligand-mediated spring network;
5. applies the appropriate periodic boundary conditions;
6. applies the initial line displacement;
7. integrates the equations of motion;
8. visualizes the nanorod configuration at a user-selected simulation time.

## Visualization

The nanorod configuration can be displayed at any selected simulation time.

For example, the user can define

```python
target_time = 0.5e-9
```

and the notebook automatically selects the closest available simulation frame.

The color of each nanorod represents the magnitude of its center-of-mass displacement,

```math
|\mathbf{u}_i| = |\mathbf{r}_i-\mathbf{r}_{i,0}|.
```

A common color normalization is used for the full simulation so that displacement amplitudes can be compared between different times.

## Simulation output

The simulation stores the time evolution of:

- nanorod center positions;
- nanorod rotation angles;
- center-of-mass displacements;
- simulation times.

The main arrays returned by the simulation are:

```text
pos_profiles
theta_profiles
u_profiles
time_profiles
```

## Testing

The project includes unit tests for the geometry and dynamics modules.

The tests cover:

- nanorod geometry;
- interaction-site construction;
- spring-network generation;
- initial line displacements;
- translational and rotational dynamics;
- simulation output;
- periodic boundary conditions;
- minimum-image corrections.

Run the complete test suite with:

```bash
python -m pytest
```

Test coverage for the numerical modules can be evaluated with:

```bash
python -m pytest \
    --cov=nanorods.geometry \
    --cov=nanorods.simulation \
    --cov-report=term-missing \
    tests/
```

## Model assumptions and limitations

The current implementation:

- uses a two-dimensional rigid-body description of the nanorods;
- includes translational and rotational degrees of freedom;
- models ligand-mediated interactions through effective harmonic springs;
- treats multiple ligand connections as parallel springs;
- considers line-displacement excitations only;
- uses periodic boundary conditions only perpendicular to the propagation direction;
- does not include atomistic deformation of the nanorods;
- does not include damping;
- does not include thermal fluctuations;
- does not perform automatic wave-velocity or dispersion-relation analysis.

The model is intended to provide a simplified numerical description of mechanical wave propagation in ordered nanorod assemblies.