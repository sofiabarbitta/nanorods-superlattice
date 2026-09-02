from setuptools import find_packages, setup

setup(
    name="nanorod-superlattice",
    version="0.1.0",
    description="Simulation of mechanical wave propagation in 2D nanorod superlattices",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.10",
)