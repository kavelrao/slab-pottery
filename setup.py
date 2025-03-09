from setuptools import setup, find_packages

setup(
    name="slab-pottery",
    version="0.1.0",
    description="Slab pottery tool for converting 3D pottery models to 2D patterns",
    author="",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "trimesh",
        "matplotlib",
        "tqdm",
        "networkx",
        "svgwrite",
    ],
)
