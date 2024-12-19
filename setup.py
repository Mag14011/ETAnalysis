from setuptools import setup, find_packages

setup(
    name="electron_transfer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tabulate",
        "seaborn",
        "scikit-learn",
        "pandas",
        "networkx",
        "sympy"
    ],
    entry_points={
        'console_scripts': [
            'et-derrida=electron_transfer.cli:derrida_main',
            'et-flux=electron_transfer.cli:flux_main',
            'et-coop=electron_transfer.cli:cooperativity_main',
            'et-explore=electron_transfer.cli:parameter_explorer_main',
            'et-visualize=electron_transfer.cli:parameter_visualization_main',  # Added visualization
            'et-redox=electron_transfer.cli:redox_titration_main',
        ],
    },
    python_requires=">=3.7",
    author="Matthew J. Guberman-Pfeffer",
    author_email="Matthew_Guberman-Pfe@baylor.edu",
    description="A package for analyzing electron transfer in multi-heme proteins",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
