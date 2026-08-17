from setuptools import setup, find_packages

setup(
    name="physrna-filter",
    version="0.1.0",
    description="Physics-informed validation of AI-generated protein-RNA complexes",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "biopython>=1.81",
        "MDAnalysis>=2.6.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
        "pdbfixer>=1.9",
        "openmm>=8.0",
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "fair-esm>=2.0.0",
    ],
    extras_require={
        "rosetta": ["pyrosetta"],
        "rnafm": ["rna-fm"],
        # SASA burial term for ProNAB benchmarking (needs MSVC on Windows)
        "benchmark": ["freesasa>=2.2"],
        # AF3 screening on Windows without C++ build tools
        "af3": [
            "biopython>=1.81",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "requests>=2.31.0",
            "torch>=2.0.0",
            "torch-geometric>=2.4.0",
            "fair-esm>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "physrna=physrna_filter.cli:main",
            "physrna-filter=physrna_filter.pipeline:_cli",
            "physrna-rank=physrna_filter.validation.rank_af3_candidates:main",
            "physrna-report=physrna_filter.validation.report_af3:main",
            "physrna-screen=physrna_filter.validation.screen_af3:main",
            "physrna-benchmark-foldbench=physrna_filter.validation.benchmark_foldbench:main",
            "physcomplex=physrna_filter.physcomplex.__main__:main",
        ],
    },
)
