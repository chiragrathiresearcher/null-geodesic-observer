from setuptools import setup, find_packages

setup(
    name             = "null-geodesic-observer",
    version          = "0.1.0",
    author           = "Chirag Rathi",
    description      = (
        "Null Geodesic Observer (NGO): Compute and collect "
        "light travel time asymmetry in curved spacetime."
    ),
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url              = "https://github.com/ChiragRathi/null-geodesic-observer",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = [
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "plotly>=5.14",
        "pandas>=2.0",
    ],
    extras_require   = {
        "dev": ["pytest", "jupyter", "notebook"]
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
