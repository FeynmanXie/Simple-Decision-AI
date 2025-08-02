"""Setup script for the project."""
from setuptools import setup, find_packages

setup(
    name="simple-decision-ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'ai-cli=interfaces.cli:main',
        ],
    },
)
