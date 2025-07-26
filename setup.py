"""
Basic setup script for slide_autoencoder development package
"""

from setuptools import setup, find_packages

# Read requirements from file
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="slide_autoencoder",
    version="0.1.0",
    author="Tereza Jurickova",
    description="Autoencoder architectures for histopathology image denoising",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=read_requirements(),
)
