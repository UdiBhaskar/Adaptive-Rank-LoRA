"""Setup script for Adaptive Rank LoRA package."""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-rank-lora",
    version="0.1.0",
    author="Uday Paila, Naveen Pandey, Balakrishna Pailla, Gaurav Aggarwal",
    author_email="udaybhaskarpaila@gmail.com",
    description="Fine-Tuning Large Language Models using LoRA with Adaptive Rank Allocation Based on Spectral Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UdiBhaskar/Adaptive-Rank-LoRA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.18.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "jupyterlab>=3.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-lora-train=src.lora_dynamic_rank:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "transformers",
        "fine-tuning",
        "lora",
        "parameter efficient",
        "spectral analysis",
        "random matrix theory",
        "heavy-tailed self-regularization",
    ],
)
