#!/usr/bin/env python3
"""
Setup script for HumanEvalComm V2 Evaluators Framework
"""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="human-eval-comm",
    version="2.0.0",
    author="Jie JW Wu, Fatemeh H. Fard",
    author_email="",
    description="HumanEvalComm: Benchmarking the Communication Competence "
                "of Code Generation for LLMs and LLM Agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jie-jw-wu/human-eval-comm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "scikit-learn>=1.3.0",
        "docker>=6.0.0",
        "pytest>=7.0.0",
        "pylint>=2.15.0",
        "bandit>=1.7.0",
        "radon>=5.1.0",
        "mypy>=1.0.0",
        "psutil>=5.9.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "pathlib>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "huggingface": [
            "transformers>=4.21.0",
            "torch>=1.12.0",
            "datasets>=2.4.0",
            "evaluate>=0.4.0",
            "plotly>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evaluate-code=examples.evaluate_code:main",
            "setup-evaluators=scripts.setup_evaluators:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
