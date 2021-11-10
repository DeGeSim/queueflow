#!/usr/bin/env python

from setuptools import find_packages, setup

install_requirements = [
    "numpy",
    "setuptools",
    "torch",
    "torch-geometric",
    "torch-sparse",
    "multiprocessing-logging",
    "tqdm",
    "uproot",
    "rich",
]

setup(
    author="Anonymous",
    author_email="mova@users.noreply.github.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.8",
    ],
    description="A library for parallel processing using `torch.multiprocessing`.",
    install_requires=install_requirements,
    keywords="queueflow",
    name="queueflow",
    packages=find_packages(include=["queueflow"]),
    install_requirements=install_requirements,
    url="https://github.com/mova/queueflow",
    version="0.1.0",
    zip_safe=False,
)
