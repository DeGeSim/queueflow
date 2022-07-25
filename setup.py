#!/usr/bin/env python

from setuptools import find_packages, setup

install_requirements = [
    "numpy >= 1.21.0",
    "torch >= 1.8.0",
    "setuptools",
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
