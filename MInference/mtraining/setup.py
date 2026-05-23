# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from setuptools import find_packages, setup

setup(
    name="mtraining",  # Name of your project
    version="0.1.0",
    packages=find_packages(),  # Automatically discover all packages
    install_requires=[],  # List dependencies if any (or use requirements.txt)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Specify the Python version
)
