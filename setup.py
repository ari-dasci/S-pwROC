#!/usr/bin/env python3

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pwROC',
    version='0.0.1dev1',
    description="Temporal generalization of ROC curves and evaluation measures for weakly labelled anomalies in time-series scenarios.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts":[
            "pwROC-cli=pwROC.cli:main",
            "pwROC-BD-cli=pwROC.pwROCBD.cli:main [pwROCBD]"
        ]
    },
    install_requires=[
        'docopt',
        'numpy',
        'pandas',
        'sklearn',
        'pyarrow',
        'seaborn'
    ],
    extras_requires={
        'pwROCBD': ['pyspark'],
    },
    python_requires='>=3.6'
)
