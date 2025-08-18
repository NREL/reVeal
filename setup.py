# -*- coding: utf-8 -*-
"""
Setup package
"""
import os
import re

from setuptools import setup, find_packages


REPO_DIR = os.path.abspath(os.path.dirname(__file__))
VERSION_FILE = os.path.join(REPO_DIR, "loci", "version.py")
DESCRIPTION = "Land Opportunity & Characterization Insights model."

with open(VERSION_FILE, encoding="utf-8") as f:
    VERSION = f.read().split("=")[-1].strip().strip('"').strip("'")

with open(os.path.join(REPO_DIR, "README.md"), encoding="utf-8") as f:
    README = f.read()

with open("requirements.txt") as f:
    INSTALL_REQUIREMENTS = f.readlines()
with open("environment.yml") as f:
    all_lines = [l.replace("\n", "").rstrip() for l in f.readlines()]
deps_start_line = all_lines.index("dependencies:") + 1
all_dep_lines = all_lines[deps_start_line:]
all_deps = [l.lstrip().replace("- ", "") for l in all_dep_lines]
if "pip:" in all_deps:
    pip_start_line = all_deps.index("pip:")
else:
    pip_start_line = -1  # pylint: disable=invalid-name
conda_deps = [l for l in all_deps[:pip_start_line] if not l.startswith("python")]
INSTALL_REQUIREMENTS += conda_deps

SKIP_DEPS = ["proj-data"]
for skip_dep in SKIP_DEPS:
    skip_matches = list(filter(re.compile(skip_dep).match, INSTALL_REQUIREMENTS))
    for skip_match in skip_matches:
        INSTALL_REQUIREMENTS.pop(INSTALL_REQUIREMENTS.index(skip_match))

with open("requirements_dev.txt") as f:
    DEV_REQUIREMENTS = f.readlines()

setup(
    name="loci",
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    author="Michael Gleason",
    author_email="mike.gleason@nrel.gov",
    packages=find_packages(),
    package_dir={"loci": "loci"},
    entry_points={
        "console_scripts": [
            "loci=loci.cli.cli:main",
        ],
    },
    zip_safe=False,
    keywords="loci",
    python_requires=">=3.12",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    test_suite="tests",
    include_package_data=True,
    package_data={"": ["data/*", "data/**/*"]},
    install_requires=INSTALL_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
    },
)
