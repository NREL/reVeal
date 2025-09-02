# -*- coding: utf-8 -*-
"""
pytest fixtures
"""
import json

import pytest
from click.testing import CliRunner

from reVeal import PACKAGE_DIR
from reVeal.grid import Grid, CharacterizeGrid

TEST_DATA_DIR = PACKAGE_DIR.parent.joinpath("tests", "data")


@pytest.fixture
def data_dir():
    """Return path to test data directory"""
    return TEST_DATA_DIR


@pytest.fixture
def cli_runner():
    """Return a click CliRunner for testing commands"""
    return CliRunner()


@pytest.fixture
def base_grid():
    """Return a Grid instance"""
    template_src = TEST_DATA_DIR / "characterize" / "grids" / "grid_2.gpkg"
    grid = Grid(template=template_src)

    return grid


@pytest.fixture
def char_grid():
    """Return a CharacterizeConfig instance"""

    in_config_path = TEST_DATA_DIR / "characterize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["data_dir"] = (TEST_DATA_DIR / "characterize").as_posix()
    config_data["grid"] = (
        TEST_DATA_DIR / "characterize" / "grids" / "grid_1.gpkg"
    ).as_posix()

    grid = CharacterizeGrid(config_data)

    return grid
