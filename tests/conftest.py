# -*- coding: utf-8 -*-
"""
pytest fixtures
"""
import json
import warnings

import pytest
from click.testing import CliRunner
import geopandas as gpd

from reVeal import PACKAGE_DIR
from reVeal.grid import (
    BaseGrid,
    CharacterizeGrid,
    NormalizeGrid,
    ScoreWeightedGrid,
    TotalDownscaleGrid,
    RegionalDownscaleGrid,
)

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
    grid = BaseGrid(template=template_src)

    return grid


@pytest.fixture
def char_grid():
    """Return a CharacterizeGrid instance"""

    in_config_path = TEST_DATA_DIR / "characterize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["data_dir"] = (TEST_DATA_DIR / "characterize").as_posix()
    config_data["grid"] = (
        TEST_DATA_DIR / "characterize" / "grids" / "grid_1.gpkg"
    ).as_posix()

    grid = CharacterizeGrid(config_data)

    return grid


@pytest.fixture
def norm_grid():
    """Return a NormalizeGrid instance"""

    in_config_path = TEST_DATA_DIR / "normalize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (
        TEST_DATA_DIR / "characterize" / "outputs" / "grid_char.gpkg"
    ).as_posix()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = NormalizeGrid(config_data)

    return grid


@pytest.fixture
def characterized_df():
    """Return an output dataframe from grid characterization"""

    in_path = TEST_DATA_DIR / "characterize" / "outputs" / "grid_char.gpkg"

    df = gpd.read_file(in_path)

    return df


@pytest.fixture
def score_wt_grid():
    """Return a ScoreWeightedGrid instance"""

    in_config_path = TEST_DATA_DIR / "score_weighted" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (
        TEST_DATA_DIR / "normalize" / "outputs" / "grid_normalized.gpkg"
    ).as_posix()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = ScoreWeightedGrid(config_data)

    return grid


@pytest.fixture
def downscale_total_grid():
    """Return a DownscaleGrid instance for downscaling total projections"""

    in_config_path = TEST_DATA_DIR / "downscale" / "config_total.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (TEST_DATA_DIR / config_data["grid"]).as_posix()
    config_data["load_projections"] = (
        TEST_DATA_DIR / config_data["load_projections"]
    ).as_posix()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = TotalDownscaleGrid(config_data)

    return grid


@pytest.fixture
def downscale_regional_grid():
    """Return a DownscaleGrid instance for downscaling regional projections"""

    in_config_path = TEST_DATA_DIR / "downscale" / "config_regional.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (TEST_DATA_DIR / config_data["grid"]).as_posix()
    config_data["load_projections"] = (
        TEST_DATA_DIR / config_data["load_projections"]
    ).as_posix()
    config_data["regions"] = (TEST_DATA_DIR / config_data["regions"]).as_posix()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = RegionalDownscaleGrid(config_data)

    return grid


@pytest.fixture
def downscale_region_weights_grid():
    """
    Return a DownscaleGrid instance for downscaling total projections using region
    weights
    """

    in_config_path = TEST_DATA_DIR / "downscale" / "config_region_weights.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (TEST_DATA_DIR / config_data["grid"]).as_posix()
    config_data["load_projections"] = (
        TEST_DATA_DIR / config_data["load_projections"]
    ).as_posix()
    config_data["regions"] = (TEST_DATA_DIR / config_data["regions"]).as_posix()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = RegionalDownscaleGrid(config_data)

    return grid
