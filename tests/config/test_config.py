# -*- coding: utf-8 -*-
"""
config.config module tests
"""
import pytest
from pydantic import ValidationError

from reVeal.config.config import BaseGridConfig, load_config


@pytest.mark.parametrize(
    "grid,err",
    [
        ("characterize/grids/grid_2.gpkg", None),
        ("not-a-grid.gpkg", ValidationError),
    ],
)
def test_basegridconfig(data_dir, grid, err):
    """
    Unit tests for BaseGridConfig.
    """

    grid_src = data_dir / grid
    if err:
        with pytest.raises(err):
            BaseGridConfig(grid=grid_src)
    else:
        BaseGridConfig(grid=grid_src)


def test_load_config_from_dict(data_dir):
    """
    Test that load_config() works on an input dictionary.
    """
    grid = data_dir / "characterize" / "grids" / "grid_2.gpkg"
    config = load_config({"grid": grid}, BaseGridConfig)
    assert isinstance(config, BaseGridConfig)


def test_load_config_from_config(data_dir):
    """
    Test that load_config() works on an input config instance.
    """
    grid = data_dir / "characterize" / "grids" / "grid_2.gpkg"
    config = load_config(BaseGridConfig(grid=grid), BaseGridConfig)
    assert isinstance(config, BaseGridConfig)


def test_load_config_from_badtype(data_dir):
    """
    Test that load_config() raises a TypeError when passed an invalid input.
    """
    grid = data_dir / "characterize" / "grids" / "grid_2.gpkg"
    with pytest.raises(TypeError, match="Invalid input for config"):
        load_config(grid, BaseGridConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
