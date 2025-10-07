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
        ("characterize/outputs/grid_char.parquet", None),
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
        config = BaseGridConfig(grid=grid_src)

        # check dynamic attributes are set
        assert config.grid_ext is not None, "grid_ext not set"
        assert config.grid_ext == grid_src.suffix, "Unexpected value for grid_ext"

        assert config.grid_flavor is not None, "grid_flavor not set"
        if config.grid_ext == ".parquet":
            expected_flavor = "geoparquet"
        else:
            expected_flavor = "ogr"
        assert config.grid_flavor == expected_flavor, "Unexpected value for grid_flavor"


def test_basegridconfig_nonexistent_grid():
    """
    Test that BaseGridConfig raises a ValidationError when passed a non-existent
    grid.
    """
    with pytest.raises(ValidationError, match="Path does not point to a file"):
        BaseGridConfig(grid="not-a-file.gpkg")


def test_basegridconfig_bad_format(tmp_path):
    """
    Test that BaseGridConfig raises a TypeError when passed an unsupported
    data format.
    """
    grid_src = tmp_path / "grid.tif"
    grid_src.touch()

    with pytest.raises(TypeError, match="Unrecognized file format"):
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
