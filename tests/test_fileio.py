# -*- coding: utf-8 -*-
"""
io module tests
"""
import json

import pytest

from loci.config import CharacterizeConfig
from loci.fileio import (
    load_characterize_config,
    get_geom_info_parquet,
    get_geom_type_parquet,
    get_geom_type_pyogrio,
    get_crs_raster,
    get_crs_pyogrio,
    get_crs_parquet,
)


@pytest.mark.parametrize("from_dict", [True, False])
def test_load_characterize_config(data_dir, from_dict):
    """
    test that load_charactrize_config() works when passed either a dict or
    CharacterizeConfig input.
    """

    in_config_path = data_dir / "characterize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)

    config_data["data_dir"] = (data_dir / "characterize").as_posix()
    config_data["grid"] = (
        data_dir / "characterize" / "grids" / "grid_1.gpkg"
    ).as_posix()

    if from_dict:
        config = load_characterize_config(config_data)
    else:
        config = load_characterize_config(CharacterizeConfig(**config_data))

    assert isinstance(config, CharacterizeConfig)


def test_load_characterize_config_typerror():
    """
    Test that laod_characterize_config() raises a TypeError when passed an unsupported
    input.
    """

    with pytest.raises(TypeError, match="Invalid input for characterize config.*"):
        load_characterize_config("string input")


def test_get_geom_info_parquet(data_dir):
    """
    Happy path test for get_geom_info_parquet(). Test that it returns dict with
    expected keys for a valid geoparquet file.
    """
    dset_src = data_dir / "characterize" / "vectors" / "fiber_to_the_premises.parquet"
    geom_info = get_geom_info_parquet(dset_src)

    expected_keys = ["encoding", "crs", "geometry_types", "bbox", "covering"]
    assert (
        list(geom_info.keys()) == expected_keys
    ), "Geometry info dictionary does not have expected keys"


def test_get_geom_info_valueerror(data_dir):
    """
    Test that get_geom_info_valueerror() raises a ValueError for a Parquet file
    without geometry columns.
    """
    dset_src = data_dir / "edge_case_inputs" / "no_geometry.parquet"
    with pytest.raises(ValueError):
        get_geom_info_parquet(dset_src)


@pytest.mark.parametrize(
    "dset_name,expected_geom_type",
    [
        ("characterize/vectors/fiber_to_the_premises.parquet", "polygon"),
        ("characterize/vectors/generators.parquet", "point"),
        ("characterize/vectors/tlines.parquet", "line"),
        ("edge_case_inputs/combo_lines_multilines.parquet", "line"),
    ],
)
def test_get_geom_type_parquet(data_dir, dset_name, expected_geom_type):
    """
    Test that get_geom_type_parquet() returns correct type for known inputs.
    """
    dset_src = data_dir / dset_name
    geom_type = get_geom_type_parquet(dset_src)
    assert geom_type == expected_geom_type, "Unexpected geometry type identified"


@pytest.mark.parametrize(
    "dset_name,expected_geom_type",
    [
        ("characterize/vectors/fiber_to_the_premises.gpkg", "polygon"),
        ("characterize/vectors/generators.gpkg", "point"),
        ("characterize/vectors/tlines.gpkg", "line"),
        ("edge_case_inputs/combo_lines_multilines.gpkg", "line"),
    ],
)
def test_get_geom_type_pyogrio(data_dir, dset_name, expected_geom_type):
    """
    Test that get_geom_type_pyogrio() returns correct type for known inputs.
    """
    dset_src = data_dir / dset_name
    geom_type = get_geom_type_pyogrio(dset_src)
    assert geom_type == expected_geom_type, "Unexpected geometry type identified"


@pytest.mark.parametrize("in_format", ["parquet", "pyogrio"])
def test_get_geom_type_error_multipoint(data_dir, in_format):
    """
    Test that get_geom_type_parquet() and get_geom_type_pyogrio() both raise a
    ValueError when passed an input dataset that is MultiPoint.
    """
    if in_format == "parquet":
        dset_src = data_dir / "edge_case_inputs" / "multipoint.parquet"
        with pytest.raises(ValueError, match="Unsupported geometry type*."):
            get_geom_type_parquet(dset_src)
    elif in_format == "pyogrio":
        dset_src = data_dir / "edge_case_inputs" / "multipoint.gpkg"
        with pytest.raises(ValueError, match="Unsupported geometry type*."):
            get_geom_type_pyogrio(dset_src)


def test_get_crs_raster(data_dir):
    """
    Test for get_crs_raster()
    """
    dset_src = data_dir / "characterize" / "rasters" / "developable.tif"
    crs = get_crs_raster(dset_src)
    assert crs == "EPSG:5070", "Unexpected CRS value"


def test_get_crs_pyogrio(data_dir):
    """
    Test for get_crs_pyogrio()
    """
    dset_src = data_dir / "characterize" / "vectors" / "generators.gpkg"
    crs = get_crs_pyogrio(dset_src)
    assert crs == "EPSG:5070", "Unexpected CRS value"


def test_get_crs_parquet(data_dir):
    """
    Test for get_crs_parquet()
    """
    dset_src = data_dir / "characterize" / "vectors" / "generators.parquet"
    crs = get_crs_parquet(dset_src)
    assert crs == "EPSG:5070", "Unexpected CRS value"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
