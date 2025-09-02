# -*- coding: utf-8 -*-
"""
grid module tests
"""
import json

import pytest
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from reVeal.grid import (
    Grid,
    CharacterizeGrid,
    create_grid,
    get_neighbors,
    get_overlay_method,
    run_characterization,
)
from reVeal.config import CharacterizeConfig


@pytest.mark.parametrize(
    "bounds,res,crs,i",
    [
        (
            [71281.01960453, 743256.58450656, 117361.01960453, 789336.58450656],
            11520,
            "EPSG:5070",
            2,
        ),
        ([-124.70999145, 24.3696785, -64.70999145, 49.3696785], 5, "EPSG:4326", 3),
    ],
)
def test_create_grid(data_dir, bounds, res, crs, i):
    """
    Unit test for create_grid().
    """

    grid_df = create_grid(res, *bounds, crs)
    expected_src = data_dir / "characterize" / "grids" / f"grid_{i}.gpkg"
    expected_df = gpd.read_file(expected_src)

    assert len(grid_df) == len(
        expected_df
    ), "Output grid does not have expected number of rows"

    grid_df["geometry"] = grid_df["geometry"].normalize()
    expected_df["geometry"] = expected_df["geometry"].normalize()
    grid_df.sort_values(by="geometry", inplace=True)
    grid_df.reset_index(drop=True, inplace=True)
    expected_df.sort_values(by="geometry", inplace=True)
    expected_df.reset_index(drop=True, inplace=True)

    equal_geoms = grid_df["geometry"].geom_equals_exact(
        expected_df["geometry"], tolerance=0.1
    )
    assert equal_geoms.all(), "Geometries do not match expected outputs"


@pytest.mark.parametrize(
    "crs,bounds,res",
    [
        ("EPSG:4326", None, None),  # check reprojection,
        (None, [78161, 757417, 90827, 763932], None),  # check bbox subsetting
        (None, None, 5000),  # check res (should raise a warning)
        (
            "EPSG:4326",
            [-95.1901, 29.8882, -95.0589, 29.9469],
            0.1,
        ),  # all three together
    ],
)
def test_init_grid_from_template(data_dir, crs, bounds, res):
    """
    Test for initializing Grid instance from a template file.
    """

    template_src = data_dir / "characterize" / "grids" / "grid_1.gpkg"

    if res is not None:
        with pytest.warns(UserWarning):
            grid = Grid(template=template_src, crs=crs, bounds=bounds, res=res)
    else:
        grid = Grid(template=template_src, crs=crs, bounds=bounds, res=res)

    template_df = gpd.read_file(template_src)

    if not crs:
        crs = template_df.crs
    assert grid.crs == crs, "Unexpected CRS"

    if bounds:
        expected_count = 2
    else:
        expected_count = len(template_df)
    assert len(grid.df) == expected_count, "Unexpected number of features in grid"

    assert grid.df.index.name == "gid", "Index of grid not set properly"


@pytest.mark.parametrize(
    "crs,bounds,res,i",
    [
        (
            "EPSG:5070",
            [71281.01960453, 743256.58450656, 117361.01960453, 789336.58450656],
            11520,
            2,
        ),  # known grid 1
        (
            "EPSG:4326",
            [-124.70999145, 24.3696785, -64.70999145, 49.3696785],
            5,
            3,
        ),  # known grid 2
        (None, [0, 1, 2, 3], 5, None),  # unspecified CRS (should raise error)
        ("EPSG:5070", None, 5, None),  # unspecified bounds (should raise error)
        ("EPSG:5070", [0, 1, 2, 3], None, None),  # unspecified res (should error)
    ],
)
def test_init_grid_from_scratch(data_dir, crs, bounds, res, i):
    """
    Test for initializing Grid instance from crs, bounds, and res parameters.
    """

    if crs is None or bounds is None or res is None:
        # if any are None, a ValueError should be raised
        with pytest.raises(ValueError, match="If template is not provided*."):
            Grid(crs=crs, bounds=bounds, res=res)
    else:
        expected_src = data_dir / "characterize" / "grids" / f"grid_{i}.gpkg"
        expected_df = gpd.read_file(expected_src)

        grid = Grid(crs=crs, bounds=bounds, res=res)

        assert len(grid.df) == len(expected_df), "Unexpected number of features in grid"
        assert grid.crs == crs, "Unexpected grid crs"
        assert grid.df.index.name == "gid", "Index of grid not set properly"


@pytest.mark.parametrize("order", [0, 1, 2])
def test_get_neighbors(base_grid, order, data_dir):
    """
    Test Grid.neighbors() function.
    """

    neighbors_df = get_neighbors(base_grid.df, order)
    expected_neighbors_src = data_dir / "grid" / f"grid_2_neighbors_{order}.gpkg"
    expected_neighbors_df = gpd.read_file(expected_neighbors_src)
    expected_neighbors_df.set_index("gid", inplace=True)

    assert_geodataframe_equal(neighbors_df, expected_neighbors_df)


@pytest.mark.parametrize("as_dict", [False, True])
def test_init_characterizegrid(data_dir, as_dict):
    """
    Test that CharacterizeGrid can be initialized from either a dictionary or
    a CharacterizeConfig.
    """

    in_config_path = data_dir / "characterize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["data_dir"] = (data_dir / "characterize").as_posix()
    config_data["grid"] = (
        data_dir / "characterize" / "grids" / "grid_1.gpkg"
    ).as_posix()

    if as_dict:
        grid = CharacterizeGrid(config_data)
    else:
        grid = CharacterizeGrid(CharacterizeConfig(**config_data))

    assert len(grid.df) == 9, "Unexpected row count in grid.df"
    assert grid.crs == "EPSG:5070", "Unexpected grid.crs"
    assert isinstance(
        grid.config, CharacterizeConfig
    ), "grid.config is not a CharacterizeConfig instance"


def test_run_characterizegrid(char_grid):
    """
    Test the run() function of CharacterizeGrid.
    """
    char_grid.run()


@pytest.mark.parametrize(
    "method_name,error_expected",
    [
        ("feature_count", False),
        ("feature count", False),
        ("featurecount", True),
        ("Feature-Count", False),
        ("sum attribute", False),
        ("sum length", False),
        ("sum attribute-length", False),
        ("sum area", False),
        ("percent covered", False),
        ("area-weighted average", False),
        ("area-apportioned sum", False),
        ("mean", False),
        ("median", False),
        ("sum", False),
        ("area", False),
        ("not a method", True),
    ],
)
def test_get_overlay_method(method_name, error_expected):
    """
    Test get_overlay_method() returns a valid callable function, when expected,
    and if not raises a NotImplementedError.
    """
    if error_expected:
        with pytest.raises(
            NotImplementedError, match="Unrecognized or unsupported method.*"
        ):
            get_overlay_method(method_name)
    else:
        f = get_overlay_method(method_name)
        assert callable(f), "Returned method is not callable"


def test_run_characterization(char_grid):
    """
    Test the run_characterization() function either returns the expected dataframe
    or raises a NotImplementedError, for methods that have not been implemented.
    """
    df = char_grid.df
    for characterization in char_grid.config.characterizations.values():
        try:
            result_df = run_characterization(df, characterization)
            assert len(result_df) == len(df), "Unexpected row count in result_df"
            assert "value" in result_df.columns, "Value column not in result_df"
        except NotImplementedError as e:
            if not str(e).startswith("Unrecognized or unsupported method"):
                raise e


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
