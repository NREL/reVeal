# -*- coding: utf-8 -*-
"""
grid module tests
"""
import pytest
import geopandas as gpd

from loci.grid import create_grid, Grid


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
    assert len(grid.grid) == expected_count, "Unexpected number of features in grid"


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

        assert len(grid.grid) == len(
            expected_df
        ), "Unexpected number of features in grid"
        assert grid.crs == crs, "Unexpected grid crs"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
