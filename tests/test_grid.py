# -*- coding: utf-8 -*-
"""
grid module tests
"""
import json
import warnings

import pytest
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from reVeal.grid import (
    BaseGrid,
    RunnableGrid,
    CharacterizeGrid,
    create_grid,
    get_neighbors,
    get_method_from_members,
    run_characterization,
    OVERLAY_METHODS,
    NORMALIZE_METHODS,
    run_weighted_scoring,
)
from reVeal.config.config import BaseGridConfig
from reVeal.config.characterize import CharacterizeConfig


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
def test_init_basegrid_from_template(data_dir, crs, bounds, res):
    """
    Test for initializing Grid instance from a template file.
    """

    template_src = data_dir / "characterize" / "grids" / "grid_1.gpkg"

    if res is not None:
        with pytest.warns(UserWarning):
            grid = BaseGrid(template=template_src, crs=crs, bounds=bounds, res=res)
    else:
        grid = BaseGrid(template=template_src, crs=crs, bounds=bounds, res=res)

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
def test_init_basegrid_from_scratch(data_dir, crs, bounds, res, i):
    """
    Test for initializing Grid instance from crs, bounds, and res parameters.
    """

    if crs is None or bounds is None or res is None:
        # if any are None, a ValueError should be raised
        with pytest.raises(ValueError, match="If template is not provided*."):
            BaseGrid(crs=crs, bounds=bounds, res=res)
    else:
        expected_src = data_dir / "characterize" / "grids" / f"grid_{i}.gpkg"
        expected_df = gpd.read_file(expected_src)

        grid = BaseGrid(crs=crs, bounds=bounds, res=res)

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


def test_init_runnablegrid(data_dir):
    """
    Test intializable of a RunnableGrid from a config. Ensure that the run() method
    raises a NotImplementedError.
    """

    grid_src = data_dir / "characterize" / "grids" / "grid_1.gpkg"
    config = BaseGridConfig(grid=grid_src)
    grid = RunnableGrid(config=config)
    assert grid.config == config, "Unexpected value for grid.config"
    with pytest.raises(NotImplementedError, match="run method not implemented"):
        grid.run()


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
    "method_name,members,error_expected",
    [
        ("feature_count", OVERLAY_METHODS, False),
        ("feature count", OVERLAY_METHODS, False),
        ("featurecount", OVERLAY_METHODS, True),
        ("Feature-Count", OVERLAY_METHODS, False),
        ("sum attribute", OVERLAY_METHODS, False),
        ("sum length", OVERLAY_METHODS, False),
        ("sum attribute-length", OVERLAY_METHODS, False),
        ("sum area", OVERLAY_METHODS, False),
        ("percent covered", OVERLAY_METHODS, False),
        ("area-weighted average", OVERLAY_METHODS, False),
        ("area-apportioned sum", OVERLAY_METHODS, False),
        ("mean", OVERLAY_METHODS, False),
        ("median", OVERLAY_METHODS, False),
        ("sum", OVERLAY_METHODS, False),
        ("area", OVERLAY_METHODS, False),
        ("not a method", OVERLAY_METHODS, True),
        ("minmax", NORMALIZE_METHODS, False),
        ("min-max", NORMALIZE_METHODS, True),
        ("percentile", NORMALIZE_METHODS, False),
        ("percentiles", NORMALIZE_METHODS, True),
    ],
)
def test_get_method_from_members(method_name, members, error_expected):
    """
    Test get_overlay_method() returns a valid callable function, when expected,
    and if not raises a NotImplementedError.
    """
    if error_expected:
        with pytest.raises(
            NotImplementedError, match="Unrecognized or unsupported method.*"
        ):
            get_method_from_members(method_name, members)
    else:
        f = get_method_from_members(method_name, members)
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


@pytest.mark.parametrize(
    "bad_expression",
    [
        "@pd.compat.os.system('echo foo')",
        "@warnings.warn('AH AH AH!')",
        "os.system('echo foo')",
    ],
)
def test_run_characterization_with_expression_injection(
    char_grid, capfd, recwarn, bad_expression
):
    """
    Unit test that ensures that attempts to inject system level commmands using
    expressions does not work.
    """

    char_grid.config.expressions = {"bad_actor": bad_expression}
    char_grid.config.characterizations = {}
    char_grid.run()
    captured_stdout = capfd.readouterr().out
    assert (
        captured_stdout == ""
    ), "stdout is not empty. Injection occurred via dataframe.eval()."
    if len(recwarn) > 0:
        assert (
            str(recwarn[0].message) != "AH AH AH!"
        ), "Warning message injected via dataframe.eval()"


def test_run_normalizegrid(norm_grid):
    """
    Test the run() function of NormalizeGrid.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm_grid.run()


def test_run_normalizegrid_overwrite_output(data_dir, norm_grid):
    """
    Test the run() function of NormalizeGrid correctly overwrites an existing
    output column.
    """
    out_col = list(norm_grid.config.attributes.keys())[0]
    norm_grid.df[out_col] = 1
    assert (
        norm_grid.df[out_col] == 1
    ).all(), "New column not added with expected values"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_df = norm_grid.run()

    expected_norm_src = data_dir / "normalize" / "outputs" / "grid_normalized.gpkg"

    expected_df = gpd.read_file(expected_norm_src)
    assert (
        result_df[out_col] == expected_df[out_col]
    ).all(), "Unexpected output values"


def test_run_weighted_scoring(data_dir, score_wt_grid):
    """
    Test that the run_weighted_scoring() method produces the expected outputs
    for known inputs.
    """

    df = score_wt_grid.df
    attributes = score_wt_grid.config.attributes

    scores_df = run_weighted_scoring(df, attributes)
    expected_scores_csv = data_dir / "score_weighted" / "outputs" / "test_scores.csv"

    expected_df = pd.read_csv(expected_scores_csv)
    assert_frame_equal(scores_df.reset_index(), expected_df)


def test_run_weighted_scoring_bad_weights(score_wt_grid):
    """
    Test that run_weighted_scoring() raises a ValueError when weights don't sum to 1.
    """

    df = score_wt_grid.df
    attributes = score_wt_grid.config.attributes
    for attribute in attributes:
        attribute.weight = 0.1

    with pytest.raises(ValueError, match="Weights of input attributes must sum to 1"):
        run_weighted_scoring(df, attributes)


def test_run_weighted_scoring_bad_col(score_wt_grid):
    """
    Test that run_weighted_scoring() raises a KeyError when passed an attribute column
    that doesn't exist.
    """

    df = score_wt_grid.df
    attributes = score_wt_grid.config.attributes
    attributes[0].attribute = "not-a-col"

    with pytest.raises(KeyError, match="not in index"):
        run_weighted_scoring(df, attributes)


def test_run_scoreweightedgrid(score_wt_grid):
    """
    Test the run() function of ScoreWeightedGrid.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score_wt_grid.run()


def test_run_scoreweightedgrid_overwrite_output(data_dir, score_wt_grid):
    """
    Test the run() function of ScoreWeightedGrid correctly overwrites an existing
    output column.
    """

    out_col = score_wt_grid.config.score_name
    score_wt_grid.df[out_col] = 1

    assert (
        score_wt_grid.df[out_col] == 1
    ).all(), "New column not added with expected values"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_df = score_wt_grid.run()

    expected_scores_csv = data_dir / "score_weighted" / "outputs" / "test_scores.csv"

    expected_df = pd.read_csv(expected_scores_csv)
    assert (
        result_df[out_col] == expected_df["value"]
    ).all(), "Unexpected output values"


def test_run_totaldownscalegrid(data_dir, downscale_total_grid):
    """
    Test the run() function of DownscaleGrid with total resolution load projections.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_df = downscale_total_grid.run()

    expected_src = (
        data_dir / "downscale" / "outputs" / "grid_downscaled_total_year_cap.gpkg"
    )
    expected_df = gpd.read_file(expected_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


def test_run_regionaldownscalegrid(data_dir, downscale_regional_grid):
    """
    Test the run() function of DownscaleGrid with regional resolution load projections.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_df = downscale_regional_grid.run()

    # replace nans with nones to avoid future warnings when we compare to expected_df
    results_df["zone_group"] = np.where(
        results_df["zone_group"].isna(), None, results_df["zone_group"]
    )

    expected_src = (
        data_dir / "downscale" / "outputs" / "grid_downscaled_regional_year_cap.gpkg"
    )
    expected_df = gpd.read_file(expected_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


def test_run_regionaldownscalegrid_weights(data_dir, downscale_region_weights_grid):
    """
    Test the run() function of DownscaleGrid with total resolution load projections
    and region weights.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_df = downscale_region_weights_grid.run()

    results_df["zone_group"] = np.where(
        results_df["zone_group"].isna(), None, results_df["zone_group"]
    )

    expected_src = (
        data_dir
        / "downscale"
        / "outputs"
        / "grid_downscaled_region_weights_year_cap.gpkg"
    )
    expected_df = gpd.read_file(expected_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-k", "test_run_downscalegrid_regional"])
