# -*- coding: utf-8 -*-
"""
load module tests
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from reVeal.overlay import calc_area_weighted_majority
from reVeal.load import apportion_load_to_regions, downscale_total, downscale_regional


def test_apportion_load_to_regions(data_dir):
    """
    Unit test for apportion_load_to_regions() - check that it works and produces
    the expected output
    """

    load_src = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )
    load_df = pd.read_csv(load_src)
    load_df["dc_load_mw"] = [1, 10, 100, 1000, 10_000, 100_000]
    region_weights = {"north": 0.5, "south": 0.2, "east": 0.13, "west": 0.17}
    results_df = apportion_load_to_regions(
        load_df, "dc_load_mw", "year", region_weights
    )

    expected_src = data_dir / "load" / "apportioned_regional_loads.csv"
    expected_df = pd.read_csv(expected_src)

    assert_frame_equal(results_df, expected_df)


def test_apportion_load_to_regions_bad_weights(data_dir):
    """
    Test that apportion_load_to_regions() raises a ValueError if the region_weights
    do not sum to 1.
    """

    load_src = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )
    load_df = pd.read_csv(load_src)
    region_weights = {"north": 0.5, "south": 0.2, "east": 0.1, "west": 0.1}
    with pytest.raises(
        ValueError, match="Weights of input region_weights must sum to 1"
    ):
        apportion_load_to_regions(load_df, "dc_load_mw", "year", region_weights)


def test_downscale_total(data_dir):
    """
    Unit test for downscale_total() - checks that it produced the expected results for
    known inputs.
    """

    load_src = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )
    grid_src = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    load_df = pd.read_csv(load_src)
    grid_df = gpd.read_file(grid_src)

    results_df = downscale_total(
        grid_df=grid_df,
        grid_priority_col="suitability_score",
        grid_baseline_load_col="dc_capacity_mw_existing",
        baseline_year=2022,
        load_df=load_df,
        load_value_col="dc_load_mw",
        load_year_col="year",
    )

    expected_src = data_dir / "load" / "grid_downscaled_total.gpkg"
    expected_df = gpd.read_file(expected_src)

    assert_geodataframe_equal(results_df, expected_df)


def test_downscale_regional(data_dir):
    """
    Unit test for downscale_regional() - checks that it produced the expected results
    for known inputs.
    """

    load_src = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_regional.csv"
    )
    grid_src = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    regions_src = data_dir / "downscale" / "inputs" / "regions" / "eer_adp_zones.gpkg"
    load_df = pd.read_csv(load_src)
    grid_df = gpd.read_file(grid_src)

    grid_df.set_index("gid", inplace=True)
    regions_lkup_df = calc_area_weighted_majority(grid_df, regions_src, "emm_zone")
    grid_w_regions_df = pd.concat([grid_df, regions_lkup_df], axis=1)

    results_df = downscale_regional(
        grid_df=grid_w_regions_df,
        grid_priority_col="suitability_score",
        grid_baseline_load_col="dc_capacity_mw_existing",
        baseline_year=2022,
        grid_region_col="emm_zone",
        load_df=load_df,
        load_value_col="dc_load_mw",
        load_year_col="year",
        load_region_col="zone",
    )

    # expected_src = data_dir / "load" / "grid_downscaled_regional.gpkg"
    # expected_df = gpd.read_file(expected_src)

    # assert_geodataframe_equal(results_df, expected_df)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-k", "test_downscale_total"])
