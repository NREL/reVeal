# -*- coding: utf-8 -*-
"""
load module tests
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import numpy as np

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


@pytest.mark.parametrize("max_site_addition_per_year", [1000, None])
@pytest.mark.parametrize("min_site_addition_per_year", [1, None])
@pytest.mark.parametrize("reduce_output", [False, True])
def test_downscale_total(data_dir, max_site_addition_per_year, min_site_addition_per_year, reduce_output):
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
        grid_capacity_col="developable_capacity_mw",
        load_df=load_df,
        load_value_col="dc_load_mw",
        load_year_col="year",
        max_site_addition_per_year=max_site_addition_per_year,
        min_site_addition_per_year=min_site_addition_per_year,
        site_saturation_limit=0.5,
        priority_power=100,
        n_bootstraps=500,
        random_seed=0,
        reduce_output=reduce_output,
    )

    suffix = "_centroid" if reduce_output else ""

    if max_site_addition_per_year:
        expected_src = (
            data_dir
            / "downscale"
            / "outputs"
            / f"grid_downscaled_total_year_cap{suffix}.gpkg"
        )
    else:
        expected_src = (
            data_dir / "downscale" / "outputs" / f"grid_downscaled_total{suffix}.gpkg"
        )

    if min_site_addition_per_year:
        # With a minimum floor, verify that no site has new load between
        # 0 (exclusive) and the minimum threshold
        for year in results_df["year"].unique():
            year_df = results_df[results_df["year"] == year]
            new_load = year_df["new_dc_load_mw"]
            violators = new_load[(new_load > 0) & (new_load < min_site_addition_per_year)]
            assert len(violators) == 0, (
                f"Year {year}: {len(violators)} sites have new load below "
                f"min_site_addition_per_year ({min_site_addition_per_year})"
            )
    else:
        expected_df = gpd.read_file(expected_src)
        assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize("max_site_addition_per_year", [1000, None])
@pytest.mark.parametrize("min_site_addition_per_year", [1, None])
@pytest.mark.parametrize("reduce_output", [False, True])
def test_downscale_regional(data_dir, max_site_addition_per_year, min_site_addition_per_year, reduce_output):
    """
    Unit test for downscale_regional() - checks that it produced the expected results
    for known inputs.
    """

    load_src = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_region_groups.csv"
    )
    grid_src = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    regions_src = data_dir / "downscale" / "inputs" / "regions" / "eer_adp_zones.gpkg"
    load_df = pd.read_csv(load_src)
    grid_df = gpd.read_file(grid_src)

    grid_df.set_index("gid", inplace=True)
    regions_lkup_df = calc_area_weighted_majority(grid_df, regions_src, "zone_group")
    grid_w_regions_df = pd.concat([grid_df, regions_lkup_df], axis=1)

    results_df = downscale_regional(
        grid_df=grid_w_regions_df,
        grid_priority_col="suitability_score",
        grid_baseline_load_col="dc_capacity_mw_existing",
        baseline_year=2022,
        grid_capacity_col="developable_capacity_mw",
        grid_region_col="zone_group",
        load_df=load_df,
        load_value_col="dc_load_mw",
        load_year_col="year",
        load_region_col="zone_group",
        max_site_addition_per_year=max_site_addition_per_year,
        min_site_addition_per_year=min_site_addition_per_year,
        site_saturation_limit=0.5,
        priority_power=100,
        n_bootstraps=100,
        random_seed=0,
        reduce_output=reduce_output,
    )

    # replace nans with nones to avoid future warnings when we compare to expected_df
    results_df["zone_group"] = np.where(
        results_df["zone_group"].isna(), None, results_df["zone_group"]
    )

    suffix = "_centroid" if reduce_output else ""

    if max_site_addition_per_year:
        expected_src = (
            data_dir
            / "downscale"
            / "outputs"
            / f"grid_downscaled_regional_year_cap{suffix}.gpkg"
        )
    else:
        expected_src = (
            data_dir
            / "downscale"
            / "outputs"
            / f"grid_downscaled_regional{suffix}.gpkg"
        )

    if min_site_addition_per_year:
        for year in results_df["year"].unique():
            year_df = results_df[results_df["year"] == year]
            new_load = year_df["new_dc_load_mw"]
            violators = new_load[(new_load > 0) & (new_load < min_site_addition_per_year)]
            assert len(violators) == 0, (
                f"Year {year}: {len(violators)} sites have new load below "
                f"min_site_addition_per_year ({min_site_addition_per_year})"
            )
    else:
        expected_df = gpd.read_file(expected_src)
        assert_geodataframe_equal(results_df, expected_df, check_like=True)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
