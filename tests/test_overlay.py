# -*- coding: utf-8 -*-
"""
overlay module tests
"""
import pytest
from geopandas.testing import assert_geodataframe_equal
import pandas as pd
import geopandas as gpd

from loci.overlay import calc_feature_count, calc_sum_attribute, calc_sum_length


def test_calc_feature_count(data_dir, base_grid):
    """
    Unit tests for calc_feature_count().
    """
    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / "generators.gpkg"
    results = calc_feature_count(zones_df, dset_src)

    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = data_dir / "overlays" / "feature_count_results.gpkg"
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "attribute,exception_type",
    [
        ("net_generation_megawatthours", None),
        ("utility_name", TypeError),
        ("not_a_column", KeyError),
    ],
)
def test_calc_sum_attribute(data_dir, base_grid, attribute, exception_type):
    """
    Unit tests for calc_sum_attribute().
    """
    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / "generators.gpkg"

    if exception_type:
        with pytest.raises(exception_type):
            calc_sum_attribute(zones_df, dset_src, attribute)
    else:
        results = calc_sum_attribute(zones_df, dset_src, attribute)

        results_df = pd.concat([zones_df, results], axis=1)
        results_df.reset_index(inplace=True)

        expected_results_src = data_dir / "overlays" / "feature_sum_results.gpkg"
        expected_df = gpd.read_file(expected_results_src)

        assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "dset_name", ["tlines.gpkg", "generators.gpkg", "fiber_to_the_premises.gpkg"]
)
def test_calc_sum_length(data_dir, base_grid, dset_name):
    """
    Unit tests for calc_sum_length().
    """

    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / dset_name

    results = calc_sum_length(zones_df, dset_src)

    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = data_dir / "overlays" / f"sum_length_results_{dset_name}"
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
