# -*- coding: utf-8 -*-
"""
load module tests
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from reVeal.load import apportion_load_to_regions


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


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
