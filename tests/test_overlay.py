# -*- coding: utf-8 -*-
"""
overlay module tests
"""
import pytest
from geopandas.testing import assert_geodataframe_equal
import pandas as pd
import geopandas as gpd

from loci.overlay import calc_feature_count


def test_calc_feature_count(data_dir, base_grid):
    """
    Check that calc_feature_count() returns correct results per grid cell.
    """
    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / "generators.gpkg"
    results = calc_feature_count(zones_df, dset_src)

    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = data_dir / "overlays" / "feature_count_results.gpkg"
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
