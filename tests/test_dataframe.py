# -*- coding: utf-8 -*-
"""
dataframe module tests
"""
import pytest

import pandas as pd
from geopandas.testing import assert_geodataframe_equal

from reVeal.dataframe import dataframe_split


@pytest.mark.parametrize("n_parts", [1, 2, 3, 4, 9, 10, 15, 120])
def test_dataframe_split(base_grid, n_parts):
    """Unit test for dataframe_split function"""
    df = base_grid.df
    splits = list(dataframe_split(df, n_parts))
    expected_n_splits = min(len(df), n_parts)
    assert len(splits) == expected_n_splits, "Unexpected number of splits"
    rebuilt_df = pd.concat(splits)

    # check that the full dataframe is represented
    assert_geodataframe_equal(df, rebuilt_df)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
