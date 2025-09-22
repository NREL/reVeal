# -*- coding: utf-8 -*-
"""
overlay module tests
"""
import pytest
import numpy as np

from reVeal.score import get_values, calc_minmax, calc_percentile


@pytest.mark.parametrize(
    "attribute,invert",
    [
        ("generator_count", True),
        ("generator_count", False),
        ("fttp_coverage_pct", True),
        ("fttp_coverage_pct", False),
    ],
)
def test_get_values(characterized_df, attribute, invert):
    """
    Unit test for get_values()
    """
    in_values = characterized_df[attribute].copy()
    values = get_values(characterized_df, attribute, invert)
    expected_count = 16
    assert len(values) == expected_count, "Unexpected number of rows returned"
    assert (values.index == characterized_df.index).all(), "Index was modified"

    if invert:
        assert ((in_values + values) == 0).all(), "Unexpected values returned"
    else:
        assert (in_values == values).all(), "Unexpected values returned"


def test_get_values_missing_attribute(characterized_df):
    """
    Test that get_values() raises a KeyError when passed a missing attribute.
    """
    with pytest.raises(KeyError, match="not a column in dataframe"):
        get_values(characterized_df, "not-a-col", False)


def test_get_values_nonnumeric_attribute(characterized_df):
    """
    Test that get_values() raises a TypeError when passed a non-numeric attribute.
    """
    characterized_df["new-col"] = "foo"
    with pytest.raises(TypeError, match="must be numeric"):
        get_values(characterized_df, "new-col", False)


@pytest.mark.parametrize(
    "attribute,invert,first_val,last_val",
    [
        ("generator_count", False, 2 / 3, 0),
        ("generator_count", True, 1 / 3, 1),
        ("med_dist_to_lh_fiber", False, 0.001, np.nan),
        ("med_dist_to_lh_fiber", True, 0.999, np.nan),
    ],
)
def test_calc_minmax(characterized_df, attribute, invert, first_val, last_val):
    """
    Unit test for calc_minmax().
    """

    scored = calc_minmax(characterized_df, attribute, invert)
    if invert:
        # invert - max values should be 0, min values should be 1
        assert (
            scored[characterized_df[attribute] == characterized_df[attribute].min()][
                "value"
            ]
            == 1
        ).all(), "Unexpected values in scored result"
        assert (
            scored[characterized_df[attribute] == characterized_df[attribute].max()][
                "value"
            ]
            == 0
        ).all(), "Unexpected values in scored result"
    else:
        # regular - max values should be 1, min values should be 0
        assert (
            scored[characterized_df[attribute] == characterized_df[attribute].min()][
                "value"
            ]
            == 0
        ).all(), "Unexpected values in scored result"
        assert (
            scored[characterized_df[attribute] == characterized_df[attribute].max()][
                "value"
            ]
            == 1
        ).all(), "Unexpected values in scored result"

    assert np.isclose(
        scored["value"].iloc[0], first_val, atol=1e-3, equal_nan=True
    ), "Unexpected first value"
    assert np.isclose(
        scored["value"].iloc[-1], last_val, atol=1e-3, equal_nan=True
    ), "Unexpected last value"


@pytest.mark.parametrize(
    "attribute,invert,first_val,last_val",
    [
        ("generator_count", False, 0.8125, 0.3125),
        ("generator_count", True, 0.1875, 0.6875),
        ("med_dist_to_lh_fiber", False, 0.125, np.nan),
        ("med_dist_to_lh_fiber", True, 0.875, np.nan),
    ],
)
def test_calc_percentile(characterized_df, attribute, invert, first_val, last_val):
    """
    Unit test for calc_percentile().
    """

    scored = calc_percentile(characterized_df, attribute, invert)

    assert np.isclose(
        scored["value"].iloc[0], first_val, atol=1e-3, equal_nan=True
    ), "Unexpected first value"
    assert np.isclose(
        scored["value"].iloc[-1], last_val, atol=1e-3, equal_nan=True
    ), "Unexpected last value"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
