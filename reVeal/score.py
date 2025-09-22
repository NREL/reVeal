# -*- coding: utf-8 -*-
"""
score module

Note that to expose methods here for by reVeal.grid.get_scoring_method() function
and functions dependent on it, the function must be prefixed with "calc_".
"""
import pandas as pd
from scipy.stats import percentileofscore


def get_values(df, attribute, invert):
    """
    Get values of the specified attribute from the input dataframe, optionally
    inverting via multiplying by -1.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    attribute : str
        Name of attribute for which to get values.
    invert : bool
        If True, invert values by multiplying by -1. If False, return values
        as is from df.

    Returns
    -------
    pandas.Series
        Series of values. The returned series will have the name "value".

    Raises
    ------
    KeyError
        A KeyError will be raised if the specified attribute is not found in the input
        dataframe
    TypeError
        A TypeError will be raised if the specified attribute is not a numeric dtype.
    """

    if attribute not in df.columns:
        raise KeyError(f"attribute {attribute} not a column in dataframe")

    if not pd.api.types.is_numeric_dtype(df[attribute]):
        raise TypeError(f"attribute {attribute} in dataframe must be numeric")

    values = df[attribute].copy()
    if invert:
        values *= -1

    values.name = "value"

    return values


def calc_percentile(df, attribute, invert):
    """
    Score values from 0 to 1 using percentile ranking.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    attribute : str
        Name of attribute to score.
    invert : bool
        If True, invert values by multiplying by -1 before scoring.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scores stored in a column named "value". Also includes the index
        from the input dataframe.
    """

    values = get_values(df, attribute, invert)
    scores = pd.Series(
        percentileofscore(values, values, kind="mean", nan_policy="omit") / 100.0,
        index=values.index,
        name="value",
    )

    scores_df = scores.to_frame()

    return scores_df


def calc_minmax(df, attribute, invert):
    """
    Score values from 0 to 1 by normalizing the range of values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    attribute : str
        Name of attribute to score.
    invert : bool
        If True, invert values by multiplying by -1 before scoring.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scores stored in a column named "value". Also includes the index
        from the input dataframe.
    """

    values = get_values(df, attribute, invert)
    scores = (values - values.min()) / (values.max() - values.min())

    scores_df = scores.to_frame()

    return scores_df
