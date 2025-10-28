# -*- coding: utf-8 -*-
"""
load module
"""
from math import isclose

import numpy as np
import pandas as pd


def apportion_load_to_regions(load_df, load_value_col, load_year_col, region_weights):
    """
    Apportion aggregate load projections to regions based on a priori input region
    weights.

    Parameters
    ----------
    load_df : pandas.DataFrame
        Load projections dataframe. Should be aggregate totals.
    load_value_col : str
        Name of column containing values of load projections.
    load_year_col : str
        Name of column containing the years associated with each projected load.
    region_weights : dict
        Dictionary indicating weights to use for apportioning load to regions. Keys
        should correspond to region names and values to the proportion of total
        load that should be apportioned to that region. All weights must sum to 1.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas dataframe with the load projections apportioned to regions.
        Dataframe will have a year column (named based on ``load_year_col``), a region
        column (named ``"region"``), and a load projection value column (named based on
        ``load_value_col``.)

    Raises
    ------
    ValueError
        A ValueError will be raised if the input region_weights do not sum to 1.
    """

    weights = np.array(list(region_weights.values()))

    if not isclose(weights.sum(), 1, abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError(
            "Weights of input region_weights must sum to 1. "
            f"Sum of input weights is: {weights.sum()}."
        )

    region_values = load_df[load_value_col].values[:, np.newaxis] * weights
    region_values_df = pd.DataFrame(
        region_values, columns=region_weights.keys(), index=load_df.index
    )

    combined_df = pd.concat([load_df, region_values_df], axis=1)
    combined_df.drop(columns=[load_value_col], inplace=True)

    region_loads_df = combined_df.melt(
        id_vars=[load_year_col], var_name="region", value_name=load_value_col
    )

    return region_loads_df


def downscale_regional(
    grid_df,
    grid_priority_col,
    grid_baseline_load_col,
    baseline_year,
    grid_region_col,
    load_df,
    load_value_col,
    load_year_col,
    load_region_col,
):
    # TODO: drop grids with unknown regions
    # TODO: check for validity/consistency of regions across datasets
    # TODO: rename columns for consistency across the input datasets

    return grid_df


def downscale_total(
    grid_df,
    grid_priority_col,
    grid_baseline_load_col,
    baseline_year,
    load_df,
    load_value_col,
    load_year_col,
):
    # TODO: rename columns for consistency across the input datasets

    return grid_df
