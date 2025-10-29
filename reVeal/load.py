# -*- coding: utf-8 -*-
"""
load module
"""
from math import isclose

import numpy as np
import pandas as pd
import tqdm


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


def downscale_total(
    grid_df,
    grid_priority_col,
    grid_baseline_load_col,
    baseline_year,
    load_df,
    load_value_col,
    load_year_col,
):
    # TODO: expose these up the stack so it can be input through the config
    # TODO: remove baseline year input from config - we aren't using it??
    grid_capacity_col = "developable_capacity_mw"
    site_saturation = 1
    priority_power = 3

    n_iters = 10_000
    n_nonzero = (grid_df[grid_priority_col] > 0).sum()
    random_seed = 0

    grid_df["_weights"] = grid_df[grid_priority_col] ** priority_power
    grid_df[f"total_{load_value_col}"] = grid_df[grid_baseline_load_col].astype(float)
    grid_df[f"new_{load_value_col}"] = float(0.0)
    # note: don't decrement off existing load because developable capacity
    # should already account for exclusions from existing buildings
    grid_df["_developable_capacity"] = grid_df[grid_capacity_col] * site_saturation
    grid_idx = grid_df.index.name
    if grid_idx is None:
        grid_idx = "index"

    grid_year_df = grid_df.reset_index()
    grid_year_df["year"] = baseline_year
    grid_years = [grid_year_df.copy()]

    load_df.sort_values(by=[load_year_col], ascending=True, inplace=True)
    for year, year_df in load_df.groupby(by=[load_year_col]):
        grid_year_df["year"] = year[0]

        if len(year_df) > 1:
            raise ValueError(f"Multiple records for load projections year {year}")
        load_projected_in_year = year_df[load_value_col].iloc[0]

        simulations = []
        # TODO: remove tqdm
        for _ in tqdm.tqdm(range(0, n_iters)):
            shuffle_df = grid_year_df.sample(
                n=n_nonzero,
                replace=False,
                weights="_weights",
                random_state=random_seed,
                ignore_index=True,
            )
            shuffle_df["_new_capacity"] = 0.0

            cumulative_developable = shuffle_df["_developable_capacity"].cumsum()
            cumulative_exceeds_total = cumulative_developable > load_projected_in_year
            last_deployed_idx = np.argmax(cumulative_exceeds_total)

            deployed_df = shuffle_df.iloc[0 : last_deployed_idx + 1]

            new_cap_col_idx = deployed_df.columns.get_loc("_new_capacity")
            dev_cap_col_idx = deployed_df.columns.get_loc("_developable_capacity")

            deployed_df.iloc[0:last_deployed_idx, new_cap_col_idx] = deployed_df.iloc[
                0:last_deployed_idx, dev_cap_col_idx
            ]

            total_from_filled_sites = deployed_df["_new_capacity"].sum()

            remaining_capacity = load_projected_in_year - total_from_filled_sites
            deployed_df.iloc[last_deployed_idx, new_cap_col_idx] = remaining_capacity

            total_deployed = deployed_df["_new_capacity"].sum()
            if not isclose(total_deployed, load_projected_in_year):
                raise ValueError("Deployed total is not equal to projected total")

            simulations.append(deployed_df[[grid_idx, "_new_capacity"]])

            random_seed += 1

        simulations_df = pd.concat(simulations, ignore_index=True)
        means_df = simulations_df.groupby(by=[grid_idx])[["_new_capacity"]].median()
        means_df["_proportion"] = (
            means_df["_new_capacity"] / means_df["_new_capacity"].sum()
        )
        means_df["_new_calibrated_capacity"] = (
            means_df["_proportion"] * load_projected_in_year
        )
        total_calibrated_deployed = means_df["_new_calibrated_capacity"].sum()
        if not isclose(total_calibrated_deployed, load_projected_in_year):
            raise ValueError("Deployed total is not equal to projected total")

        grid_year_df.set_index(grid_idx, inplace=True)
        grid_year_df.loc[means_df.index, f"new_{load_value_col}"] = means_df[
            "_new_calibrated_capacity"
        ]
        grid_year_df[f"total_{load_value_col}"] += grid_year_df[f"new_{load_value_col}"]
        grid_year_df["_developable_capacity"] -= grid_year_df[f"new_{load_value_col}"]
        grid_year_df[f"new_{load_value_col}"] = float(0.0)
        grid_year_df.reset_index(inplace=True)

        grid_years.append(grid_year_df.copy())

    grid_projections_df = pd.concat(grid_years, ignore_index=True)
    grid_projections_df.set_index([grid_idx, "year"], inplace=True)

    return grid_projections_df


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
