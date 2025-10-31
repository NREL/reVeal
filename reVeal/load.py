# -*- coding: utf-8 -*-
"""
load module
"""
import logging
from math import isclose
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tqdm


LOGGER = logging.getLogger(__name__)


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


def _simulate_deployment(
    load_projected_in_year, grid_year_df, grid_idx, grid_weights, random_seed
):
    """
    Helper function for downscale_total() that simulates deployment of load
    to the grid of sites. Randomly shuffled sites, with weights, and downscales
    the total projected load to sites based on the shuffled order.

    Parameters
    ----------
    load_projected_in_year : float
        Total projected load to downscale.
    grid_year_df : pandas.DataFrame
        Grid of sites to which load will be downscaled. In addition to ``grid_idx`` and
        ``grid_weights`` columns, must also have a column named
        ``_developable_capacity``.
    grid_idx : str
        Name of column in ``grid_year_df`` corresponding to the index.
    grid_weights : str
        Name of column in ``grid_year_df`` to use for weighting random shuffle.
    random_seed : int
        Random seed to use for the random shuffle.

    Returns
    -------
    pandas.DataFrame
        Returns a simplified DataFrame of the downscaled results with two columns:
        an index column named based on ``grid_idx``, and a ``_new_capacity`` column
        indicating the downscaled capacity.

    Raises
    ------
    ValueError
        A ValueError will be raised if, during consistency checks, the downscaled
        capacity does not sum to the specified ``load_projected_in_year``
    """
    shuffle_df = grid_year_df.sample(
        frac=1,
        replace=False,
        weights=grid_weights,
        random_state=random_seed,
        ignore_index=True,
    )
    shuffle_df["_new_capacity"] = 0.0

    cumulative_developable = shuffle_df["_developable_capacity_inc"].cumsum()
    cumulative_exceeds_total = cumulative_developable > load_projected_in_year
    last_deployed_idx = np.argmax(cumulative_exceeds_total)

    deployed_df = shuffle_df.iloc[0 : last_deployed_idx + 1]

    new_cap_col_idx = deployed_df.columns.get_loc("_new_capacity")
    dev_cap_col_idx = deployed_df.columns.get_loc("_developable_capacity_inc")

    deployed_df.iloc[0:last_deployed_idx, new_cap_col_idx] = deployed_df.iloc[
        0:last_deployed_idx, dev_cap_col_idx
    ]

    total_from_filled_sites = deployed_df["_new_capacity"].sum()

    remaining_capacity = load_projected_in_year - total_from_filled_sites
    deployed_df.iloc[last_deployed_idx, new_cap_col_idx] = remaining_capacity

    total_deployed = deployed_df["_new_capacity"].sum()
    if not isclose(total_deployed, load_projected_in_year):
        raise ValueError("Deployed total is not equal to projected total")

    return deployed_df[[grid_idx, "_new_capacity", "_developable_capacity_inc"]]


def downscale_total(
    grid_df,
    grid_priority_col,
    grid_baseline_load_col,
    baseline_year,
    grid_capacity_col,
    load_df,
    load_value_col,
    load_year_col,
    max_site_addition_per_year=None,
    site_saturation_limit=1,
    priority_power=1,
    n_bootstraps=10_000,
    random_seed=0,
    max_workers=None,
):
    """
    Downscale aggregate load projections to grid based on grid priority column.
    Note that this method uses a random bootstrapping approach to achieve greater
    dispersion of load across multiple grid cells, and the degree of dispersion can
    be tuned manually using input parameters such as ``site_saturation_limit``,
    ``priority_power``, and ``n_bootstraps``.

    Parameters
    ----------
    grid_df : pandas.DataFrame
        Pandas dataframe where each record represents a site to which load projections
        may be downscaled
    grid_priority_col : str
        Name of column in ``grid_df`` to use for prioritizing sites for downscaling
        load.
    grid_baseline_load_col : str
        Name of column in ``grid_df`` with numeric values indicating the baseline, or
        initial, load in each site, corresponding to the ``baseline_year``.
    baseline_year : int
        Year corresponding to the baseline load values in ``grid_baseline_load_col``.
    grid_capacity_col : str
        Name of column in ``grid_df`` indicating the developable capacity of
        load within each site.
    load_df : pandas.DataFrame
        Dataframe containing aggregate load projections for the area encompassing the
        input ``grid_df`` sites.
    load_value_col : str
        Name of column in ``load_df`` containing projections of load.
    load_year_col : str
        Name of column in ``load_df`` containing year values corresponding to load
        projections.
    max_site_addition_per_year : float, optional
        Value indicating the maximum allowable increment of load that can be added in
        a given year to an individual site. The default value is None, which will not
        apply a cap. This value can be used to ensure that the rate of expansion of
        large load capacity in localized areas is not unrealistically rapid. Using
        this parameter can also have the effect of achieving greater geographic
        dispersion of load: since there is a limit to the pace at which individual
        sites can build out load, more sites are typically required for the same amount
        of project load.
    site_saturation_limit : float, optional
        Adjustment factor limit the developable capacity of load within each site.
        This value is used to scale the values in the ``grid_capacity_col``. For
        example, to limit the maximum deployed load in each site to half of the
        actual developable load, use ``site_saturation_limit=0.5``. The lower this
        value is set, the greater the degree of dispersion of load across sites will
        be. The dfault is 1, which leaves the values in the ``grid_capacity_col``
        unmodified.
    priority_power : int, optional
        This factor can be used to exaggerate the influence of the values in
        ``grid_priority_col``, such that higher values have an increased likelihood of
        load deployment and lower values have a decreased likelihood. This effect is
        implemented by raising the values in ``grid_priority_col`` to the specified
        ``priority_power``. As a result, if the input  values in ``grid_priority_col``
        are < 1, setting ``priority_power`` to high values can result in completely
        eliminating lower priority sites from consideration. The default value is 1,
        which leaves the values in ``grid_priority_col`` unmodified. To achieve
        less dispersion and greater clustering of downscaled load in higher priority
        sites, increase this value.
    n_bootstraps : int, optional
        Number of bootstraps to simulate in each projection year. Default is 10,000.
        In general, larger values will produce more stable results, with less chance
        for lower priority sites to receive large amounts of deployed load. However,
        larger values will also cause longer run times.
    random_seed : int, optional
        Random seed to use for reproducible bootstrapping. Default is 0. In general,
        this value does not need to be modified. The exception is if you are interested
        in testing sensitivities and/or producing multiple realizations or scenarios of
        deployment for a given set of values in ``load_priority_col``.
    max_workers : int, optional
        Number of workers to use for bootstrapping. By default None, which uses all
        available workers. In general, this value should only be changed if you are
        running into out-of-memory errors.

    Returns
    -------
    pandas.DataFrame
        Returns DataFrame consisting of load projections downscaled to the grid.
        This dataframe will contain all of the columns from the input ``grid_df``,
        as well as three new columns, including ``year`` (indicating the year
        of the projection) and a "new_" and "total_" load column, named with a suffix
        corresponding to the ``load_value_col``.

    Raises
    ------
    ValueError
        A ValueError will be raised if internal consistency checks for downscaled
        results do not pass.
    """

    grid_df["_weight"] = grid_df[grid_priority_col] ** priority_power
    grid_df[f"total_{load_value_col}"] = grid_df[grid_baseline_load_col].astype(float)
    grid_df[f"new_{load_value_col}"] = float(0.0)
    # note: don't decrement off existing load because developable capacity
    # should already account for exclusions from existing buildings
    grid_df["_developable_capacity"] = (
        grid_df[grid_capacity_col] * site_saturation_limit
    )
    grid_idx = grid_df.index.name
    if grid_idx is None:
        named_index = False
        grid_idx = "index"
    else:
        named_index = True

    grid_year_df = grid_df.reset_index()
    grid_year_df["year"] = baseline_year
    prior_year = baseline_year
    grid_years = [grid_year_df.copy()]

    load_df.sort_values(by=[load_year_col], ascending=True, inplace=True)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for group_id, year_df in load_df.groupby(by=[load_year_col]):
            year = group_id[0]
            grid_year_df["year"] = year
            grid_year_df[f"new_{load_value_col}"] = float(0.0)
            if max_site_addition_per_year:
                years_since_prior = year - prior_year
                grid_year_df["_developable_capacity_inc"] = np.minimum(
                    max_site_addition_per_year * years_since_prior,
                    grid_year_df["_developable_capacity"],
                )
            else:
                grid_year_df["_developable_capacity_inc"] = grid_year_df[
                    "_developable_capacity"
                ]

            if len(year_df) > 1:
                raise ValueError(f"Multiple records for load projections year {year}")
            load_projected_in_year = year_df[load_value_col].iloc[0]

            simulations = []
            futures = {}
            grid_year_sub_df = grid_year_df[grid_year_df["_weight"] > 0][
                [grid_idx, "_developable_capacity_inc", "_weight"]
            ]
            with tqdm.tqdm(
                total=n_bootstraps, desc=f"Running bootstraps for year {year}"
            ) as pbar:
                for i in range(0, n_bootstraps):
                    future = pool.submit(
                        _simulate_deployment,
                        load_projected_in_year=load_projected_in_year,
                        grid_year_df=grid_year_sub_df,
                        grid_idx=grid_idx,
                        grid_weights="_weight",
                        random_seed=random_seed,
                    )
                    futures[future] = i
                    random_seed += 1

                for future in as_completed(futures):
                    i = futures[future]
                    deployed_df = future.result()
                    simulations.append(deployed_df)
                    pbar.update(1)

            simulations_df = pd.concat(simulations, ignore_index=True)
            means_df = simulations_df.groupby(by=[grid_idx])[
                ["_new_capacity", "_developable_capacity_inc"]
            ].mean()
            means_df["_proportion"] = (
                means_df["_new_capacity"] / means_df["_new_capacity"].sum()
            )
            means_df["_new_calibrated_capacity"] = (
                means_df["_proportion"] * load_projected_in_year
            )
            total_calibrated_deployed = means_df["_new_calibrated_capacity"].sum()

            if not isclose(total_calibrated_deployed, load_projected_in_year):
                raise ValueError("Deployed total is not equal to projected total")

            overbuilt = (
                means_df["_new_calibrated_capacity"]
                > means_df["_developable_capacity_inc"]
            )
            if overbuilt.any():
                raise ValueError(
                    f"Downscaled load for {overbuilt.sum} sites exceeds the maximum "
                    f"developable capacity in year {year}."
                )

            grid_year_df.set_index(grid_idx, inplace=True)
            grid_year_df.loc[means_df.index, f"new_{load_value_col}"] = means_df[
                "_new_calibrated_capacity"
            ]
            grid_year_df[f"total_{load_value_col}"] += grid_year_df[
                f"new_{load_value_col}"
            ]
            grid_year_df["_developable_capacity"] -= grid_year_df[
                f"new_{load_value_col}"
            ]
            grid_year_df.reset_index(inplace=True)

            grid_years.append(grid_year_df.copy())
            prior_year = year

    grid_projections_df = pd.concat(grid_years, ignore_index=True)

    drop_cols = ["_developable_capacity", "_developable_capacity_inc", "_weight"]
    if not named_index:
        drop_cols.append(grid_idx)
    grid_projections_df.drop(columns=drop_cols, inplace=True)

    return grid_projections_df


def downscale_regional(
    grid_df,
    grid_priority_col,
    grid_baseline_load_col,
    baseline_year,
    grid_capacity_col,
    grid_region_col,
    load_df,
    load_value_col,
    load_year_col,
    load_region_col,
    max_site_addition_per_year=None,
    site_saturation_limit=1,
    priority_power=1,
    n_bootstraps=10_000,
    random_seed=0,
    max_workers=None,
):
    """
    Downscale regional load projections to grid based on grid priority column.
    This method works by wrapping downscale_total() over multiple regions.

    Parameters
    ----------
    grid_df : pandas.DataFrame
        Pandas dataframe where each record represents a site to which load projections
        may be downscaled
    grid_priority_col : str
        Name of column in ``grid_df`` to use for prioritizing sites for downscaling
        load.
    grid_baseline_load_col : str
        Name of column in ``grid_df`` with numeric values indicating the baseline, or
        initial, load in each site, corresponding to the ``baseline_year``.
    baseline_year : int
        Year corresponding to the baseline load values in ``grid_baseline_load_col``.
    grid_capacity_col : str
        Name of column in ``grid_df`` indicating the developable capacity of
        load within each site. Note that this value can modified using the
        ``site_saturation_limit`` parameter.
    grid_region_col : str
        Name of column in ``grid_df`` indicating the region of each site. Values
        should match those in the ``load_df`` ``load_region_col`` column (although
        they do not need to be the same type case).
    load_df : pandas.DataFrame
        Dataframe containing aggregate load projections for the area encompassing the
        input ``grid_df`` sites.
    load_value_col : str
        Name of column in ``load_df`` containing projections of load.
    load_year_col : str
        Name of column in ``load_df`` containing year values corresponding to load
        projections.
    load_region_col : str
        Name of column in ``load_df`` indicating the region of each projected value.
        Values should match those in the ``grid_df`` ``grid_region_col`` column
        (although they do not need to be the same type case).
    max_site_addition_per_year : float, optional
        Value indicating the maximum allowable increment of load that can be added in
        a given year to an individual site. The default value is None, which will not
        apply a cap. This value can be used to ensure that the rate of expansion of
        data center capacity in localized areas is not unrealistically rapid. Using
        this parameter can also have the effect of achieving greater geographic
        dispersion of load: since there is a limit to the pace at which individual
        sites can build out load, more sites are typically required for the same amount
        of project load.
    site_saturation_limit : float, optional
        Adjustment factor limit the developable capacity of load within each site.
        This value is used to scale the values in the ``grid_capacity_col``. For
        example, to limit the maximum deployed load in each site to half of the
        actual developable load, use ``site_saturation_limit=0.5``. The lower this
        value is set, the greater the degree of dispersion of load  across sites will
        be. The dfault is 1, which leaves the values in the ``grid_capacity_col``
        unmodified.
    priority_power : int, optional
        This factor can be used to exaggerate the influence of the values in
        ``grid_priority_col``, such that higher values have an increased likelihood of
        load deployment and lower values have a decreased likelihood. This effect is
        implemented by raising the values in ``grid_priority_col`` to the specified
        ``priority_power``. As a result, if the input  values in ``grid_priority_col``
        are < 1, setting ``priority_power`` to high values can result in completely
        eliminating lower priority sites from consideration. The default value is 1,
        which leaves the values in ``grid_priority_col`` unmodified. To achieve
        less dispersion and greater clustering of downscaled load in higher priority
        sites, increase this value.
    n_bootstraps : int, optional
        Number of bootstraps to simulate in each projection year. Default is 10,000.
        In general, larger values will produce more stable results, with less chance
        for lower priority sites to receive large amounts of deployed load. However,
        larger values will also cause longer run times.
    random_seed : int, optional
        Random seed to use for reproducible bootstrapping. Default is 0. In general,
        this value does not need to be modified. The exception is if you are interested
        in testing sensitivities and/or producing multiple realizations or scenarios of
        deployment for a given set of values in ``load_priority_col``.
    max_workers : int, optional
        Number of workers to use for bootstrapping. By default None, which uses all
        available workers. In general, this value should only be changed if you are
        running into out-of-memory errors.


    Returns
    -------
    pandas.DataFrame
        Returns DataFrame consisting of load projections downscaled to the grid.
        This dataframe will contain all of the columns from the input ``grid_df``,
        as well as three new columns, including ``year`` (indicating the year
        of the projection) and a "new_" and "total_" load column, named with a suffix
        corresponding to the ``load_value_col``.

    Raises
    ------
    ValueError
        A ValueError will be raised if internal consistency checks for downscaled
        results do not pass or if the region names do not match between ``grid_df`` and
        ``load_df``.
    """

    grid_idx = grid_df.index.name
    if grid_idx is None:
        named_index = False
        grid_idx = "index"
    else:
        named_index = True

    grid_df["_region"] = grid_df[grid_region_col].str.lower()
    load_df["_region"] = load_df[load_region_col].str.lower()

    grid_regions = grid_df["_region"][~grid_df["_region"].isna()].unique().tolist()
    load_regions = load_df["_region"].unique().tolist()
    differences = list(set(grid_regions).symmetric_difference(set(load_regions)))
    if len(differences) > 0:
        raise ValueError("Region names do not match between grid_df and load_df.")

    region_results = []
    for region in load_regions:
        LOGGER.info(f"Downscaling load projections for region {region}")
        grid_region_df = grid_df[grid_df["_region"] == region].copy()
        load_region_df = load_df[load_df["_region"] == region].copy()
        region_downscaled_df = downscale_total(
            grid_df=grid_region_df,
            grid_priority_col=grid_priority_col,
            grid_baseline_load_col=grid_baseline_load_col,
            baseline_year=baseline_year,
            grid_capacity_col=grid_capacity_col,
            load_df=load_region_df,
            load_value_col=load_value_col,
            load_year_col=load_year_col,
            max_site_addition_per_year=max_site_addition_per_year,
            site_saturation_limit=site_saturation_limit,
            priority_power=priority_power,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
            max_workers=max_workers,
        )
        region_downscaled_df.drop(columns=["_region"], inplace=True)
        region_results.append(region_downscaled_df)

    # fill in results for grid cells with unknown (NA) region
    na_region_df = grid_df[grid_df["_region"].isna()].reset_index()
    if len(na_region_df) > 0:
        na_region_df.drop(columns=["_region"], inplace=True)
        na_region_df[f"total_{load_value_col}"] = 0.0
        na_region_df[f"new_{load_value_col}"] = 0.0

        years_df = pd.DataFrame({"year": load_df[load_year_col].unique()})
        na_region_downscaled_df = pd.merge(na_region_df, years_df, how="cross")

        region_results.append(na_region_downscaled_df)

    grid_projections_df = pd.concat(region_results, ignore_index=True)
    if not named_index:
        grid_projections_df.drop(columns=grid_idx, inplace=True)

    return grid_projections_df
