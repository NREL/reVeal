# -*- coding: utf-8 -*-
"""
grid module
"""
import warnings
from inspect import getmembers, isfunction
import re
import logging
from math import isclose

import pandas as pd
import geopandas as gpd
from libpysal import graph
import numpy as np
from shapely.geometry import box

from reVeal.config.config import load_config, BaseGridConfig
from reVeal.config.characterize import CharacterizeConfig
from reVeal.config.normalize import NormalizeConfig, GRID_IDX
from reVeal.config.score_weighted import ScoreWeightedConfig
from reVeal.config.downscale import TotalDownscaleConfig, RegionalDownscaleConfig
from reVeal import overlay, normalization, load

OVERLAY_METHODS = {
    k[5:]: v for k, v in getmembers(overlay, isfunction) if k.startswith("calc_")
}
NORMALIZE_METHODS = {
    k[5:]: v for k, v in getmembers(normalization, isfunction) if k.startswith("calc_")
}

LOGGER = logging.getLogger(__name__)


def create_grid(res, xmin, ymin, xmax, ymax, crs):
    """
    Create a regularly spaced grid at the specified resolution covering the
    specified bounds.

    Parameters
    ----------
    res : float
        Resolution of the grid (i.e., size of each grid cell along one dimension)
        measured in units of the specified CRS.
    xmin : float
        Minimum x coordinate of bounding box.
    ymin : float
        Minimum y coordinate of bounding box.
    xmax : float
        Maximum x coordinate of bounding box.
    ymax : float
        Maximum y coordinate of bounding box.
    crs : str
        Coordinate reference system (CRS) of grid_resolution and bounds. Will also
        be assigned to the returned GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the resulting grid.
    """

    grid_df = gpd.GeoDataFrame(
        geometry=[
            box(x, y, x + res, y + res)
            for x in np.arange(xmin, xmax, res)
            for y in np.arange(ymin, ymax, res)
        ],
        crs=crs,
    )
    grid_df["grid_id"] = grid_df.index

    return grid_df


def get_neighbors(grid_df, order):
    """
    Create new geometry for each cell in the input grid, consisting of a union with
    neighboring cells of the specified contiguity order.

    Parameters
    ----------
    grid_df : geopandas.GeoDataFrame
        Input grid geodataframe. This should be a polygon geodataframe where all
        geometries form a coverage (i.e., a non-overlapping mesh) and neighboring
        geometries share only points or segments of the exterior boundaries. This
        function also assumes that the index of zones_df is unique for each feature. If
        either of these are not the case, unexpected results may occur.
    order : int
        Neighbor order to apply. For example, order=1 will group all first-order
        queen's contiguity neighbors into a new grid cell, labeled based on the
        center grid cell.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the grid transformed into larger cells based on
        neighbors.
    """
    if order == 0:
        return grid_df.copy()

    grid = grid_df.copy()

    # build contiguity matrix
    cont = graph.Graph.build_contiguity(grid, rook=False)
    if order > 1:
        cont = cont.higher_order(k=order, lower_order=True)

    # create a "complete" adjacency lookup, that includes center cells
    adjacent_df = cont.adjacency.reset_index()
    centers_df = pd.DataFrame({"focal": grid.index, "neighbor": grid.index})
    combined_df = pd.concat(
        [centers_df, adjacent_df[["focal", "neighbor"]]], ignore_index=True
    )

    # join in geometries and dissolve into groups
    combined_df.rename(columns={"neighbor": "join_id"}, inplace=True)
    grid["join_id"] = grid.index
    combined_gdf = grid.merge(combined_df, how="left", on="join_id")
    dissolved_df = combined_gdf[["focal", "geometry"]].dissolve(
        by="focal", as_index=True
    )

    # overwrite geometries in original grid with dissolved geometries
    grid.loc[dissolved_df.index, ["geometry"]] = dissolved_df["geometry"]
    grid.drop(columns=["join_id"], inplace=True)

    return grid


def get_method_from_members(method_name, members):
    """
    Helper function to look up a callable from a group of options.

    Parameters
    ----------
    method_name : str
        Name of method to retrieve. Should use spaces where the desired callable uses
        underscores.
    member : dict
        Dictionary where keys indicate method names and values are the corresponding
        callable function. Names used as keys should use underscores where the
        method_name uses spaces.

    Returns
    -------
    Callable
        Method as a function.

    Raises
    ------
    NotImplementedError
        A NotImplementedError will be raised if a function cannot be found
        corresponding to the input method_name.
    """
    pattern = r"[\W\s]+"
    # Replace all matches of the pattern with a single underscore
    sanitized_method = re.sub(pattern, "_", method_name).strip("_").lower()

    method = members.get(sanitized_method)
    if not method:
        raise NotImplementedError(f"Unrecognized or unsupported method: {method_name}")

    return method


def run_characterization(df, characterization):
    """
    Execute a single characterization on an input grid.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input grid geodataframe. This should be a polygon geodataframe where all
        geometries form a coverage (i.e., a non-overlapping mesh) and neighboring
        geometries share only points or segments of the exterior boundaries. This
        function also assumes that the index of df is unique for each feature. If
        either of these are not the case, unexpected results may occur.
    characterization : :class:`reVeal.config.characterize.Characterization`
        Input information describing characterization to be run, in the form of
        a Characterization instance.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the output
        values from the characterization for each zone. The index from the input df
        is also included.
    """
    grid_df = get_neighbors(df, characterization.neighbor_order)
    if characterization.buffer_distance > 0:
        grid_df["geometry"] = grid_df["geometry"].buffer(
            characterization.buffer_distance
        )

    method = get_method_from_members(characterization.method, OVERLAY_METHODS)
    result_df = method(grid_df, **characterization.model_dump())

    return result_df


def run_normalization(df, attribute, normalize_method, invert):
    """
    Execute a single characterization on an input grid.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input grid geodataframe
    attribute : str
        Name of column in input GeoDataFrame to normalize
    normalize_method : str
        Method to use for normalizing attribute.
    invert : bool,optional
        If True, normalize with values inverted (i.e., low values will be closer to 1,
        and higher values closer to 0). Default is False, under which values are
        normalized with low values closer to 0 and high values closer to 1.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the output
        normalized values for the specified attribute. The index from the input df
        is also included.
    """
    method = get_method_from_members(normalize_method, NORMALIZE_METHODS)
    normalized = method(df, attribute, invert)

    return normalized


def run_weighted_scoring(df, attributes):
    """
    Calculated weighted score for the input dataframe based on the specified attributes
    and corresponding weights.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input grid geodataframe
    attributes : List[reVeal.config.score_weighted.Attribute]
        List of Attribute models specifying the attribute columns and
        corresponding weights to use.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the output
        weighted score derived from the specified inputs. The index from the input df
        is also included.

    Raises
    ------
    ValueError
        A ValueError will be raised if the input attribute weights do not
        sum to 1.
    """

    cols = [a.attribute for a in attributes]
    weights = np.array([a.weight for a in attributes])

    sum_weights = weights.sum()

    if not isclose(sum_weights, 1, abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError(
            "Weights of input attributes must sum to 1. "
            f"Sum of input weights is: {sum_weights}."
        )

    scores = (df[cols] * weights).sum(axis=1)
    scores.name = "value"

    scores_df = scores.to_frame()

    return scores_df


class BaseGrid:
    """
    Grid base class
    """

    def __init__(self, res=None, bounds=None, crs=None, template=None):
        """
        Initialize a Grid instance from a template or input parameters.

        Parameters
        ----------
        res : float
            Resolution of the grid (i.e., size of each grid cell along one dimension)
            measured in units of the specified CRS. Required if template=None.
            Ignored if template is provided. Default is None.
        crs : str
            Coordinate reference system (CRS) for the grid. Required if template=None.
            If template is provided, the grid will be reprojected to this CRS. Default
            is None.
        bounds : tuple, optional
            The spatial bounds for the grid in the format [xmin, ymin, xmax, ymax],
            in units of crs (or the template CRS). Required if template=None.
            If template is provided, the grid will be subset to the cells intersecting
            the specified bounds. Default is None.
        template : str, optional
            Path to a template file for the grid. Input template should be a vector
            polygon dataset. Default is None.
        """
        if not template:
            if res is None or crs is None or bounds is None:
                raise ValueError(
                    "If template is not provided, grid_size, crs, and bounds must be "
                    "specified."
                )
            self.df = create_grid(res, *bounds, crs)
        else:
            if res is not None:
                warnings.warn(
                    "res specified but template provided. res will be ignored."
                )

            grid = gpd.read_file(template)
            if crs:
                grid.to_crs(crs, inplace=True)
            if bounds:
                bounds_box = box(*bounds)
                self.df = grid[grid.intersects(bounds_box)].copy()
            else:
                self.df = grid

        self.crs = self.df.crs
        self._add_index()

    def _add_index(self):
        """
        Adds gid column to self.df and sets as index.
        """
        if GRID_IDX in self.df.columns:
            warnings.warn(
                f"{GRID_IDX} column already exists in self.dataframe. Values will be "
                "overwritten."
            )
        self.df[GRID_IDX] = range(0, len(self.df))
        self.df.set_index(GRID_IDX, inplace=True)


class RunnableGrid(BaseGrid):
    """
    Subclass of BaseGrid for running operations.
    """

    CONFIG_CLASS = BaseGridConfig

    def __init__(self, config):
        """
        Initialize grid from configuration.

        Parameters
        ----------
        config : [dict, CharacterizeConfig]
            Input configuration as either a dictionary or a CharacterizationConfig
            instance. If a dictionary, validation will be performed to ensure
            inputs are valid.
        """
        config = load_config(config, config_class=self.CONFIG_CLASS)
        super().__init__(template=config.grid)
        self.config = config

    def run(self):
        """
        Place holder for run method, to be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            A NotImplementedError is always raised since this is just a template
            for subclasses.
        """
        raise NotImplementedError(
            "run method not implemented for RunnableGrid base class"
        )


class CharacterizeGrid(RunnableGrid):
    """
    Subclass of RunnableGrid for running characterizations.
    """

    CONFIG_CLASS = CharacterizeConfig

    def run(self):
        """
        Run grid characterization based on the input configuration.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the characterized grid.
        """
        results = []
        for attr_name, char_info in self.config.characterizations.items():
            LOGGER.info(f"Running characterization for output column '{attr_name}'")
            try:
                char_df = run_characterization(self.df, char_info)
                char_df.rename(columns={"value": attr_name}, inplace=True)
                results.append(char_df)
            except NotImplementedError:
                warnings.warn(f"Method {char_info.method} not supported")

        results_df = pd.concat([self.df] + results, axis=1)

        for attr_name, expression in self.config.expressions.items():
            LOGGER.info(f"Running expression for output column '{attr_name}'")
            try:
                results_df[attr_name] = results_df.eval(
                    expression, local_dict={}, global_dict={}
                )
            except pd.errors.UndefinedVariableError as e:
                warnings.warn(f"Unable to derive output values for {attr_name}: {e}")

        LOGGER.info("Checking for NA values in results dataframe.")
        na_check = results_df.isna().any()
        if na_check.any():
            cols_with_nas = na_check.keys()[na_check.values].tolist()
            warnings.warn(
                "NAs encountered in results dataframe in the following columns: "
                f"{cols_with_nas}"
            )

        return results_df


class NormalizeGrid(RunnableGrid):
    """
    Subclass of RunnableGrid for normalizing attribute values.
    """

    CONFIG_CLASS = NormalizeConfig

    def run(self):
        """
        Run attribute normalization based on the input configuration.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with normalized attributes.
        """
        results = []
        for attr_name, attr_info in self.config.attributes.items():
            LOGGER.info(f"Running normalization for output column '{attr_name}'")
            try:
                norm_df = run_normalization(
                    self.df,
                    attr_info.attribute,
                    attr_info.normalize_method,
                    attr_info.invert,
                )
                norm_df.rename(columns={"value": attr_name}, inplace=True)
                results.append(norm_df)
            except NotImplementedError:
                warnings.warn(f"Method {attr_info.normalize_method} not supported")

        keep_cols = [
            c for c in self.df.columns if c not in self.config.attributes.keys()
        ]
        results_df = pd.concat([self.df[keep_cols]] + results, axis=1)

        LOGGER.info("Checking for NA values in results dataframe.")
        na_check = results_df[self.config.attributes.keys()].isna().any()
        if na_check.any():
            cols_with_nas = na_check.keys()[na_check.values].tolist()
            warnings.warn(
                "NAs encountered in results dataframe in the following columns: "
                f"{cols_with_nas}"
            )

        return results_df


class ScoreWeightedGrid(RunnableGrid):
    """
    Subclass of RunnableGrid for calculating weighted composite score.
    """

    CONFIG_CLASS = ScoreWeightedConfig

    def run(self):
        """
        Run weighted composite scoring based on the input configuration.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with scored attributes.
        """
        score_col = self.config.score_name
        scores_df = run_weighted_scoring(self.df, self.config.attributes)

        results_df = self.df.copy()
        results_df[score_col] = scores_df["value"]

        na_check = results_df[score_col].isna().any()
        if na_check:
            warnings.warn(
                f"NAs encountered in output weighted score column {score_col}"
            )

        return results_df


class TotalDownscaleGrid(RunnableGrid):
    """
    Subclass of RunnableGrid for downscaling aggregate load projections to sites in
    grid.
    """

    CONFIG_CLASS = TotalDownscaleConfig

    def run(self):
        """
        Run load downscaling based on the input configuration.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with downscaled load values attributes.
        """

        LOGGER.info("Loading load projections")
        load_df = pd.read_csv(self.config.load_projections)

        LOGGER.info("Downscaling regional load projections to grid")
        downscaled_df = load.downscale_total(
            grid_df=self.df,
            grid_priority_col=self.config.grid_priority,
            grid_baseline_load_col=self.config.grid_baseline_load,
            baseline_year=self.config.baseline_year,
            grid_capacity_col=self.config.grid_capacity,
            load_df=load_df,
            load_value_col=self.config.load_value,
            load_year_col=self.config.load_year,
            max_site_addition_per_year=self.config.max_site_addition_per_year,
            site_saturation_limit=self.config.site_saturation_limit,
            priority_power=self.config.priority_power,
            n_bootstraps=self.config.n_bootstraps,
            random_seed=self.config.random_seed,
            hide_pbar=True,
        )

        return downscaled_df


class RegionalDownscaleGrid(RunnableGrid):
    """
    Subclass of RunnableGrid for downscaling regional load projections to sites in
    grid.
    """

    CONFIG_CLASS = RegionalDownscaleConfig

    def run(self):
        """
        Run load downscaling based on the input configuration.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with downscaled load values attributes.
        """

        LOGGER.info("Loading load projections.")
        load_df = pd.read_csv(self.config.load_projections)

        LOGGER.info("Assigning grid cells to regions")
        regions_lkup_df = overlay.calc_area_weighted_majority(
            self.df, self.config.regions, self.config.region_names
        )
        self.df = pd.concat([self.df, regions_lkup_df], axis=1)

        if self.config.region_weights:
            LOGGER.info("Apportioning load to regions based on weights")
            regional_load_df = load.apportion_load_to_regions(
                load_df,
                self.config.load_value,
                self.config.load_year,
                self.config.region_weights,
            )
            self.config.load_regions = "region"
        else:
            regional_load_df = load_df

        LOGGER.info("Downscaling aggregate load to grid")
        downscaled_df = load.downscale_regional(
            grid_df=self.df,
            grid_priority_col=self.config.grid_priority,
            grid_baseline_load_col=self.config.grid_baseline_load,
            baseline_year=self.config.baseline_year,
            grid_capacity_col=self.config.grid_capacity,
            grid_region_col=self.config.region_names,
            load_df=regional_load_df,
            load_value_col=self.config.load_value,
            load_year_col=self.config.load_year,
            load_region_col=self.config.load_regions,
            max_site_addition_per_year=self.config.max_site_addition_per_year,
            site_saturation_limit=self.config.site_saturation_limit,
            priority_power=self.config.priority_power,
            n_bootstraps=self.config.n_bootstraps,
            random_seed=self.config.random_seed,
            hide_pbar=True,
        )

        return downscaled_df
