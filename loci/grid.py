# -*- coding: utf-8 -*-
"""
grid module
"""
import warnings
from inspect import getmembers, isfunction
import re
import logging

import pandas as pd
import geopandas as gpd
from libpysal import graph
import numpy as np
from shapely.geometry import box

from loci.config import load_characterize_config
from loci import overlay

OVERLAY_METHODS = {
    k[5:]: v for k, v in getmembers(overlay, isfunction) if k.startswith("calc_")
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


def get_overlay_method(method_name):
    """
    Get and return the function corresponding to the input overlay method name.

    Parameters
    ----------
    method_name : str
        Name of overlay method to retrieve.

    Returns
    -------
    Callable
        Overlay method as a function.

    Raises
    ------
    NotImplementedError
        A NotImplementedError will be raised if a function cannot be found
        corresponding to the specified method_name.
    """
    pattern = r"[\W\s]+"
    # Replace all matches of the pattern with a single underscore
    sanitized_method = re.sub(pattern, "_", method_name).strip("_").lower()

    method = OVERLAY_METHODS.get(sanitized_method)
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
    characterization : :class:`loci.config.Characterization`
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

    method = get_overlay_method(characterization.method)
    result_df = method(grid_df, **characterization.model_dump())

    return result_df


class Grid:
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
        self._add_gid()

    def _add_gid(self):
        """
        Adds gid column to self.df and sets as index.
        """
        if "gid" in self.df.columns:
            warnings.warn(
                "gid column already exists in self.dataframe. Values will be "
                "overwritten."
            )
        self.df["gid"] = range(0, len(self.df))
        self.df.set_index("gid", inplace=True)


class CharacterizeGrid(Grid):
    """
    Subclass of Grid for running characterizations.
    """

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
        config = load_characterize_config(config)
        super().__init__(template=config.grid)
        self.config = config

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
                results_df[attr_name] = results_df.eval(expression)
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
