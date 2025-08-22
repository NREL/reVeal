# -*- coding: utf-8 -*-
"""
grid module
"""
from pathlib import Path
import warnings

import pyproj
import rasterio
import geopandas as gpd

from exactextract.exact_extract import exact_extract
import pandas as pd
from libpysal import graph
from shapely.geometry import box
import numpy as np

from loci.config import load_characterize_config


# stop to_crs() bugs
pyproj.network.set_network_enabled(active=False)


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

    def neighbors(self, order):
        """
        Create new geometry for each cell in the grid that consists of a union with
        neighboring cells of the specified order.

        Parameters
        ----------
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
            return self.df.copy()

        grid = self.df.copy()

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

    def _get_grid(self, neighbor=False):
        """Get the grid, optionally with neighbor geometries."""
        if neighbor:
            grid = self.neighbors(order=1)
        else:
            grid = self.df.copy()
        return grid

    def _vector_proximity(self, df, grid, stem):
        """Calculate proximity of vector data to grid cells."""
        joined = gpd.sjoin_nearest(
            grid, df, how="left", distance_col=f"proximity_{stem}"
        )
        joined = joined.drop_duplicates(subset="grid_id")
        grid[f"proximity_{stem}"] = joined[f"proximity_{stem}"].values
        return grid

    def _vector_length(self, df, grid, stem):
        """Calculate length of vector data within grid cells."""
        inter = gpd.overlay(
            grid[["grid_id", "geometry"]], df[["geometry"]], how="intersection"
        )
        inter["seg_length"] = inter.geometry.length
        length_series = inter.groupby("grid_id")["seg_length"].sum()
        col_name = f"length_{stem}"
        grid[col_name] = grid["grid_id"].map(length_series).fillna(0)
        return grid

    def _vector_count(self, df, grid, stem):
        """Count occurrences of vector data within grid cells."""
        joined = gpd.sjoin(grid, df, how="left", predicate="intersects")
        counts = joined.groupby("grid_id")["index_right"].count()
        count_col = f"count_{stem}"
        grid[count_col] = grid["grid_id"].map(counts).fillna(0).astype(int)
        return grid

    def _vector_aggregate(self, df, grid, stem, value_col=None, func="sum"):
        """Aggregate vector data within grid cells."""
        joined = gpd.sjoin(grid, df, how="left", predicate="intersects")
        grp = joined.groupby("grid_id")[value_col]
        if func == "sum":
            agg_series = grp.sum()
        elif func in ("mean", "avg"):
            agg_series = grp.mean()
        else:
            raise ValueError(f"Unsupported aggregation function: {func}")

        grid[f"{func}_{stem}_{value_col}"] = grid["grid_id"].map(agg_series)
        return grid

    def _vector_intersects(self, df, grid, stem):
        """Flag each grid cell True/False if it intersects any feature."""
        joined = gpd.sjoin(
            grid[["grid_id", "geometry"]],
            df[["geometry"]],
            how="left",
            predicate="intersects",
        )

        intersects = joined.groupby("grid_id")["index_right"].count()
        col_name = f"intersects_{stem}"
        grid[col_name] = grid["grid_id"].map(intersects).fillna(0).astype(int) > 0
        return grid

    def _aggregate_vector_within_grid(
        self, df_path, value_col=None, agg_func="sum", buffer=None, neighbor=False
    ):
        """Aggregate vector data within grid cells.

        Parameters
        ----------
        df_path : str
            Path to the vector data file.
        value_col : str, optional
            Name of the column to aggregate, by default None
        agg_func : str, optional
            Aggregation function, by default "sum"
            Supported functions are:
            "proximity", "count", "sum", "mean", "avg", "length",
            and "intersects".
        buffer : float, optional
            Buffer distance to apply to grid geometries, by default None
        neighbor : bool, optional
            Whether to include neighboring grid cells, by default False

        Returns
        -------
        gpd.GeoDataFrame
            The updated grid with aggregated values.
        """
        grid = self._get_grid(neighbor)

        if buffer is not None:
            grid["geometry"] = grid.geometry.buffer(buffer)

        df = gpd.read_file(df_path).to_crs(self.crs)
        stem = Path(df_path).stem
        func = agg_func.lower()

        if isinstance(value_col, float) and pd.isna(value_col):
            value_col = None
        if isinstance(value_col, str) and not value_col.strip():
            value_col = None

        if func == "proximity":
            grid = self._vector_proximity(df, grid, stem)
        elif func == "count":
            grid = self._vector_count(df, grid, stem)
        elif func in ("sum", "mean", "avg"):
            grid = self._vector_aggregate(df, grid, stem, value_col, func)
        elif func == "length":
            grid = self._vector_length(df, grid, stem)
        elif func == "intersects":
            grid = self._vector_intersects(df, grid, stem)
        else:
            raise ValueError(f"Unsupported aggregation function: {func}")

        return grid

    def _aggregate_raster_within_grid(
        self, raster_path, agg_func="sum", buffer=None, neighbor=False
    ):
        """Aggregate raster values within grid cells using exactextract.

        Parameters
        ----------
        raster_path : str
            Path to the raster file.
        agg_func : str, optional
            Aggregation function to use, by default "sum"
            Supported functions are: "sum", "mean", "avg", "min", "max".
        buffer : float, optional
            Buffer distance to apply to grid geometries, by default None
        neighbor : bool, optional
            Whether to include neighboring grid cells, by default False

        Returns
        -------
        gpd.GeoDataFrame
            The updated grid with aggregated values.
        """
        grid = self._get_grid(neighbor)

        if buffer is not None:
            grid["geometry"] = grid.geometry.buffer(buffer)

        with rasterio.open(raster_path) as src:
            grid = grid.to_crs(src.crs)

        stem = Path(raster_path).stem
        func = agg_func.lower()

        result = exact_extract(
            rast=raster_path,
            vec=grid,
            include_cols=["grid_id"],
            ops=[func],
            output="pandas",
        )

        col_name = f"{func}_{stem}"
        result.rename(columns={f"{func}": col_name}, inplace=True)

        return result

    def aggregate_within_grid(
        self, df_path, value_col=None, agg_func="sum", buffer=None, neighbor=False
    ):
        """Aggregate data within grid cells.

        Parameters
        ----------
        df_path : str
            Path to the vector data file.
        value_col : str, optional
            Name of the column to aggregate, by default None
        agg_func : str, optional
            Aggregation function to use, by default "sum"
        buffer : float, optional
            Buffer distance to apply to grid geometries, by default None

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with aggregated values.
        """
        ext = Path(df_path).suffix.lower()
        if ext in [".tif", ".tiff"]:
            grid = self._aggregate_raster_within_grid(
                df_path, agg_func, buffer, neighbor
            )
        elif ext in [".gpkg", ".geojson", ".shp"]:
            grid = self._aggregate_vector_within_grid(
                df_path, value_col, agg_func, buffer, neighbor
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return grid


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
        for attr_name, char_info in self.config.characterizations.items():
            pass

    # def run_method(self, neighbor_order=0, buffer_distance=0):
    #     grid_df = self.neighbors(neighbor_order)
    #     if buffer_distance > 0:
    #         grid_df["geometry"] = grid_df["geometry"].buffer(buffer_distance)

    #     pass
    #     TODO: start again here
    #     layer = self.aggregate_within_grid(
    #         char_info.dset_src,
    #         value_col=val_col,
    #         agg_func=dset.method,
    #         buffer=dset.buffer_distance,
    #         neighbor=dset.neighbor_order,
    #     )

    #     merge_cols = [
    #         c for c in layer.columns if c not in ("geometry", "grid_id")
    #     ]
    #     out_grid = out_grid.merge(
    #         layer[["grid_id"] + merge_cols], on="grid_id", how="left"
    #     )

    # return out_grid
