# -*- coding: utf-8 -*-
"""
overlay module

Note that to expose methods here for by loci.grid.get_overlay_method() function
and functions dependent on it, the function must be prefixed with "calc_".
"""
import geopandas as gpd

# import rasterio
# from exactextract.exact_extract import exact_extract

from loci.fileio import read_vectors


def calc_feature_count(zones_df, dset_src, **kwargs):
    """
    Calculate the count of features intersecting each zone in input zones dataframe.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which feature counts will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset to be counted. Must be in the same CRS as
        the zones_df.
    **kwargs :
        Arbitrary keyword arguments. Note that none of these are used, but this
        allows passing an arbitrary dictionary that includes both used and unused
        parameters as input to the function.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the count
        of features in each zone. The index from the input zones_df is also included.
    """
    # pylint: disable=unused-argument

    features_df = read_vectors(dset_src)
    features_df["feature_count"] = 1
    join_df = gpd.sjoin(
        zones_df[["geometry"]],
        features_df[["geometry", "feature_count"]],
        how="left",
        predicate="intersects",
    )
    counts_df = join_df.groupby(by=join_df.index)[["feature_count"]].sum()
    counts_df["feature_count"] = counts_df["feature_count"].fillna(0).astype(int)
    counts_df.rename(columns={"feature_count": "value"}, inplace=True)

    return counts_df


# def _vector_length(self, df, grid, stem):
#     """Calculate length of vector data within grid cells."""
#     inter = gpd.overlay(
#         grid[["grid_id", "geometry"]], df[["geometry"]], how="intersection"
#     )
#     inter["seg_length"] = inter.geometry.length
#     length_series = inter.groupby("grid_id")["seg_length"].sum()
#     col_name = f"length_{stem}"
#     grid[col_name] = grid["grid_id"].map(length_series).fillna(0)
#     return grid

# def _vector_aggregate(self, df, grid, stem, value_col=None, func="sum"):
#     """Aggregate vector data within grid cells."""
#     joined = gpd.sjoin(grid, df, how="left", predicate="intersects")
#     grp = joined.groupby("grid_id")[value_col]
#     if func == "sum":
#         agg_series = grp.sum()
#     elif func in ("mean", "avg"):
#         agg_series = grp.mean()
#     else:
#         raise ValueError(f"Unsupported aggregation function: {func}")

#     grid[f"{func}_{stem}_{value_col}"] = grid["grid_id"].map(agg_series)
#     return grid


# def _aggregate_raster_within_grid(
#     self, raster_path, agg_func="sum", buffer=None, neighbor=False
# ):
#     """Aggregate raster values within grid cells using exactextract.

#     Parameters
#     ----------
#     raster_path : str
#         Path to the raster file.
#     agg_func : str, optional
#         Aggregation function to use, by default "sum"
#         Supported functions are: "sum", "mean", "avg", "min", "max".
#     buffer : float, optional
#         Buffer distance to apply to grid geometries, by default None
#     neighbor : bool, optional
#         Whether to include neighboring grid cells, by default False

#     Returns
#     -------
#     gpd.GeoDataFrame
#         The updated grid with aggregated values.
#     """
#     grid = self._get_grid(neighbor)

#     if buffer is not None:
#         grid["geometry"] = grid.geometry.buffer(buffer)

#     with rasterio.open(raster_path) as src:
#         grid = grid.to_crs(src.crs)

#     stem = Path(raster_path).stem
#     func = agg_func.lower()

#     result = exact_extract(
#         rast=raster_path,
#         vec=grid,
#         include_cols=["grid_id"],
#         ops=[func],
#         output="pandas",
#     )

#     col_name = f"{func}_{stem}"
#     result.rename(columns={f"{func}": col_name}, inplace=True)

#     return result
