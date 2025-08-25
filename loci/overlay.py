# -*- coding: utf-8 -*-
"""
overlay module

Note that to expose methods here for by loci.grid.get_overlay_method() function
and functions dependent on it, the function must be prefixed with "calc_".
"""
import geopandas as gpd
import pandas as pd

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


def calc_sum_attribute(zones_df, dset_src, attribute, **kwargs):
    """
    Calculate the sum of the specified attribute for all features intersecting each
    zone in input zones dataframe.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which feature counts will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset with attribute to be summed. Must be in the same
        CRS as the zones_df.
    attribute : str
        Name of attribute in dset_src to sum.
    **kwargs :
        Arbitrary keyword arguments. Note that none of these are used, but this
        allows passing an arbitrary dictionary that includes both used and unused
        parameters as input to the function.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the sum
        of the attribute of features in each zone. The index from the input zones_df is
        also included.
    """
    # pylint: disable=unused-argument

    features_df = read_vectors(dset_src)
    if attribute not in features_df.columns:
        raise KeyError(f"attribute {attribute} not a column in {dset_src}")

    if not pd.api.types.is_numeric_dtype(features_df[attribute]):
        raise TypeError(f"attribute {attribute} in {dset_src} must be numeric")

    join_df = gpd.sjoin(
        zones_df[["geometry"]],
        features_df[["geometry", attribute]],
        how="left",
        predicate="intersects",
    )
    sums_df = join_df.groupby(by=join_df.index)[[attribute]].sum()
    sums_df[attribute] = sums_df[attribute].fillna(0).astype(sums_df[attribute].dtype)
    sums_df.rename(columns={attribute: "value"}, inplace=True)

    return sums_df


def calc_sum_length(zones_df, dset_src, **kwargs):
    """
    Calculate the sum of lengths of input features intersecting each zone in input
    zones dataframe.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which feature counts will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset with geometries whose lengths will be summed.
        Expected to a be a LineString or MultiLineString input, though this is not
        checked. Results for Points/MultiPoints will be returned as all zeros, and
        results for Polygons/MultiPolygons will represent perimeters. Must be in the
        same CRS as the zones_df.
    **kwargs :
        Arbitrary keyword arguments. Not used, but allows passing extra parameters.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the sum of
        lengths of features in each zone. The index from the input zones_df is also
        included.
    """
    # pylint: disable=unused-argument

    features_df = read_vectors(dset_src)

    zone_idx = zones_df.index.name

    intersection_df = gpd.overlay(
        features_df[["geometry"]],
        zones_df.reset_index()[[zone_idx, "geometry"]],
        how="intersection",
        keep_geom_type=True,
        make_valid=True,
    )

    intersection_df["value"] = intersection_df.length
    sums_df = intersection_df.groupby(by=zone_idx)[["value"]].sum()

    complete_sums_df = sums_df.reindex(zones_df.index, fill_value=0)

    return complete_sums_df


def calc_sum_attribute_length(zones_df, dset_src, attribute, **kwargs):
    """
    Calculate the sum of attribute-length of input features intersecting each zone in
    input zones dataframe. Attribute-length is defined as the product of the length
    of each feature in each zone and the specified attribute value
    (i.e., length * attribute).

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which feature counts will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset with geometries whose attribute-lengths will be
        summed. Expected to a be a LineString or MultiLineString input, though this is
        not checked. Results for Points/MultiPoints will be returned as all zeros, and
        results for Polygons/MultiPolygons will use perimeters instead of lengths. Must
        be in the same CRS as the zones_df.
    attribute : str
        Name of attribute in dset_src to use for calculating attribute-lengths.
    **kwargs :
        Arbitrary keyword arguments. Not used, but allows passing extra parameters.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the sum of
        attribute-lengths of features in each zone. The index from the input zones_df
        is also included.
    """
    # pylint: disable=unused-argument

    features_df = read_vectors(dset_src)

    if attribute not in features_df.columns:
        raise KeyError(f"attribute {attribute} not a column in {dset_src}")

    if not pd.api.types.is_numeric_dtype(features_df[attribute]):
        raise TypeError(f"attribute {attribute} in {dset_src} must be numeric")

    zone_idx = zones_df.index.name

    intersection_df = gpd.overlay(
        features_df[["geometry", attribute]],
        zones_df.reset_index()[[zone_idx, "geometry"]],
        how="intersection",
        keep_geom_type=True,
        make_valid=True,
    )

    intersection_df["value"] = intersection_df.length * intersection_df[attribute]
    sums_df = intersection_df.groupby(by=zone_idx)[["value"]].sum()

    complete_sums_df = sums_df.reindex(zones_df.index, fill_value=0)

    return complete_sums_df


def calc_sum_area(zones_df, dset_src, **kwargs):
    """
    Calculate the sum of combined areas of input features intersecting each zone in
    input zones dataframe. Intersecting features are unioned before calculating areas,
    such that the total area cannot exceed the size of the zone.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which feature counts will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset with geometries whose areas will be summed.
        Expected to a be a Polygon or MultiPolygon input, though this is not
        checked. Results for Points/MultiPoints and LineStrings/MultiLineStrings will
        be returned as all zeros. Must be in the same CRS as the zones_df.
    **kwargs :
        Arbitrary keyword arguments. Not used, but allows passing extra parameters.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the sum of
        combined area of features in each zone. The index from the input zones_df is
        also included.
    """
    # pylint: disable=unused-argument

    features_df = read_vectors(dset_src)

    zone_idx = zones_df.index.name

    intersection_df = gpd.overlay(
        features_df[["geometry"]],
        zones_df.reset_index()[[zone_idx, "geometry"]],
        how="intersection",
        keep_geom_type=True,
        make_valid=True,
    )

    dissolved_df = intersection_df.dissolve(by=zone_idx, method="unary", as_index=True)

    dissolved_df["value"] = dissolved_df.area
    dissolved_df.drop(columns="geometry", inplace=True)

    areas_df = dissolved_df.reindex(zones_df.index, fill_value=0)

    return areas_df


# "sum area",
# "percent covered",

# "area-weighted attribute average",
# "area-apportioned attribute sum",
# "mean",
# "median",
# "sum",
# "area",

# Older code, kept temporarily for reference
# import rasterio
# from exactextract.exact_extract import exact_extract
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
