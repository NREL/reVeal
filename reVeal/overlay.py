# -*- coding: utf-8 -*-
"""
overlay module

Note that to expose methods here for by reVeal.grid.get_overlay_method() function
and functions dependent on it, the function must be prefixed with "calc_".
"""
# pylint: disable=unused-argument
import geopandas as gpd
import pandas as pd
from exactextract.exact_extract import exact_extract
from osgeo.gdal import UseExceptions
import rasterio

from reVeal.fileio import read_vectors

UseExceptions()


def calc_feature_count(zones_df, dset_src, **kwargs):
    """
    Calculate the count of features intersecting each zone in input zones dataframe.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset to be counted. Must be in the same CRS as
        the zones_df.
    **kwargs
        Arbitrary keyword arguments. Note that none of these are used, but this
        allows passing an arbitrary dictionary that includes both used and unused
        parameters as input to the function.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the count
        of features in each zone. The index from the input zones_df is also included.
    """

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
    zone in input zones dataframe. If no features intersect a given zone, a value of
    zero will be returned for that zone.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
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
        Input zones dataframe, to which results will be aggregated. This
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
        Input zones dataframe, to which results will be aggregated. This
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
        Input zones dataframe, to which results will be aggregated. This function
        assumes that the index of zones_df is unique for each feature. If this is not
        the case, unexpected results may occur.
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


def calc_percent_covered(zones_df, dset_src, **kwargs):
    """
    Calculate the percent of each zone covered by the union of the intersecting.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset with geometries to be included in calculating
        coverage percents. Expected to a be a Polygon or MultiPolygon input, though
        this is not checked. Results for Points/MultiPoints and
        LineStrings/MultiLineStrings will be returned as all zeros. Must be in the same
        CRS as the zones_df.
    **kwargs :
        Arbitrary keyword arguments. Not used, but allows passing extra parameters.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the percent
        of each zone coverd by input features. The index from the input zones_df is
        also included.
    """

    feature_areas_df = calc_sum_area(zones_df, dset_src)
    feature_areas_df.rename(columns={"value": "feature_area"}, inplace=True)

    zone_areas_df = pd.DataFrame(zones_df.area, columns=["zone_area"])
    all_areas_df = zone_areas_df.merge(
        feature_areas_df, how="left", left_index=True, right_index=True
    )
    all_areas_df["value"] = (
        all_areas_df["feature_area"] / all_areas_df["zone_area"] * 100
    )

    return all_areas_df[["value"]]


def calc_area_weighted_average(zones_df, dset_src, attribute, **kwargs):
    """
    Calculate the area-weighted average of the specified attribute for input features
    intersecting each zone in input zones dataframe. Area-weighted average is defined
    as the sum of all intersecting features' area x attribute value divided by the sum
    of all areas.

    Does not attribute any value to areas where features are not present -- i.e.,
    if only a small portion of a zone is covered by features, but those features have
    high attribute values, the result will have a high attribute value. In cases
    where this matters, use this function in conjunction with calc_percent_covered()
    or calc_sum_area().

    If no features intersect a given zone, a value of NA will be returned for that
    zone.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset with geometries to be included in calculating
        coverage percents. Expected to a be a Polygon or MultiPolygon input, though
        this is not checked. Results for Points/MultiPoints and
        LineStrings/MultiLineStrings will be returned as all NAs since those features
        have zero area. Must be in the same CRS as the zones_df.
    attribute : str
        Name of attribute in dset_src to use for calculating area-weighted average.
    **kwargs :
        Arbitrary keyword arguments. Not used, but allows passing extra parameters.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the
        area-weighted average of attribute values of features in each zone. The index
        from the input zones_df is also included.
    """

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

    intersection_df["area"] = intersection_df.area
    intersection_df["product"] = intersection_df.area * intersection_df[attribute]
    avg_df = intersection_df.groupby(by=zone_idx)[["product", "area"]].sum()
    avg_df["value"] = avg_df["product"] / avg_df["area"]

    complete_avg_df = avg_df[["value"]].reindex(zones_df.index)

    return complete_avg_df


def calc_area_apportioned_sum(zones_df, dset_src, attribute, **kwargs):
    """
    Calculate the area-apportioned sum of the specified attribute for input features
    intersecting each zone in input zones dataframe. Area-apportioning for each feature
    works by determining the proportion of each feature intersecting a given zone,
    as a function of its intersecting area divided by the its total area. The attribute
    value of the feature is then multiplied by this proportion to "apportion" the value
    to the zone. The apportioned values for all features intersecting each zone are
    then summed.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input vector dataset with geometries to be included in calculating
        coverage percents. Expected to a be a Polygon or MultiPolygon input, though
        this is not checked. Results for Points/MultiPoints and
        LineStrings/MultiLineStrings will be returned as all zeros since those features
        have zero area. Must be in the same CRS as the zones_df.
    attribute : str
        Name of attribute in dset_src to use for calculating area-weighted average.
    **kwargs :
        Arbitrary keyword arguments. Not used, but allows passing extra parameters.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the
        area-apportioned sum of attribute values of features in each zone. The index
        from the input zones_df is also included.
    """

    features_df = read_vectors(dset_src)

    if attribute not in features_df.columns:
        raise KeyError(f"attribute {attribute} not a column in {dset_src}")

    if not pd.api.types.is_numeric_dtype(features_df[attribute]):
        raise TypeError(f"attribute {attribute} in {dset_src} must be numeric")

    zone_idx = zones_df.index.name

    features_df["total_area"] = features_df.area

    intersection_df = gpd.overlay(
        features_df[["geometry", attribute, "total_area"]],
        zones_df.reset_index()[[zone_idx, "geometry"]],
        how="intersection",
        keep_geom_type=True,
        make_valid=True,
    )

    intersection_df["area"] = intersection_df.area
    intersection_df["proportion"] = (
        intersection_df["area"] / intersection_df["total_area"]
    )
    intersection_df["value"] = (
        intersection_df[attribute] * intersection_df["proportion"]
    )
    sums_df = intersection_df.groupby(by=zone_idx)[["value"]].sum()

    complete_sums_df = sums_df[["value"]].reindex(zones_df.index, fill_value=0)

    return complete_sums_df


def zonal_statistic(zones_df, dset_src, stat, weights_dset_src=None, **kwargs):
    """
    Calculate zonal statistic for the specified statistic.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input raster dataset to be summarized.
    stat : str
        Zonal statistic to calculate. For valid options and compatability with
        use of weights, see:
        https://isciences.github.io/exactextract/operations.html#built-in-operations.
    weights_dset_src : str, optional
        Optional path to datset to use for weights. Note that only some options for
        stat support use of weights. See stat for more information.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the
        aggregate statistic of raster values within each zone. The index from the
        input zones_df is also included.
    """

    zone_idx = zones_df.index.name
    if weights_dset_src is not None:
        stat = f"weighted_{stat}"

    stats_df = exact_extract(
        rast=dset_src,
        vec=zones_df.reset_index(),
        ops=[stat],
        weights=weights_dset_src,
        include_cols=[zone_idx],
        output="pandas",
    )
    stats_df.set_index(zone_idx, inplace=True)
    stats_df.rename(columns={stat: "value"}, inplace=True)

    return stats_df


def calc_median(zones_df, dset_src, **kwargs):
    """
    Calculate zonal median of raster values over the input zones.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input raster dataset to be summarized.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the
        median raster value within each zone. The index from the input zones_df is also
        included.
    """

    return zonal_statistic(zones_df, dset_src, stat="median")


def calc_mean(zones_df, dset_src, weights_dset_src, **kwargs):
    """
    Calculate zonal mean or weighted mean of raster values over the input zones.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input raster dataset to be summarized.
    weights_dset_src : str, optional
        Optional path to datset to use for weights. If specified, the mean for each
        zone will be weighted based on the values in this dataset.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the
        mean or weighted mean raster value within each zone. The index from the input
        zones_df is also included.
    """

    return zonal_statistic(
        zones_df, dset_src, stat="mean", weights_dset_src=weights_dset_src
    )


def calc_sum(zones_df, dset_src, weights_dset_src, **kwargs):
    """
    Calculate zonal sum or weighted sum of raster values over the input zones.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input raster dataset to be summarized.
    weights_dset_src : str, optional
        Optional path to datset to use for weights. If specified, the sum for each
        zone will be weighted based on the values in this dataset.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the
        sum or weighted sum of raster values within each zone. The index from the input
        zones_df is also included.
    """

    return zonal_statistic(
        zones_df, dset_src, stat="sum", weights_dset_src=weights_dset_src
    )


def calc_area(zones_df, dset_src, **kwargs):
    """
    Calculate the area of a raster within each zone. This function works by summing
    the raster values in each zone and then multiplying by the pixel size. See
    assumptions under dset_src.

    Parameters
    ----------
    zones_df : geopandas.GeoDataFrame
        Input zones dataframe, to which results will be aggregated. This
        function assumes that the index of zones_df is unique for each feature. If
        this is not the case, unexpected results may occur.
    dset_src : str
        Path to input raster dataset to be summarized. This function assumes that
        the values in this dataset range from 0 to 1, indicating the fraction of the
        pixel to count when summing the total area inthe zone. If the input raster's
        values do no range from 0 (no inclusion) to 1 (full inclusion), the output
        results may be nonsensical.

    Returns
    -------
    pandas.DataFrame
        Returns a pandas DataFrame with a "value" column, representing the total area
        of raster within each zone. The index from the input zones_df is also included.
    """

    sums_df = zonal_statistic(zones_df, dset_src, stat="sum")
    with rasterio.open(dset_src, "r") as src:
        height, width = src.res

    pixel_area = height * width

    areas_df = sums_df * pixel_area

    return areas_df
