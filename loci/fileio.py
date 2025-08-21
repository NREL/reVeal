# -*- coding: utf-8 -*-
"""
io module
"""
import pyogrio
import rasterio
import pyproj
from geopandas.io.arrow import (
    _read_parquet_schema_and_metadata,
    _validate_and_decode_metadata,
)

from loci.config import CharacterizeConfig

GEOMETRY_TYPES = {
    "Point": "point",
    "Polygon": "polygon",
    "LineString": "line",
    "MultiPolygon": "polygon",
    "MultiLineString": "line",
}


def load_characterize_config(characterize_config):
    """
    Load config for grid characterization.

    Parameters
    ----------
    characterize_config : [dict, CharacterizeConfig]
        Input configuration. If a dictionary, it will be converted to an instance of
        CharacterizeConfig, with validation. If a CharacterizeConfig, the input
        will be returned unchanged.

    Returns
    -------
    CharacterizeConfig
        Output CharacterizeConfig instance.

    Raises
    ------
    TypeError
        A TypeError will be raised if the input is neither a dict or CharacterizeConfig
        instance.
    """

    if isinstance(characterize_config, dict):
        return CharacterizeConfig(**characterize_config)

    if isinstance(characterize_config, CharacterizeConfig):
        return characterize_config

    raise TypeError(
        "Invalid input for characterize config. Must be an instance of "
        "either dict or CharacterizeConfig."
    )


def get_geom_info_parquet(dset_src):
    """
    Reads geometry information from geoparquet file.

    Parameters
    ----------
    dset_src : str
        Path to GeoParquet file.

    Returns
    -------
    dict
        Returns a dictionary of geometry information from parquet file.
    """
    _, metadata = _read_parquet_schema_and_metadata(dset_src, None)
    geo_metadata = _validate_and_decode_metadata(metadata)
    geom_col = geo_metadata["primary_column"]
    geom_col_info = geo_metadata["columns"][geom_col]

    return geom_col_info


def get_geom_type_parquet(dset_src):
    """
    Determine the generic geometry type of of an input GeoParquet dataset.

    Parameters
    ----------
    dset_src : str
        Path to input GeoParquet dataset.

    Returns
    -------
    str
        Geometry type. One of: "point", "line", or "polygon"."

    Raises
    ------
    ValueError
        A ValueError will be raised if any of the following issues are encountered:
        - The geometry type cannot be parsed from the schema of the input file
        - There are multiple geometry types in the input file
        - The input geometry type is not a valid/supported option.
    """
    geom_col_info = get_geom_info_parquet(dset_src)
    in_geom_types = geom_col_info.get("geometry_types")
    geom_types = []
    for in_geom_type in in_geom_types:
        std_geom_type = GEOMETRY_TYPES.get(in_geom_type)
        if std_geom_type is None:
            raise ValueError(
                f"Unsupported geometry type: {in_geom_type}."
                f"Supported options are: {GEOMETRY_TYPES.keys()}"
            )
        geom_types.append(std_geom_type)
    if len(set(geom_types)) > 1:
        raise ValueError(
            f"Multiple geometry types encountered in {dset_src}: {geom_types}."
        )

    return geom_types[0]


def get_geom_type_pyogrio(dset_src):
    """
    Determine the generic geometry type of of an input vector dataset that can be read
    by pyogrio.

    Parameters
    ----------
    dset_src : str
        Path to input vector dataset.

    Returns
    -------
    str
        Geometry type. One of: "point", "line", or "polygon"."

    Raises
    ------
    ValueError
        A ValueError will be raised if the input dataset is not one of the known
        formats.
    """
    dset_info = pyogrio.read_info(dset_src)
    geom_type = GEOMETRY_TYPES.get(dset_info["geometry_type"])
    if geom_type is None:
        raise ValueError(
            f"Unsupported geometry type: {dset_info['geometry_type']}."
            f"Supported options are: {GEOMETRY_TYPES.keys()}"
        )

    return geom_type


def get_crs_raster(dset_src):
    """
    Get the coordinate reference system of a raster dataset.

    Parameters
    ----------
    dset_src : str
        Path to dataset.

    Returns
    -------
    str
        CRS as an EPSG code.
    """
    with rasterio.open(dset_src, "r") as src:
        crs = src.crs

    authority_code = ":".join(crs.to_authority())

    return authority_code


def get_crs_pyogrio(dset_src):
    """
    Get the coordinate reference system of a vector dataset that can be opened with
    pyogrio.

    Parameters
    ----------
    dset_src : str
        Path to dataset.

    Returns
    -------
    str
        CRS as an EPSG code.
    """
    dset_info = pyogrio.read_info(dset_src)
    authority_code = dset_info["crs"]
    if authority_code is None:
        raise ValueError(f"Could not determine CRS  for {dset_src})")

    return authority_code


def get_crs_parquet(dset_src):
    """
    Get the coordinate reference system of a GeoParquet dataset.

    Parameters
    ----------
    dset_src : str
        Path to dataset.

    Returns
    -------
    str
        CRS as an EPSG code.
    """
    geom_col_info = get_geom_info_parquet(dset_src)
    crs_info = geom_col_info.get("crs")
    if crs_info is None:
        raise ValueError(f"Could not determine CRS  for {dset_src})")
    crs = pyproj.CRS.from_user_input(crs_info)
    authority_code = ":".join(crs.to_authority())

    return authority_code
