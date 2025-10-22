# -*- coding: utf-8 -*-
"""
config.downscale module
"""
from typing import Optional

from pydantic import (
    model_validator,
    PositiveInt,
    FilePath,
)
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pyogrio._ogr import _get_drivers_for_path

from reVeal.config.config import BaseEnum, BaseGridConfig
from reVeal.errors import CSVReadError, FileFormatError
from reVeal.fileio import (
    get_attributes_parquet,
    get_attributes_pyogrio,
    get_geom_type_parquet,
    get_geom_type_pyogrio,
    get_crs_parquet,
    get_crs_pyogrio,
)


class ProjectionResolutionEnum(BaseEnum):
    """
    Enumeration for allowable load resolutions. Case insensitive.
    """

    TOTAL = "total"
    REGIONAL = "regional"


class TotalResolutionEnum(BaseEnum):
    """
    Enumeration for allowable load resolutions for total downscaling. Case insensitive.
    """

    TOTAL = "total"


class RegionalResolutionEnum(BaseEnum):
    """
    Enumeration for allowable load resolutions for regional downscaling. Case
    insensitive.
    """

    REGIONAL = "regional"


class BaseDownscaleConfig(BaseGridConfig):
    """
    Base model for DownscaleConfig with only required inputs and datatypes.
    """

    # pylint: disable=too-few-public-methods

    # Input at instantiation
    grid_priority: str
    grid_baseline_load: str
    baseline_year: PositiveInt
    projection_resolution: ProjectionResolutionEnum
    load_projections: FilePath
    load_value: str
    load_year: str

    @model_validator(mode="after")
    def validate_grid(self):
        """
        Validate the input grid dataset can be opened, has the expected attributes, and
        those attributes are numeric.
        """
        if self.grid_flavor == "geoparquet":
            dset_attributes = get_attributes_parquet(self.grid)
        else:
            dset_attributes = get_attributes_pyogrio(self.grid)

        check_cols = [self.grid_priority, self.grid_baseline_load]

        for check_col in check_cols:
            dtype = dset_attributes.get(check_col)
            if not dtype:
                raise ValueError(
                    f"Specified attribute {check_col} does not exist in the input "
                    f"dataset {self.grid}."
                )
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Specified grid attribute {check_col} must be numeric."
                )

        return self

    @model_validator(mode="after")
    def validate_load_projections(self):
        """
        Validate the input load_projections dataset can be opened, has the expected
        attributes, and  those attributes are numeric. Also ensures that the
        projections are for years after the the specified baseline year.
        """

        try:
            df = pd.read_csv(self.load_projections)
        except UnicodeDecodeError as e:
            raise CSVReadError("Unable to parse input as 'utf-8' text.") from e
        except pd.errors.ParserError as e:
            raise FileFormatError(
                "Unable to parse text as CSV. Is the input formatted correctly?"
            ) from e
        except Exception as e:
            raise CSVReadError("Pandas raised error reading input as CSV") from e

        check_cols = [self.load_value, self.load_year]
        dset_attributes = df.dtypes
        for check_col in check_cols:
            dtype = dset_attributes.get(check_col)
            if not dtype:
                raise ValueError(
                    f"Specified attribute {check_col} does not exist in the input "
                    f"load_projections dataset {self.load_projections}."
                )
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Specified load_projections attribute {check_col} must be numeric."
                )

        min_year = df[self.load_year].min()
        if min_year < self.baseline_year:
            raise ValueError(
                f"First year in load_projections ({min_year}) predates the input "
                f"baseline_year ({self.baseline_year})."
            )

        return self


class TotalDownscaleConfig(BaseDownscaleConfig):
    """
    Model for total downscaling configuration. Extends BaseDownscaleConfig with
    additional validations.
    """

    # pylint: disable=too-few-public-methods
    projection_resolution: TotalResolutionEnum

    @model_validator(mode="after")
    def validate_load_projections_duplicates(self):
        """
        Validate the input load_projections dataset can be opened, has the expected
        attributes, and  those attributes are numeric. Also ensures that the
        projections are for years after the the specified baseline year.
        """

        df = pd.read_csv(self.load_projections)
        if df[self.load_year].duplicated().any():
            raise ValueError("Input load_projections dataset has duplicate years")

        return self


class RegionalDownscaleConfig(BaseDownscaleConfig):
    """
    Model for regional downscaling configuration. Extends BaseDownscaleConfig with
    additional validations for regional downscaling.
    """

    # pylint: disable=too-few-public-methods
    projection_resolution: RegionalResolutionEnum
    load_regions: Optional[str] = None
    region_weights: Optional[dict] = None
    regions: FilePath
    region_names: str
    # Dynamically derived attributes
    regions_ext: Optional[str] = None
    regions_flavor: Optional[str] = None

    @model_validator(mode="after")
    def set_regions_ext(self):
        """
        Dynamically set the regions_ext property.
        """
        self.regions_ext = self.regions.suffix

        return self

    @model_validator(mode="after")
    def set_regions_flavor(self):
        """
        Dynamically set the regions_flavor.

        Raises
        ------
        TypeError
            A TypeError will be raised if the input dset is not either a geoparquet
            or compatible with reading with ogr.
        """
        if self.regions_ext == ".parquet":
            self.regions_flavor = "geoparquet"
        elif _get_drivers_for_path(self.grid):
            self.regions_flavor = "ogr"
        else:
            raise TypeError(f"Unrecognized file format for {self.regions}.")

        return self

    @model_validator(mode="after")
    def validate_regions(self):
        """
        Validates the input regions dataset:
        1. Has either Polygon or MultiPolygon geometries.
        2. Has a column corresponding to the input region_names parameter
        3. Has a CRS matching the input grid.
        """

        if self.regions_flavor == "geoparquet":
            geom_type = get_geom_type_parquet(self.regions)
            attributes = get_attributes_parquet(self.regions)
            crs = get_crs_parquet(self.regions)
        elif self.regions_flavor == "ogr":
            geom_type = get_geom_type_pyogrio(self.regions)
            attributes = get_attributes_pyogrio(self.regions)
            crs = get_crs_pyogrio(self.regions)
        else:
            raise TypeError(f"Unrecognized file format for {self.regions}.")

        valid_geom_types = ["polygon", "multipolygon"]
        if geom_type not in valid_geom_types:
            raise TypeError(
                "Input regions dataset must have geometries of one of the following "
                f"types: {valid_geom_types}."
            )

        if self.region_names not in attributes:
            raise ValueError(
                f"region_names attribute {self.region_names} does not exist in source "
                f"dataset {self.regions}."
            )

        if crs != self.grid_crs:
            raise ValueError(
                f"CRS of regions dataset {self.regions} ({crs}) does not match grid "
                f"CRS ({self.grid_crs})."
            )

        return self

    # check that either load_regions or region_weights is provided,b ut not both
    # if load_regions is provided:
    # validate the the load region column exists in the input loads dataframe
    # check for duplicates by load_region and year
    # check that load_regions maps to regions in regions dataset???
    # maybe move this downstream?
    # if region_weights is provided:
    # check that region_weights maps to regions in the regions dataset???
    # maybe move this downstream?


# class DownscaleConfig(BaseDownscaleConfig):
#     """
#     Model for downscaling configuration. Based on the inputs, wraps either
#     BaseDownScaleConfig or RegionalDownscaleConfig.
#     """
#     # pylint: disable=too-few-public-methods

#     def __init__(self, *args, **kwargs):

#         base_config = super(**self)
#         resolution = base_config.projection_resolution
#         if resolution == "total":
#             self = TotalDownscaleConfig(**self)
#         elif resolution == "regional":
#             self = RegionalDownscaleConfig(**self)
#         else:
#             raise ValueError(
#                 f"Unexpected input for projection_resolution: {resolution}"
#                 f"Expected values are: {ProjectionResolutionEnum}"
#             )
