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

from reVeal.config.config import BaseEnum, BaseGridConfig
from reVeal.errors import CSVReadError, FileFormatError
from reVeal.fileio import (
    get_attributes_parquet,
    get_attributes_pyogrio,
)


class ProjectionResolutionEnum(BaseEnum):
    """
    Enumeration for allowable load resolutions. Case insensitive.
    """

    TOTAL = "total"
    REGIONAL = "regional"


class OutputValuesEnum(BaseEnum):
    """
    Enumeration for allowable output value types. Case insensitive.
    """

    INCREMENTAL = "incremental"
    CUMULATIVE = "cumulative"


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
    output_values: Optional[OutputValuesEnum] = "incremental"

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
                    f"Specified grid attribute {check_col} does not exist in the input "
                    f"dataset {self.grid}."
                )
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Specified grid attribute {check_col} must be numeric."
                )

        return self

    @model_validator(mode="after")
    def validate_load_growth(self):
        """
        Validate the input load_projections dataset can be opened, has the expected
        attributes, and  those attributes are numeric. Also ensures that the
        projections are for years after the the specified baseline year.
        """

        try:
            df = pd.read_csv(self.load_projections)
        except UnicodeDecodeError as e:
            raise CSVReadError(
                "Unable to parse input as 'utf-8' text. Is the input a CSV?"
            ) from e
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
                f"First year in load_projections ({min_year}) precedes the input "
                f"baseline_year ({self.baseline_year})."
            )

        return self


# class RegionalDownscaleConfig(BaseDownscaleConfig):
#     """
#     Model for regional downscaling configuration. Extends BaseDownscaleConfig with
#     additional validations for regional downscaling.
#     """
#     # pylint: disable=too-few-public-methods


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
#             self = base_config
#         elif resolution == "regional":
#             self = RegionalDownscaleConfig(**self)
#         else:
#             raise ValueError(
#                 f"Unexpected input for projection_resolution: {resolution}"
#                 f"Expected values are: {ProjectionResolutionEnum}"
#             )
