# -*- coding: utf-8 -*-
"""
config.score_attributes module
"""
from typing import Optional

from pyogrio._ogr import _get_drivers_for_path
from pydantic import (
    field_validator,
    model_validator,
    FilePath,
)
from pandas.api.types import is_numeric_dtype

from reVeal.config.config import BaseEnum, BaseModelStrict
from reVeal.fileio import get_attributes_parquet, get_attributes_pyogrio


class AttributeScoringMethodEnum(BaseEnum):
    """
    Enumeration for allowable scoring methods. Case insensitive.
    """

    PERCENTILE = "percentile"
    MINMAX = "minmax"


class Attribute(BaseModelStrict):
    """
    Inputs for a single attribute entry in the ScoreAttributesConfig.
    """

    # Input at instantiation
    attribute: str
    score_method: AttributeScoringMethodEnum
    dset_src: FilePath
    # Derived dynamically
    dset_ext: Optional[str] = None
    dset_flavor: Optional[str] = None

    @model_validator(mode="after")
    def set_dset_ext(self):
        """
        Dynamically set the dset_ext property.
        """
        self.dset_ext = self.dset_src.suffix

        return self

    @model_validator(mode="after")
    def set_dset_flavor(self):
        """
        Dynamically set the dset_flavor.

        Raises
        ------
        TypeError
            A TypeError will be raised if the input dset is not either a geoparquet
            or compatible with reading with ogr.
        """
        if self.dset_ext == ".parquet":
            self.dset_flavor = "geoparquet"
        elif _get_drivers_for_path(self.dset_src):
            self.dset_flavor = "ogr"
        else:
            raise TypeError(f"Unrecognized file format for {self.dset_src}.")

        return self

    @model_validator(mode="after")
    def attribute_check(self):
        """
        Check that attribute is present in the input dataset and is a numeric datatype.

        Raises
        ------
        ValueError
            A ValueError will be raised if attribute does not exist in the input
            dataset.
        TypeError
            A TypeError will be raised if the input attribute exists in the dataset
            but is not a numeric datatype.
        """

        if self.dset_flavor == "geoparquet":
            dset_attributes = get_attributes_parquet(self.dset_src)
        else:
            dset_attributes = get_attributes_pyogrio(self.dset_src)

        attr_dtype = dset_attributes.get(self.attribute)
        if not attr_dtype:
            raise ValueError(f"Attribute {self.attribute} not found in {self.dset_src}")
        if not is_numeric_dtype(attr_dtype):
            raise TypeError(
                f"Attribute {self.attribute} in {self.dset_src} is invalid "
                f"type {attr_dtype}. Must be a numeric dtype."
            )

        return self


class ScoreAttributesConfig(BaseModelStrict):
    """
    Configuration for characterize command.
    """

    # pylint: disable=too-few-public-methods

    # Input at instantiation
    grid: FilePath
    attributes: dict

    @model_validator(mode="before")
    def propagate_grid(self):
        """
        Propagate the top level grid parameter down to elements of
        attributes before validation.

        Returns
        -------
        self
            Returns self.
        """
        for v in self["attributes"].values():
            if "dset_src" not in v:
                v["dset_src"] = self["dset_src"]

        return self

    @field_validator("attributes")
    def validate_attributes(cls, value):
        """
        Validate each entry in the input attributes dictionary.

        Parameters
        ----------
        value : dict
            Input attributes.

        Returns
        -------
        dict
            Validated attributes, which each value converted
            into an instance of Attribute.
        """
        # pylint: disable=no-self-argument

        for k, v in value.items():
            value[k] = Attribute(**v)

        return value
