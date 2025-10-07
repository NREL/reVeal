# -*- coding: utf-8 -*-
"""
config.score_attributes module
"""
from typing import Optional
import warnings

from pydantic import (
    field_validator,
    model_validator,
    FilePath,
)
from pandas.api.types import is_numeric_dtype

from reVeal.config.config import BaseEnum, BaseModelStrict, BaseGridConfig
from reVeal.fileio import (
    attribute_is_numeric,
    get_attributes_parquet,
    get_attributes_pyogrio,
)

GRID_IDX = "gid"


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
    invert: bool = False

    @model_validator(mode="after")
    def attribute_check(self):
        """
        Check that attribute is present in the input dataset and is a numeric datatype.

        Raises
        ------
        TypeError
            A TypeError will be raised if the input attribute exists in the dataset
            but is not a numeric datatype.
        """

        if not attribute_is_numeric(self.dset_src, self.attribute):
            raise TypeError(
                f"Attribute {self.attribute} in {self.dset_src} is invalid type. Must "
                "be a numeric dtype."
            )
        return self


class BaseScoreAttributesConfig(BaseGridConfig):
    """
    Base model for ScoreAttributesConfig with only required inputs and datatypes.
    """

    # pylint: disable=too-few-public-methods

    # Input at instantiation
    attributes: dict = {}
    score_method: Optional[AttributeScoringMethodEnum] = None
    invert: bool = False


class ScoreAttributesConfig(BaseScoreAttributesConfig):
    """
    Configuration for score-attributes command.
    """

    # pylint: disable=too-few-public-methods

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
        if self.get("attributes"):
            for v in self["attributes"].values():
                if "dset_src" not in v:
                    v["dset_src"] = self["grid"]

        return self

    @model_validator(mode="before")
    def base_validator(self):
        """
        Ensures that the base validation is run on input data types before
        other "before"-mode model validators.

        Returns
        -------
        self
            Returns self.
        """
        BaseScoreAttributesConfig(**self)

        return self

    @model_validator(mode="before")
    def check_attributes_and_score_method(self):
        """
        Check that either attributes or score_method was provided as an input.
        """
        if not self.get("score_method") and not self.get("attributes"):
            raise ValueError("Either score_method or attributes must be specified.")

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

    @model_validator(mode="after")
    def propagate_score_method(self):
        """
        If the top-level score method is specified, populate the attributes property
        so that it includes all numeric attributes in the input grid. All attributes
        will use the specified top-level score method except for any that were input
        separately via the attributes parameter.
        """

        if self.score_method:
            if self.grid_flavor == "geoparquet":
                dset_attributes = get_attributes_parquet(self.grid)
            else:
                dset_attributes = get_attributes_pyogrio(self.grid)

            attributes = {}
            for attr, attr_dtype in dset_attributes.items():
                if attr != GRID_IDX and is_numeric_dtype(attr_dtype):
                    out_col = f"{attr}_score"
                    if out_col in dset_attributes:
                        warnings.warn(
                            f"Output column {out_col} exists in input grid and will be "
                            "overwritten in output."
                        )
                    attributes[out_col] = Attribute(
                        attribute=attr,
                        score_method=self.score_method,
                        dset_src=self.grid,
                        invert=self.invert,
                    )
            # preserve any existing attributes that were explicitly defined
            attributes.update(self.attributes)

            self.attributes = attributes

        return self
