"""
config.score_composite module
"""
from typing import List
import warnings
from math import isclose

from typing_extensions import Annotated
from pydantic import model_validator, FilePath, Field

from reVeal.config.config import BaseModelStrict, BaseGridConfig
from reVeal.fileio import (
    attribute_is_numeric,
    get_attributes_parquet,
    get_attributes_pyogrio,
)


class Attribute(BaseModelStrict):
    """
    Inputs for a single attribute entry in the ScoreCompositeConfig.
    """

    # Input at instantiation
    attribute: str
    weight: Annotated[float, Field(strict=True, gt=0, le=1)]
    dset_src: FilePath

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


class BaseScoreWeightedConfig(BaseGridConfig):
    """
    Base model for ScoreWeightedConfig with only required inputs and datatypes.
    """

    # pylint: disable=too-few-public-methods

    # Input at instantiation
    attributes: List
    score_name: str


class ScoreWeightedConfig(BaseScoreWeightedConfig):
    """
    Configuration for score-weighted command.
    """

    attributes: List[Attribute]
    score_name: str

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
        for attribute in self["attributes"]:
            if "dset_src" not in attribute:
                attribute["dset_src"] = self["grid"]

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
        BaseScoreWeightedConfig(**self)

        return self

    @model_validator(mode="after")
    def validate_sum_attribute_weights(self):
        """
        Validate that the sum of all attribute weights is equal to 1.
        """

        sum_weights = 0
        for attribute in self.attributes:
            sum_weights += attribute.weight

        if not isclose(sum_weights, 1, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError(
                "Weights of input attributes must sum to 1. "
                f"Sum of input weights is: {sum_weights}."
            )

        return self

    @model_validator(mode="after")
    def validate_score_name(self):
        """
        Check whether the output attribute specified by the score_name property
        already exists in the input dataset. If so, raise a warning.
        """

        if self.grid_flavor == "geoparquet":
            dset_attributes = get_attributes_parquet(self.grid)
        else:
            dset_attributes = get_attributes_pyogrio(self.grid)

        if self.score_name in dset_attributes:
            warnings.warn(
                f"Output column {self.score_name} exists in input grid and will be "
                "overwritten in output."
            )

        return self
