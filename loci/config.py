# -*- coding: utf-8 -*-
"""
config module
"""
from typing import Optional
import warnings

from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    FilePath,
    DirectoryPath,
    constr,
    NonNegativeInt,
    StrictBool,
)


class BaseModelStrict(BaseModel):
    """
    Customizing BaseModel to perform strict checking that will raise a ValidationError
    for extra parameters.
    """

    # pylint: disable=too-few-public-methods
    model_config = {"extra": "forbid"}


VALID_CHARACTERIZATION_METHODS = {
    "feature count": {
        "valid_inputs": ["Point"],
        "attribute_required": False,
    },
    "sum attribute": {
        "valid_inputs": ["Point"],
        "attribute_required": True,
    },
    "sum length": {
        "valid_inputs": ["Line"],
        "attribute_required": False,
    },
    "sum attribute-length": {
        "valid_inputs": ["Line"],
        "attribute_required": True,
    },
    "sum area": {
        "valid_inputs": ["Polygon"],
        "attribute_required": False,
    },
    "area-weighted attribute average": {
        "valid_inputs": ["Polygon"],
        "attribute_required": True,
    },
    "percent covered": {
        "valid_inputs": ["Polygon"],
        "attribute_required": False,
    },
    "area-apportioned attribute sum": {
        "valid_inputs": ["Polygon"],
        "attribute_required": True,
    },
    "mean": {
        "valid_inputs": ["Raster"],
        "attribute_required": False,
    },
    "median": {
        "valid_inputs": ["Raster"],
        "attribute_required": False,
    },
    "sum": {
        "valid_inputs": ["Raster"],
        "attribute_required": False,
    },
    "area": {
        "valid_inputs": ["Raster"],
        "attribute_required": False,
    },
}


class Characterization(BaseModelStrict):
    """
    Inputs for a single entry in the characterizations config.
    """

    # pylint: disable=too-few-public-methods

    dset: str
    method: constr(to_lower=True)
    attribute: Optional[str] = None
    apply_exclusions: Optional[StrictBool] = False
    neighbor_order: Optional[NonNegativeInt] = 0.0
    buffer_distance: Optional[float] = 0.0

    @field_validator("method")
    def is_valid_method(cls, value):
        """
        Check that method is one of the allowable values.

        Parameters
        ----------
        value : str
            Input value

        Returns
        -------
        str
            Output value

        Raises
        ------
        ValueError
            A ValueError will be raised if the input method is invalid.
        """
        # pylint: disable=no-self-argument

        if value not in VALID_CHARACTERIZATION_METHODS:
            raise ValueError(
                f"Invalid method specified: {value}. "
                f"Valid options are: {VALID_CHARACTERIZATION_METHODS}"
            )
        return value

    @model_validator(mode="after")
    def attribute_check(self):
        """
        Check that attribute is provided for required methods and warn if attribute
        is provided for methods where it doesn't apply.

        Raises
        ------
        ValueError
            A ValueError will be raised if attribute is missing for a required method.
        """
        method_info = VALID_CHARACTERIZATION_METHODS.get(self.method)
        if method_info is None or method_info.get("attribute_required") is None:
            raise ValueError(
                "Missing information required to determine if attribute is required "
                f"for the specified method {self.method}"
            )
        attribute_required = method_info.get("attribute_required")
        if attribute_required and self.attribute is None:
            raise ValueError(
                f"attribute was not provided, but is required for method {self.method}"
            )
        if not attribute_required and self.attribute:
            warnings.warn(
                f"attribute specified but will not be applied for {self.method}"
            )

        return self


class CharacterizeConfig(BaseModelStrict):
    """
    Configuration for characterize command.
    """

    # pylint: disable=too-few-public-methods

    data_dir: DirectoryPath
    grid: FilePath
    characterizations: dict
    expressions: Optional[dict] = None

    @field_validator("characterizations")
    def validate_characterizations(cls, value):
        """
        Validate each entry in the input charactrizations dictionary.

        Parameters
        ----------
        value : dict
            Input characterizations.

        Returns
        -------
        dict
            Validated characterizations, which each value converted
            into an instance of CharacterizationSpec.
        """
        # pylint: disable=no-self-argument

        for k, v in value.items():
            value[k] = Characterization(**v)

        return value

    @field_validator("expressions")
    def validate_expressions(cls, value):
        """
        Check that each entry in the expressions dictionary is a string.

        Parameters
        ----------
        value : dict
            Input expressions.

        Returns
        -------
        dict
            Validated expressions.
        """
        # pylint: disable=no-self-argument
        for k, v in value.items():
            if not isinstance(v, str):
                raise TypeError(
                    f"Invalid input for expressions entry {k}: {v}. Must be a string."
                )

        return value
