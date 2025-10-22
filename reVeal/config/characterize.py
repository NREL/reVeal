# -*- coding: utf-8 -*-
"""
config.characterize module
"""
from typing import Optional
import warnings
from pathlib import Path

from rasterio.drivers import raster_driver_extensions
from pyogrio._ogr import _get_drivers_for_path
from pydantic import (
    field_validator,
    model_validator,
    FilePath,
    DirectoryPath,
    constr,
    NonNegativeInt,
    PositiveInt,
)
from rex.utilities import check_eval_str

from reVeal.fileio import (
    get_geom_type_pyogrio,
    get_geom_type_parquet,
    get_crs_raster,
    get_crs_pyogrio,
    get_crs_parquet,
    attribute_is_numeric,
)
from reVeal.config.config import BaseEnum, BaseModelStrict, BaseGridConfig


VALID_CHARACTERIZATION_METHODS = {
    "feature count": {
        "valid_inputs": ["point"],
        "attribute_required": False,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "sum attribute": {
        "valid_inputs": ["point"],
        "attribute_required": True,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "sum length": {
        "valid_inputs": ["line"],
        "attribute_required": False,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "sum attribute-length": {
        "valid_inputs": ["line"],
        "attribute_required": True,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "sum area": {
        "valid_inputs": ["polygon"],
        "attribute_required": False,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "area-weighted average": {
        "valid_inputs": ["polygon"],
        "attribute_required": True,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "percent covered": {
        "valid_inputs": ["polygon"],
        "attribute_required": False,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "area-apportioned sum": {
        "valid_inputs": ["polygon"],
        "attribute_required": True,
        "supports_weights": False,
        "supports_parallel": False,
        "supports_where": True,
    },
    "mean": {
        "valid_inputs": ["raster"],
        "attribute_required": False,
        "supports_weights": True,
        "supports_parallel": True,
        "supports_where": False,
    },
    "median": {
        "valid_inputs": ["raster"],
        "attribute_required": False,
        "supports_weights": False,
        "supports_parallel": True,
        "supports_where": False,
    },
    "sum": {
        "valid_inputs": ["raster"],
        "attribute_required": False,
        "supports_weights": True,
        "supports_parallel": True,
        "supports_where": False,
    },
    "area": {
        "valid_inputs": ["raster"],
        "attribute_required": False,
        "supports_weights": True,
        "supports_parallel": True,
        "supports_where": False,
    },
}


class DatasetFormatEnum(BaseEnum):
    """
    Enumeration for allowable dataset formats. Case insensitive.
    """

    RASTER = "raster"
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"


class Characterization(BaseModelStrict):
    """
    Inputs for a single characterization entry in the CharacterizeConfig.
    """

    # pylint: disable=too-few-public-methods

    # Input at instantiation
    dset: str
    data_dir: DirectoryPath
    method: constr(to_lower=True)
    attribute: Optional[str] = None
    weights_dset: Optional[str] = None
    parallel: bool = True
    max_workers: Optional[PositiveInt] = None
    neighbor_order: NonNegativeInt = 0
    buffer_distance: float = 0.0
    where: Optional[str] = None
    # Derived dynamically
    dset_src: FilePath
    dset_format: Optional[DatasetFormatEnum] = None
    dset_ext: Optional[str] = None
    crs: Optional[str] = None
    weights_dset_src: Optional[FilePath] = None

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

    @model_validator(mode="before")
    def set_dset_src(self):
        """
        Dynamically set the the dset_source property by joining input data_dir
        and dset.
        """

        if self.get("data_dir") and self.get("dset"):
            self["dset_src"] = Path(self["data_dir"]) / self["dset"]

        return self

    @model_validator(mode="before")
    def set_weights_dset_src(self):
        """
        Dynamically set the the weights_dset_src property by joining input data_dir
        and weights_dset.
        """

        if self.get("data_dir") and self.get("weights_dset"):
            self["weights_dset_src"] = Path(self["data_dir"]) / self["weights_dset"]

        return self

    @model_validator(mode="after")
    def set_dset_ext(self):
        """
        Dynamically set the dset_ext property.
        """

        self.dset_ext = self.dset_src.suffix

        return self

    @model_validator(mode="after")
    def set_dset_format(self):
        """
        Dynamically set the the dset_source property.
        """

        if self.dset_ext == ".parquet":
            dset_format = get_geom_type_parquet(self.dset_src)
        elif _get_drivers_for_path(self.dset):
            dset_format = get_geom_type_pyogrio(self.dset_src)
        elif self.dset_ext[1:] in raster_driver_extensions():
            # note: order matters in these checks - do raster to avoid confusion on
            # gpkg
            dset_format = "raster"
        else:
            raise TypeError(f"Unsupported file format for {self.dset_src}.")

        self.dset_format = DatasetFormatEnum(dset_format)

        return self

    @model_validator(mode="after")
    def where_check(self):
        """
        Check that entry for where does not contain any questionable code.
        Also issues a warning if where is specified but doesn't apply to the specified
        method.
        """

        if self.where:
            # always check, even if it doesn't apply (overkill, but just in case)
            check_eval_str(self.where)
            method_info = VALID_CHARACTERIZATION_METHODS.get(self.method)
            if not method_info.get("supports_where"):
                warnings.warn(
                    f"where specified but will not be applied for {self.method}"
                )

        return self

    @model_validator(mode="after")
    def attribute_check(self):
        """
        Check that attribute is provided for required methods and warn if attribute
        is provided for methods where it doesn't apply. Also ensure that the attribute
        is present in the input dataset and is a numeric datatype.

        Raises
        ------
        ValueError
            A ValueError will be raised if attribute is missing for a required method
            or does not exist in the input dataset.
        TypeError
            A TypeError will be raised if the input attribute exists in the dataset
            but is not a numeric datatype.
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
        if attribute_required and self.attribute:
            if not attribute_is_numeric(self.dset_src, self.attribute):
                raise TypeError(
                    f"Attribute {self.attribute} in {self.dset_src} is invalid type. "
                    "Must be a numeric dtype."
                )
        return self

    @model_validator(mode="after")
    def set_crs(self):
        """
        Dynamically set the crs property.
        """

        if self.dset_format == "raster":
            self.crs = get_crs_raster(self.dset_src)
        elif self.dset_ext == ".parquet":
            self.crs = get_crs_parquet(self.dset_src)
        else:
            self.crs = get_crs_pyogrio(self.dset_src)

        return self

    @model_validator(mode="after")
    def check_method_applicability(self):
        """
        Check that the specified method is applicable to the input dset_format.
        """

        applicable_types = VALID_CHARACTERIZATION_METHODS.get(self.method, {}).get(
            "valid_inputs"
        )
        if self.dset_format not in applicable_types:
            raise ValueError(
                f"Incompatible method ({self.method}) and dataset format "
                f"({self.dset_format}) for dataset {self.dset_src}"
            )

        return self

    @model_validator(mode="after")
    def weights_dset_check(self):
        """
        Check that, if weights_dset is provided, the selected method is applicable
        to the method. If not, warn the user.
        """

        if self.weights_dset:
            method_info = VALID_CHARACTERIZATION_METHODS.get(self.method)
            if not method_info.get("supports_weights"):
                warnings.warn(
                    f"weights_dset specified but will not be applied for {self.method}"
                )

        return self

    @model_validator(mode="after")
    def parallel_check(self):
        """
        Check that, if parallel is set to True or max_workers was provided, the
        selected method can be parallelized. If not, warn the user.
        """

        if self.parallel or self.max_workers:
            method_info = VALID_CHARACTERIZATION_METHODS.get(self.method)
            if not method_info.get("supports_parallel"):
                warnings.warn(
                    "parallel specified as True and/or max_workers provided but "
                    f"parallel processing is not implemented for {self.method}."
                )

        return self


class BaseCharacterizeConfig(BaseGridConfig):
    """
    Base model for CharacterizeConfig with only required inputs and datatypes.
    """

    # pylint: disable=too-few-public-methods

    # Input at instantiation
    data_dir: DirectoryPath
    characterizations: dict
    expressions: Optional[dict] = None


class CharacterizeConfig(BaseCharacterizeConfig):
    """
    Configuration for characterize command.
    """

    # pylint: disable=too-few-public-methods

    @model_validator(mode="before")
    def propagate_datadir(self):
        """
        Propagate the top level data_dir parameter down to elements of
        characterizations before validation.

        Returns
        -------
        self
            Returns self.
        """

        for v in self["characterizations"].values():
            if "data_dir" not in v:
                v["data_dir"] = self["data_dir"]

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

        BaseCharacterizeConfig(**self)

        return self

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
            into an instance of Characterization.
        """
        # pylint: disable=no-self-argument

        for k, v in value.items():
            value[k] = Characterization(**v)

        return value

    @field_validator("expressions")
    def validate_expressions(cls, value):
        """
        Check that each entry in the expressions dictionary is a string and does not
        contain any questionable code.

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
            check_eval_str(v)

        return value

    @model_validator(mode="after")
    def validate_crs(self):
        """
        Check that CRSs of individual characterizations match CRS of the grid.
        """

        for characterization in self.characterizations.values():
            if characterization.crs != self.grid_crs:
                raise ValueError(
                    f"CRS of input dataset {characterization.dset_src} "
                    f"({characterization.crs}) does not match grid CRS "
                    f"({self.grid_crs})."
                )
        return self
