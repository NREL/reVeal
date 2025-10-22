# -*- coding: utf-8 -*-
"""
config.config module
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, FilePath, model_validator
from pyogrio._ogr import _get_drivers_for_path

from reVeal.fileio import get_crs_pyogrio, get_crs_parquet


class BaseModelStrict(BaseModel):
    """
    Customizing BaseModel to perform strict checking that will raise a ValidationError
    for extra parameters.
    """

    # pylint: disable=too-few-public-methods
    model_config = {"extra": "forbid"}


class BaseEnum(str, Enum):
    """
    Base Enumeration. Extends standard Enum to be case-insensitive.

    Raises
    ------
    ValueError
        A ValueError is raised if the input value is not one of the known
        types when cast to lower case.
    """

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        raise ValueError(f"{value} is not a valid option of {cls}")


class BaseGridConfig(BaseModelStrict):
    """
    Base Configuration object for gridded commands.
    """

    grid: FilePath
    # Dynamically derived attributes
    grid_ext: Optional[str] = None
    grid_flavor: Optional[str] = None
    grid_crs: Optional[str] = None

    @model_validator(mode="after")
    def set_grid_ext(self):
        """
        Dynamically set the grid_ext property.
        """
        self.grid_ext = self.grid.suffix

        return self

    @model_validator(mode="after")
    def set_grid_flavor(self):
        """
        Dynamically set the dset_flavor.

        Raises
        ------
        TypeError
            A TypeError will be raised if the input dset is not either a geoparquet
            or compatible with reading with ogr.
        """
        if self.grid_ext == ".parquet":
            self.grid_flavor = "geoparquet"
        elif _get_drivers_for_path(self.grid):
            self.grid_flavor = "ogr"
        else:
            raise TypeError(f"Unrecognized file format for {self.grid}.")

        return self

    @model_validator(mode="after")
    def set_grid_crs(self):
        """
        Dynamically set the crs property.
        """

        if self.grid_flavor == "geoparquet":
            self.grid_crs = get_crs_parquet(self.grid)
        else:
            self.grid_crs = get_crs_pyogrio(self.grid)

        return self


def load_config(config, config_class):
    """
    Load config to specified pydantic model.

    Parameters
    ----------
    config : [dict, BaseModelStrict]
        Input configuration. If a dictionary, it will be converted to an instance of
        CharacterizeConfig, with validation. If a CharacterizeConfig, the input
        will be returned unchanged.
    config_class : type
        Config class to return. Should be a subclass of BaseModelStrict.

    Returns
    -------
    BaseModelStrict
        Output instance of config class.

    Raises
    ------
    TypeError
        A TypeError will be raised if the input is neither a dict nor an instance of
        the input config_class.
    """

    if isinstance(config, dict):
        return config_class(**config)

    if isinstance(config, config_class):
        return config

    raise TypeError(
        "Invalid input for config. Must be an instance of either dict or input "
        "config_class."
    )
