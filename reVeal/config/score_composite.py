"""
config.score_composite module
"""
# from typing import Optional
# import warnings

from typing_extensions import Annotated
from pydantic import (
    # field_validator,
    model_validator,
    FilePath,
    Field,
)

from reVeal.config.config import BaseModelStrict  # , BaseGridConfig
from reVeal.fileio import attribute_is_numeric


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
