"""
config.analyze_features module - Configuration for analyze-features command.
"""
from typing import List, Literal, Optional

from pydantic import Field, model_validator
from typing_extensions import Annotated

from reVeal.config.config import BaseGridConfig


class AnalyzeFeaturesConfig(BaseGridConfig):
    """
    Configuration for the analyze-features command.

    Defines inputs for computing feature correlation analysis, hierarchical
    clustering, and generating exclusion suggestions on a normalized grid.
    """

    attributes: Optional[List[str]] = None
    exclude_attributes: Optional[List[str]] = None
    correlation_method: Literal["spearman", "pearson"] = "spearman"
    cluster_threshold: Annotated[float, Field(gt=0, le=2)] = 0.7

    @model_validator(mode="after")
    def _validate_attribute_options(self):
        """Ensure at most one attribute selection method is specified."""
        if self.attributes is not None and self.exclude_attributes is not None:
            raise ValueError(
                "Only one of 'attributes' or 'exclude_attributes' "
                "can be specified."
            )
        return self
