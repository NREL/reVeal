"""
config.learn_weights module - Configuration for learn-weights command.
"""
from typing import List, Literal, Optional

from pydantic import model_validator, FilePath, Field
from typing_extensions import Annotated

from reVeal.config.config import BaseModelStrict, BaseGridConfig


class LearnWeightsConfig(BaseGridConfig):
    """
    Configuration for the learn-weights command.

    Defines inputs for training a PU (Positive-Unlabeled) ExtraTrees model
    on a normalized grid to derive feature importance weights.
    """

    labels: FilePath
    attributes: Optional[List[str]] = None
    n_estimators: Annotated[int, Field(ge=10, le=10000)] = 500
    class_prior: Optional[Annotated[float, Field(gt=0, lt=1)]] = None
    background_samples: Annotated[int, Field(ge=100)] = 10000
    test_size: Annotated[float, Field(gt=0, lt=1)] = 0.2
    validation_size: Annotated[float, Field(gt=0, lt=1)] = 0.1
    n_jobs: Annotated[int, Field(ge=1)] = 1
    random_state: int = 42
    score_name: str = "suitability_score"
    crs: str = "EPSG:5070"
    tune: bool = False
    n_trials: Annotated[int, Field(ge=1, le=1000)] = 20
    tuning_metric: Literal["auc", "tpr"] = "auc"
