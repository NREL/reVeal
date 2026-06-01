# -*- coding: utf-8 -*-
"""
cli.learn_weights module - Sets up learn-weights command for use with NLR-GAPs CLI.
"""
import json
import logging
from pathlib import Path

from pydantic import ValidationError
from gaps.cli import as_click_command, CLICommandFromFunction

from reVeal.config.learn_weights import LearnWeightsConfig
from reVeal.log import get_logger, remove_streamhandlers
from reVeal.learn_weights import run_learn_weights

LOGGER = logging.getLogger(__name__)


def _log_inputs(config):
    """
    Emit log messages summarizing user inputs.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """
    LOGGER.info(f"Inputs config: {json.dumps(config, indent=4, default=str)}")


def _preprocessor(config, job_name, log_directory, verbose):
    """
    Preprocess user-input configuration.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    job_name : str
        Name of job being run.
    verbose : bool
        Flag to signal DEBUG verbosity.

    Returns
    -------
    dict
        Configuration dictionary modified to include additional parameters.
    """
    if verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    get_logger(
        __name__, log_level=log_level, out_path=log_directory / f"{job_name}.log"
    )

    LOGGER.info("Validating input configuration file")
    try:
        learn_config = {
            k: config.get(k)
            for k in LearnWeightsConfig.model_fields.keys()
            if k in config
        }
        LearnWeightsConfig(**learn_config)
    except ValidationError as e:
        LOGGER.error(
            "Configuration did not pass validation. "
            f"The following issues were identified:\n{e}"
        )
        raise e
    LOGGER.info("Input configuration file is valid.")

    config["_local"] = (
        config.get("execution_control", {}).get("option", "local") == "local"
    )
    _log_inputs(config)

    return config


def run(
    grid,
    labels,
    out_dir,
    attributes=None,
    exclude_attributes=None,
    n_estimators=500,
    class_prior=None,
    background_samples=10000,
    test_size=0.2,
    validation_size=0.1,
    n_jobs=1,
    random_state=42,
    score_name="suitability_score",
    crs="EPSG:5070",
    tune=False,
    n_trials=20,
    tuning_metric="auc",
    _local=True,
):
    """
    Learn feature weights from labeled data using PU (Positive-Unlabeled) learning.

    Trains a PUExtraTrees model on a normalized grid using point labels as positive
    samples and auto-sampled background cells as unlabeled samples. Outputs a
    score-weighted configuration JSON with feature importances normalized as weights
    (summing to 1.0).

    Parameters
    ----------
    grid : str
        Path to the normalized grid (output of ``reVeal normalize``). Must be a
        vector dataset readable by pyogrio with numeric ``*_score`` columns.
    labels : str
        Path to a point geometry dataset (GeoPackage, shapefile, etc.) containing
        positive sample locations. These represent known sites (e.g., data center
        locations).
    out_dir : str
        Output directory. Results will be saved as ``config_score_weighted.json``
        and ``learn_weights_metrics.json``.
    attributes : list of str, optional
        List of column names from the grid to use as features. If not specified,
        all columns ending with ``_score`` are used automatically.
    exclude_attributes : list of str, optional
        Score columns to exclude from auto-detected features. All ``*_score``
        columns except those listed will be used. Mutually exclusive with
        ``attributes``.
    n_estimators : int, optional
        Number of trees in the PUExtraTrees forest. Default is 500.
    class_prior : float, optional
        Prior probability that a sample is positive. If not specified, computed
        automatically as ``n_positive / (n_positive + n_background)``.
    background_samples : int, optional
        Number of background (unlabeled) cells to sample from non-intersecting
        grid cells. Default is 10000.
    test_size : float, optional
        Fraction of positive cells to hold out for model evaluation. Default is 0.2.
    validation_size : float, optional
        Fraction of training positives for validation. Default is 0.1.
    n_jobs : int, optional
        Number of parallel jobs for model training. Default is 1.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    score_name : str, optional
        Name of the output score column in the generated config. Default is
        ``suitability_score``.
    crs : str, optional
        Coordinate reference system for spatial operations. Default is ``EPSG:5070``.
    tune : bool, optional
        If True and class_prior is not set, use Optuna to find the optimal
        class_prior by maximizing the tuning_metric on the validation set.
        Default is False.
    n_trials : int, optional
        Number of Optuna trials for hyperparameter tuning. Default is 20.
    tuning_metric : str, optional
        Metric to maximize during tuning: 'auc' or 'tpr'. Default is 'auc'.
    _local : bool
        Flag indicating local vs HPC execution. Not user-provided.
    """
    # pylint: disable=unused-argument

    if _local:
        remove_streamhandlers(LOGGER.parent)

    config = LearnWeightsConfig(
        grid=grid,
        labels=labels,
        attributes=attributes,
        exclude_attributes=exclude_attributes,
        n_estimators=n_estimators,
        class_prior=class_prior,
        background_samples=background_samples,
        test_size=test_size,
        validation_size=validation_size,
        n_jobs=n_jobs,
        random_state=random_state,
        score_name=score_name,
        crs=crs,
        tune=tune,
        n_trials=n_trials,
        tuning_metric=tuning_metric,
    )

    LOGGER.info("Running learn-weights pipeline...")
    results = run_learn_weights(config)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save score-weighted config
    config_out = out_path / "config_score_weighted.json"
    LOGGER.info(f"Saving score-weighted config to {config_out}")
    with open(config_out, "w") as f:
        json.dump(results["config"], f, indent=2)

    # Save metrics
    if results["metrics"]:
        metrics_out = out_path / "learn_weights_metrics.json"
        LOGGER.info(f"Saving metrics to {metrics_out}...")
        with open(metrics_out, "w") as f:
            json.dump(results["metrics"], f, indent=2)

    # Save tuning results
    if results.get("tuning"):
        tuning_out = out_path / "learn_weights_tuning.json"
        LOGGER.info(f"Saving tuning results to {tuning_out}...")
        with open(tuning_out, "w") as f:
            json.dump(results["tuning"], f, indent=2)

    LOGGER.info("learn-weights complete.")


learn_weights_cmd = CLICommandFromFunction(
    function=run,
    name="learn-weights",
    add_collect=False,
    config_preprocessor=_preprocessor,
)

main = as_click_command(learn_weights_cmd)


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        LOGGER.exception("Error running reVeal learn-weights command.")
        raise
