# -*- coding: utf-8 -*-
"""
cli.score_weighted module - Sets up score-attributes command for use with nrel-gaps
CLI
"""
import logging
import json
from pathlib import Path

from pydantic import ValidationError
from gaps.cli import as_click_command, CLICommandFromFunction

from reVeal.config.score_weighted import ScoreWeightedConfig
from reVeal.log import get_logger, remove_streamhandlers
from reVeal.grid import ScoreWeightedGrid

LOGGER = logging.getLogger(__name__)


def _log_inputs(config):
    """
    Emit log messages summarizing user inputs.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    LOGGER.info(f"Inputs config: {json.dumps(config, indent=4)}")


def _preprocessor(config, job_name, log_directory, verbose):
    """
    Preprocess user-input configuration.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    job_name : str
        Name of `job being run. Derived from the name of the folder containing the
        user configuration file.
    verbose : bool
        Flag to signal ``DEBUG`` verbosity (``verbose=True``).

    Returns
    -------
    dict
        Configuration dictionary modified to include additional or augmented
        parameters.
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
        score_config = {
            k: config.get(k)
            for k in ScoreWeightedConfig.model_fields.keys()
            if k in config
        }
        ScoreWeightedConfig(**score_config)
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
    attributes,
    score_name,
    out_dir,
    _local=True,
):
    """
    Convert specified attribute values of input grid to a scale of 0 to 1 using the
    specified method(s). Outputs a new GeoPackage containing the input grid with added
    attributes for scored attributes.

    Parameters
    ----------
    grid : str
        Path to vector dataset for which attribute scoring will be performed.
        Must be an existing vector dataset in a format that can be opened by pyogrio.
        Does not strictly need to be a grid, or even a polygon dataset, but must be
        a vector dataset.
    attributes: list
        List of dictionaries, each specifying the name of an attribute to be included
        in the composite weighted score, and a corresponding weight. Each dictionary
        should have the following keys:
            - "attribute": String indicating the name of the attribute to include in
                the composite weight.
            - "weight": Float in the range of >0, <=1 indicating the weight to apply to
             the attribute in the composite weighted score.
        Note that all weights must sum to 1.
    score_name : str
        Name of the output column in which the resulting weighted scores will be
        stored. If this column exists in the input grid, it will be overwritten.
    out_dir : str
        Output parent directory. Results will be saved to a file named
        "grid_char_weighted_scores.gpkg".
    _local : bool
        Flag indicating whether the code is being run locally or via HPC job
        submissions. NOTE: This is not a user provided parameter - it is determined
        dynamically by based on whether config["execution_control"]["option"] == "local"
        (defaults to True if not specified).
    """
    # pylint: disable=unused-argument

    # streamhandler is added in by gaps before kicking off the subprocess and
    # will produce duplicate log messages if running locally, so remove it
    if _local:
        remove_streamhandlers(LOGGER.parent)

    config = ScoreWeightedConfig(
        grid=grid,
        attributes=attributes,
        score_name=score_name,
    )

    LOGGER.info("Initializing ScoreWeightedGrid from input config...")
    score_weighted_grid = ScoreWeightedGrid(config)
    LOGGER.info("Initialization complete.")

    LOGGER.info("Calculating weighted scores...")
    out_grid_df = score_weighted_grid.run()
    LOGGER.info("Weighted scoring complete.")

    out_gpkg = Path(out_dir).joinpath("grid_char_weighted_scores.gpkg").expanduser()
    LOGGER.info(f"Saving results to {out_gpkg}...")
    out_grid_df.to_file(out_gpkg)
    LOGGER.info("Saving complete.")


score_weighted_cmd = CLICommandFromFunction(
    function=run,
    name="score-weighted",
    add_collect=False,
    config_preprocessor=_preprocessor,
)
main = as_click_command(score_weighted_cmd)


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        LOGGER.exception("Error running reVeal score-attributes command.")
        raise
