# -*- coding: utf-8 -*-
"""
cli.normalize module - Sets up normalize command for use with nrel-gaps CLI
"""
import logging
import json
from pathlib import Path

from pydantic import ValidationError
from gaps.cli import as_click_command, CLICommandFromFunction

from reVeal.config.normalize import NormalizeConfig
from reVeal.log import get_logger, remove_streamhandlers
from reVeal.grid import NormalizeGrid

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
        normalize_config = {
            k: config.get(k) for k in NormalizeConfig.model_fields.keys() if k in config
        }
        NormalizeConfig(**normalize_config)
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
    out_dir,
    attributes=None,
    normalize_method=None,
    invert=False,
    _local=True,
):
    """
    Convert specified attribute values of input grid to a scale of 0 to 1 using the
    specified method(s). Outputs a new GeoPackage containing the input grid with added
    attributes for normalized attributes.

    Parameters
    ----------
    grid : str
        Path to vector dataset for which attribute normalization will be performed.
        Must be an existing vector dataset in a format that can be opened by pyogrio.
        Does not strictly need to be a grid, or even a polygon dataset, but must be
        a vector dataset.
    out_dir : str
        Output parent directory. Results will be saved to a file named
        "grid_char_norm.gpkg".
    attributes: dict, optional
        Attributes to be normalized. Must be a dictionary keyed by the name of
        the output column for each normalized attribute. Each value must be another
        dictionary with the following keys:
            - "attribute": String indicating the name of the attribute to normalize.
            - "normalize_method": Method to use for normalizing the attribute to a
                scale from 0 to 1. Refer to :obj:
            - "invert": Boolean option. If specified as True, normalized values will be
                inverted such that low values will be closer to 1, and higher values
                closer to 0. Default is False, under which values are normalized with
                low values closer to 0 and high values closer to 1.
        If ``attributes`` is not specified, ``normalize_method`` must be provided.
    normalize_method : str, optional
        Optional default method to be used for normalization. If specified, this method
        will be applied to all numeric attributes in the input grid that are not
        specified separately in the input ``attributes``. Each output column will be
        named based on the corresponding input column plus a suffix "_score". If
        ``normalize_method`` is not specified, ``attributes`` must be provided
    invert : bool, optional
        If specified as True and ``normalize_method`` is provided, all attributes not
        specified separately in ``attributes`` will be normalized with values inverted
        (i.e., low values will be closer to 1, and higher values closer to 0). Default
        is False, under which values are normalized with low values closer to 0 and
        high values closer to 1. Note that this parameter will have no effect if
        ``normalize_method`` is not specified.
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

    if attributes is None:
        attributes = {}

    config = NormalizeConfig(
        grid=grid,
        attributes=attributes,
        normalize_method=normalize_method,
        invert=invert,
    )

    LOGGER.info("Initializing NormalizeGrid from input config...")
    normalized_grid = NormalizeGrid(config)
    LOGGER.info("Initialization complete.")

    LOGGER.info("Running attribute normalization...")
    out_grid_df = normalized_grid.run()
    LOGGER.info("Attribute normalization complete.")

    out_gpkg = Path(out_dir).joinpath("grid_char_norm.gpkg").expanduser()
    LOGGER.info(f"Saving results to {out_gpkg}...")
    out_grid_df.to_file(out_gpkg)
    LOGGER.info("Saving complete.")


normalize_cmd = CLICommandFromFunction(
    function=run,
    name="normalize",
    add_collect=False,
    config_preprocessor=_preprocessor,
)
main = as_click_command(normalize_cmd)


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        LOGGER.exception("Error running reVeal normalize command.")
        raise
