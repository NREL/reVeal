# -*- coding: utf-8 -*-
"""cli.characterize module - Sets up characterize command for use with nrel-gaps CLI"""
import logging
import json
from pathlib import Path

from pydantic import ValidationError
from gaps.cli import as_click_command, CLICommandFromFunction

from reVeal.config.characterize import CharacterizeConfig
from reVeal.log import get_logger, remove_streamhandlers
from reVeal.grid import CharacterizeGrid

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
        char_config = {
            k: config.get(k)
            for k in CharacterizeConfig.model_fields.keys()
            if k in config
        }
        CharacterizeConfig(**char_config)
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
    data_dir,
    grid,
    characterizations,
    expressions,
    out_dir,
    max_workers=None,
    _local=True,
):
    """
    Characterize a vector grid based on specified raster and vector datasets.
    Outputs a new GeoPackage containing the input grid with added attributes for the
    user-specified characterizations.

    Parameters
    ----------
    data_dir : str
        Path to parent directory containing all geospatial raster and vector datasets
        to be used for grid characterization.
    grid : str
        Path to gridded vector dataset for which characterization will be performed.
        Must be an existing vector polygon dataset in a format that can be opened by
        pyogrio. Does not strictly need to be a grid, but some functionality may
        not work if it is not.
    characterizations: dict
        Characterizations to be performed. Must be a dictionary keyed by the name of
        the output attribute for each characterization. Each value must be another
        dictionary with the following keys:
            - "dset": String indicating relative path within data_dir to dataset to be
                characterized.
            - "method": String indicating characterization method to be performed.
                Refer to
                :obj:`reVeal.config.characterize.VALID_CHARACTERIZATION_METHODS`.
            - "attribute": Attribute to summarize. Only required for certain methods.
                Default is None/null.
            - "weights_dset": String indicating relative path within data_dir to
                dataset to be used as weights. Only applies to characterization
                methods for rasters; ignored otherwise.
            - "neighbor_order": Integer indicating the order of neighbors to include
                in the characterization of each grid cell. For example,
                neighbor_order=1 would result in included first-order queen's case
                neighbors. Optional, default is 0 which does not include neighbors.
            - "buffer_distance": Float indicating buffer distance to apply in the
                characterization of each grid cell. Units are based on the CRS of the
                input grid dataset. For instance, a value of 500 in CRS EPGS:5070
                would apply a buffer of 500m to each grid cell before characterization.
                Optional, default is 0 which does not apply a buffer.
            - "parallel": Boolean indicating whether to run the characterization in
                parallel. This method is only applicable to methods specified as
                "supports_parallel" in
                :obj:`reVeal.config.VALID_CHARACTERIZATION_METHODS`. Default is True,
                which will run applicable method in parallel and have no effect for
                other methods. This value should only be changed to False for small
                input grids, where the performance overhead of setting up parallel
                processing will outweigh the speedup of running operations in parallel.
                As a general rule of thumb, as long as the number of grid cells in your
                grid is an order of magnitude larger than the number of cores
                available, using ```parallel=True``` should yield improved performance.
    expressions: dict
        Additional expressions to be calculated. Must be a dictionary keyes by the name
        of the output attribute for each expression. Each value must be a string
        indicating the expression to be calculated. Expression strings can reference
        one or more attributes/keys referenced in the characterizations dictionary.
    out_dir : str
        Output parent directory. Results will be saved to a file named "grid_char.gpkg".
    max_workers : [int, NoneType], optional
        Maximum number of workers to use for multiprocessing, by default None, which
        uses all available CPUs.
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

    config = CharacterizeConfig(
        data_dir=data_dir,
        grid=grid,
        characterizations=characterizations,
        expressions=expressions,
    )

    LOGGER.info("Initializing CharacterizeGrid from input config.")
    characterize_grid = CharacterizeGrid(config)
    LOGGER.info("Initialization complete.")

    LOGGER.info("Running grid characterization")
    out_grid_df = characterize_grid.run()
    LOGGER.info("Grid characterization complete.")

    out_gpkg = Path(out_dir).joinpath("grid_char.gpkg").expanduser()
    LOGGER.info(f"Saving results to {out_gpkg}.")
    out_grid_df.to_file(out_gpkg)
    LOGGER.info("Saving complete.")


characterize_cmd = CLICommandFromFunction(
    function=run,
    name="characterize",
    add_collect=False,
    config_preprocessor=_preprocessor,
)
main = as_click_command(characterize_cmd)


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        LOGGER.exception("Error running reVeal characterize command.")
        raise
