# -*- coding: utf-8 -*-
"""
cli.downscale module - Sets up downscale command for use with nrel-gaps CLI
"""
import logging
import json
from pathlib import Path

from pydantic import ValidationError
from gaps.cli import as_click_command, CLICommandFromFunction

from reVeal.config.downscale import DownscaleConfig
from reVeal.log import get_logger, remove_streamhandlers
from reVeal.grid import DownscaleGrid

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
        downscale_config = {
            k: config.get(k) for k in DownscaleConfig.model_fields.keys() if k in config
        }
        DownscaleConfig(**downscale_config)
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
    grid_priority,
    grid_baseline_load,
    baseline_year,
    projection_resolution,
    load_projections,
    load_value,
    load_year,
    out_dir,
    regions=None,
    region_names=None,
    load_regions=None,
    region_weights=None,
    max_workers=None,
    _local=True,
):
    """
    Downscale load projections to grid based on priority values.

    Outputs a new GeoPackage containing the input grid with added
    attributes for downscaled load by year.

    Parameters
    ----------
    grid : str
        Path to vector dataset for which attribute scoring will be performed.
        Must be an existing vector dataset in a format that can be opened by
        ``pyogrio``. Does not strictly need to be a grid, or even a polygon dataset,
        but must be a vector dataset.
    grid_priority : str
        Name of attribute column in ``grid`` dataset to use for prioritizing load
        downscaling.
    grid_baseline_load : str
        Name of attribute column in ``grid`` dataset containing values for baseline
        (i.e., starting) load in each grid cell in the corresponding ``baseline_year``.
    baseline_year : int
        Year corresponding to the baseline load values in the ``grid_baseline_load``
        column.
    projection_resolution : str
        Resolution of ``load_projections`` dataset. Refer to
        :obj:`reVeal.config.downscale.ProjectionResolutionEnum`.
    load_projections : str
        Path to ``load_projections`` dataset. Expected to be a CSV file.
    load_value : str
        Name of column containing load values in ``load_projections`` dataset to
        disaggregate.

        .. note::
            This value will also be used as the name for the column containing
            downscaled load values in the output GeoPackage.
    load_year : str
        Name of column in ``load_projections`` dataset containing year values.
    out_dir : str
        Output parent directory. Results will be saved to a file named
        "grid_load_projections.gpkg".
    regions : str, optional
        Path to vector dataset containing regions to use in disaggregation. Required
        if ``projections_resolution == "regional"``.
    region_names : str, optional
        Name of attribute column containing the name or identifier of regions in the
        ``regions`` dataset.
    load_regions : str, optional
        Name of column in ``load_projections`` dataset containing region names, if
        applicable. Specify this option when the input ``load_projections`` are
        resolved to the regional level. Values in this column should match values in
        the ``region_names`` column of the ``regions`` dataset.

        .. note::
            If ``projection_resolution == "regional"``, either this option or
            ``region_weights``, but not both, must be specified.
    region_weights : dict, optional
        Dictionary indicating weights to use for apportioning load to regions before
        disaggregating. Keys should match values in the ``region_names`` column of
        the ``regions`` dataset.

        .. note::
            If ``projection_resolution == "regional"``, either this option or
            ``load_regions``, but not both, must be specified.
    max_workers : [int, NoneType], optional
        Maximum number of workers to use for multiprocessing when running downscaling.
        By default None, will use all available workers.
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

    config = DownscaleConfig(
        grid=grid,
        grid_priority=grid_priority,
        grid_baseline_load=grid_baseline_load,
        baseline_year=baseline_year,
        projection_resolution=projection_resolution,
        load_projections=load_projections,
        load_value=load_value,
        load_year=load_year,
        projection_resolution=projection_resolution,
        regions=regions,
        region_names=region_names,
        out_dir=out_dir,
        load_regions=load_regions,
        region_weights=region_weights,
    )

    LOGGER.info("Initializing DownscaleGrid from input config...")
    downscale_grid = DownscaleGrid(config)
    LOGGER.info("Initialization complete.")

    LOGGER.info("Downscaling laod projections...")
    out_grid_df = downscale_grid.run()
    LOGGER.info("Downscaling complete.")

    out_gpkg = Path(out_dir).joinpath("grid_load_projections.gpkg").expanduser()
    LOGGER.info(f"Saving results to {out_gpkg}...")
    out_grid_df.to_file(out_gpkg)
    LOGGER.info("Saving complete.")


downscale_cmd = CLICommandFromFunction(
    function=run,
    name="downscale",
    add_collect=False,
    config_preprocessor=_preprocessor,
)

main = as_click_command(downscale_cmd)


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        LOGGER.exception("Error running reVeal downscale command.")
        raise
