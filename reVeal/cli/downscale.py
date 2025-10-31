# -*- coding: utf-8 -*-
"""
cli.downscale module - Sets up downscale command for use with nrel-gaps CLI
"""
import logging
import json
from pathlib import Path

from pydantic import ValidationError
from gaps.cli import as_click_command, CLICommandFromFunction

from reVeal.config.downscale import (
    DownscaleConfig,
    TotalDownscaleConfig,
    RegionalDownscaleConfig,
)
from reVeal.log import get_logger, remove_streamhandlers
from reVeal.grid import TotalDownscaleGrid, RegionalDownscaleGrid

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
        candidate_keys = list(TotalDownscaleConfig.model_fields.keys()) + list(
            RegionalDownscaleConfig.model_fields.keys()
        )
        downscale_config = {k: config.get(k) for k in candidate_keys if k in config}
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
    grid_capacity,
    projection_resolution,
    load_projections,
    load_value,
    load_year,
    out_dir,
    regions=None,
    region_names=None,
    load_regions=None,
    region_weights=None,
    max_site_addition_per_year=None,
    site_saturation_limit=1,
    priority_power=1,
    n_bootstraps=10_000,
    random_seed=0,
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
    grid_capacity : str
        Name of attribute column in ``grid`` dataset indicating the developable
        capacity of load within each site.
    projection_resolution : str
        Resolution of ``load_projections`` dataset. Refer to
        :obj:`reVeal.config.downscale.ProjectionResolutionEnum`.
    load_projections : str
        Path to ``load_projections`` dataset. Expected to be a CSV file.
    load_value : str
        Name of column containing load values in ``load_projections`` dataset to
        disaggregate.

        .. important::
            The projected values of load found in this column are expected to be and
            will be treated as incremental additions of load in each year, NOT
            cumulative values.

        .. note::
            This value will be used as the name the columns containing downscaled load
            values in the output grid GeoPackage.
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
        the ``regions`` dataset. Values should indicate the proportion of aggregate
        load to apportion to the corresponding region. Values must sum to 1.

        .. note::
            If ``projection_resolution == "regional"``, either this option or
            ``load_regions``, but not both, must be specified.
    max_site_addition_per_year : float, optional
        Value indicating the maximum allowable increment of load that can be added in
        a given year to an individual site. The default value is None, which will not
        apply a cap. This value can be used to ensure that the rate of expansion of
        large load capacity in localized areas is not unrealistically rapid. Using
        this parameter can also have the effect of achieving greater geographic
        dispersion of load: since there is a limit to the pace at which individual
        sites can build out load, more sites are typically required for the same amount
        of project load.
    site_saturation_limit : float, optional
        Adjustment factor limit the developable capacity of load within each site.
        This value is used to scale the values in the ``grid_capacity``. For
        example, to limit the maximum deployed load in each site to half of the
        actual developable load, use ``site_saturation_limit=0.5``. The lower this
        value is set, the greater the degree of dispersion of load across sites will
        be. The dfault is 1, which leaves the values in the ``grid_capacity``
        unmodified.
    priority_power : int, optional
        This factor can be used to exaggerate the influence of the values in
        ``grid_priority``, such that higher values have an increased likelihood of
        load deployment and lower values have a decreased likelihood. This effect is
        implemented by raising the values in ``grid_priority`` to the specified
        ``priority_power``. As a result, if the input  values in ``grid_priority``
        are < 1, setting ``priority_power`` to high values can result in completely
        eliminating lower priority sites from consideration. The default value is 1,
        which leaves the values in ``grid_priority`` unmodified. To achieve
        less dispersion and greater clustering of downscaled load in higher priority
        sites, increase this value.
    n_bootstraps : int, optional
        Number of bootstraps to simulate in each projection year. Default is 10,000.
        In general, larger values will produce more stable results, with less chance
        for lower priority sites to receive large amounts of deployed load. However,
        larger values will also cause longer run times.
    random_seed : int, optional
        Random seed to use for reproducible bootstrapping. Default is 0. In general,
        this value does not need to be modified. The exception is if you are interested
        in testing sensitivities and/or producing multiple realizations or scenarios of
        deployment for a given set of values in ``load_priority``.
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
        grid_capacity=grid_capacity,
        projection_resolution=projection_resolution,
        load_projections=load_projections,
        load_value=load_value,
        load_year=load_year,
        regions=regions,
        region_names=region_names,
        out_dir=out_dir,
        load_regions=load_regions,
        region_weights=region_weights,
        max_site_addition_per_year=max_site_addition_per_year,
        site_saturation_limit=site_saturation_limit,
        priority_power=priority_power,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
    )

    if max_workers is not None:
        if config.max_workers is None:
            config.max_workers = max_workers

    if isinstance(config, TotalDownscaleConfig):
        LOGGER.info("Initializing TotalDownscaleConfig from input config...")
        downscale_grid = TotalDownscaleGrid(config)
    elif isinstance(config, RegionalDownscaleConfig):
        LOGGER.info("Initializing RegionalDownscaleGrid from input config...")
        downscale_grid = RegionalDownscaleGrid(config)
    else:
        raise TypeError(
            f"Unexpected type of config: {type(config)}. Must be one of the following: "
            "[TotalDownscaleConfig, RegionalDownscaleConfig]"
        )
    LOGGER.info("Initialization complete.")

    LOGGER.info("Running downscaling...")
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
