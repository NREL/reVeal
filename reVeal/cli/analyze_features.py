# -*- coding: utf-8 -*-
"""
cli.analyze_features module - Sets up analyze-features command for use with
NLR-GAPs CLI.
"""
import json
import logging
from pathlib import Path

import geopandas as gpd
from pydantic import ValidationError
from gaps.cli import as_click_command, CLICommandFromFunction

from reVeal.config.analyze_features import AnalyzeFeaturesConfig
from reVeal.feature_analysis import (
    compute_correlation_matrix,
    compute_feature_clusters,
    save_analysis_outputs,
    suggest_exclusions,
)
from reVeal.log import get_logger, remove_streamhandlers

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
    log_directory : Path
        Path to log directory.
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
        af_config = {
            k: config.get(k)
            for k in AnalyzeFeaturesConfig.model_fields.keys()
            if k in config
        }
        AnalyzeFeaturesConfig(**af_config)
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
    exclude_attributes=None,
    correlation_method="spearman",
    cluster_threshold=0.7,
    _local=True,
):
    """
    Analyze feature correlations and clusters in a normalized grid.

    Computes a correlation matrix, performs hierarchical clustering, generates
    dendrogram and heatmap plots, and suggests redundant features for exclusion.

    Parameters
    ----------
    grid : str
        Path to the normalized grid (output of ``reVeal normalize``). Must be a
        vector dataset readable by pyogrio with numeric ``*_score`` columns.
    out_dir : str
        Output directory. Analysis artifacts will be saved to an ``analysis/``
        subdirectory.
    attributes : list of str, optional
        List of column names from the grid to use as features. If not specified,
        all columns ending with ``_score`` are used automatically.
    exclude_attributes : list of str, optional
        Score columns to exclude from auto-detected features. All ``*_score``
        columns except those listed will be used. Mutually exclusive with
        ``attributes``.
    correlation_method : str, optional
        Correlation method: 'spearman' or 'pearson'. Default is 'spearman'.
    cluster_threshold : float, optional
        Distance threshold for hierarchical clustering. Lower values produce
        more clusters (features must be more similar to cluster together).
        Default is 0.7.
    _local : bool
        Flag indicating local vs HPC execution. Not user-provided.
    """
    # pylint: disable=unused-argument

    if _local:
        remove_streamhandlers(LOGGER.parent)

    def _progress(msg):
        """Print progress to stdout and log."""
        print(msg, flush=True)
        LOGGER.info(msg)

    config = AnalyzeFeaturesConfig(
        grid=grid,
        attributes=attributes,
        exclude_attributes=exclude_attributes,
        correlation_method=correlation_method,
        cluster_threshold=cluster_threshold,
    )

    # Read grid
    _progress(f"Reading grid from {config.grid}...")
    grid_df = gpd.read_file(config.grid, engine="pyogrio", use_arrow=True)
    grid_df.fillna(0, inplace=True)
    _progress(f"Grid loaded: {len(grid_df):,} cells, {len(grid_df.columns)} columns.")

    # Resolve attributes
    if config.attributes is not None:
        features = config.attributes
    elif config.exclude_attributes is not None:
        all_score_cols = [c for c in grid_df.columns if c.endswith("_score")]
        exclude_set = set(config.exclude_attributes)
        features = [c for c in all_score_cols if c not in exclude_set]
        LOGGER.info(
            f"Excluded {len(exclude_set)} attributes, "
            f"using {len(features)} of {len(all_score_cols)} score columns."
        )
    else:
        features = [c for c in grid_df.columns if c.endswith("_score")]
        LOGGER.info(f"Auto-detected {len(features)} score attributes.")

    if not features:
        raise ValueError(
            "No attributes found. Provide 'attributes' in the config or ensure "
            "the grid has columns ending with '_score'."
        )

    _progress(
        f"Starting feature analysis ({config.correlation_method} correlation, "
        f"{len(features)} features, {len(grid_df):,} grid cells)..."
    )

    # Compute correlation matrix
    X = grid_df[features].to_numpy()
    _progress(
        f"Computing {config.correlation_method} correlation matrix "
        f"({len(grid_df):,} samples x {len(features)} features)... "
        f"this may take a few minutes for large grids."
    )
    corr_matrix = compute_correlation_matrix(
        X, features, method=config.correlation_method
    )
    _progress("Correlation matrix computed. Computing feature clusters...")

    # Compute clusters
    cluster_result = compute_feature_clusters(
        corr_matrix, threshold=config.cluster_threshold
    )
    n_clusters = len(cluster_result["clusters"])
    _progress(f"Feature analysis complete. Found {n_clusters} clusters.")

    # Suggest exclusions
    suggested = suggest_exclusions(
        clusters=cluster_result["clusters"],
        corr_matrix=corr_matrix,
    )

    # Save outputs
    out_path = Path(out_dir)
    analysis_dir = out_path / "analysis"
    _progress(f"Saving analysis outputs to {analysis_dir}...")

    save_analysis_outputs(
        corr_matrix=corr_matrix,
        cluster_result=cluster_result,
        out_dir=analysis_dir,
    )

    exclusions_out = analysis_dir / "suggested_exclusions.json"
    with open(exclusions_out, "w") as f:
        json.dump(suggested, f, indent=2)

    _progress(f"Analysis complete. Outputs saved to {analysis_dir}")


analyze_features_cmd = CLICommandFromFunction(
    function=run,
    name="analyze-features",
    add_collect=False,
    config_preprocessor=_preprocessor,
)

main = as_click_command(analyze_features_cmd)


if __name__ == "__main__":
    try:
        main(obj={})
    except Exception:
        LOGGER.exception("Error running reVeal analyze-features command.")
        raise
