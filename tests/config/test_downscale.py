# -*- coding: utf-8 -*-
"""
config.characdownscale module tests
"""
import pytest

# from pydantic import ValidationError

from reVeal.config.downscale import BaseDownscaleConfig


@pytest.mark.parametrize(
    "baseline_year",
    [2020, 2023],
)
@pytest.mark.parametrize(
    "projection_resolution", ["regional", "total", "REGIONAL", "TOTAL"]
)
@pytest.mark.parametrize(
    "output_values", ["incremental", "cumulative", "INCREMENTAL", "CUMULATIVE"]
)
def test_basedownscaleconfig_valid_inputs(
    data_dir, baseline_year, projection_resolution, output_values
):
    """
    Test that BaseDownsaleConfig can be instantiated with valid inputs.
    """

    grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    load_projections = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )
    config = {
        "grid": grid,
        "grid_priority": "suitability_score",
        "grid_baseline_load": "dc_capacity_mw_existing",
        "baseline_year": baseline_year,
        "load_projections": load_projections,
        "projection_resolution": projection_resolution,
        "load_value": "dc_load_gw",
        "load_year": "year",
        "output_values": output_values,
    }

    BaseDownscaleConfig(**config)


# add tests for validate_load_growth validation errors:
# bad CSV format
# bad file format
# missing columns
# non-numeric columns
# invalid baseline year

# for grid dataset:
# bad format - not geospatial??? - mabye already checked in base grid class?
# TODO: check that the grid_priority and grid_load columns exist and are
# numeric

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
