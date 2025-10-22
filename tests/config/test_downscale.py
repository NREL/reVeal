# -*- coding: utf-8 -*-
"""
config.characdownscale module tests
"""
from csv import QUOTE_NONNUMERIC

import pytest
import geopandas as gpd
import pandas as pd
from pydantic import ValidationError

from reVeal.config.downscale import (
    BaseDownscaleConfig,
    TotalDownscaleConfig,
    RegionalDownscaleConfig,
)
from reVeal.errors import CSVReadError, FileFormatError


@pytest.mark.parametrize(
    "baseline_year",
    [2020, 2023],
)
@pytest.mark.parametrize(
    "projection_resolution", ["regional", "total", "REGIONAL", "TOTAL"]
)
def test_basedownscaleconfig_valid_inputs(
    data_dir, baseline_year, projection_resolution
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
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    BaseDownscaleConfig(**config)


@pytest.mark.parametrize(
    "update_parameters",
    [
        {"grid_priority": "best_site_score"},
        {"grid_baseline_load": "existing_mw"},
        {"load_value": "dc_load_gw"},
        {"load_year": "yr"},
    ],
)
def test_validate_missing_attribute(data_dir, update_parameters):
    """
    Test that BaseDownscaleConfig raises a ValueError when a non-existent column is
    specified for required grid or load_projections columns.
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
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }
    config.update(update_parameters)

    with pytest.raises(ValueError, match="Specified attribute .* does not exist"):
        BaseDownscaleConfig(**config)


@pytest.mark.parametrize("test_col", ["suitability_score", "dc_capacity_mw_existing"])
def test_validate_grid_nonnumeric_attribute(data_dir, tmp_path, test_col):
    """
    Test that BaseDownscaleConfig raises a ValueError when a non-numeric column is
    specified for either the grid_priority or grid_baseline_load columns.
    """

    src_grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    grid_df = gpd.read_file(src_grid)
    grid_df[test_col] = grid_df[test_col].astype(str)
    grid = tmp_path / "grid.gpkg"
    grid_df.to_file(grid)

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
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    with pytest.raises(ValueError, match="Specified grid attribute .* must be numeric"):
        BaseDownscaleConfig(**config)


def test_validate_load_projections_fileformaterror(data_dir, tmp_path):
    """
    Test that BaseDownscaleConfig raises a FileFormatError when passed an input
    load_projections file that is not a valid CSV file.
    """

    grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    load_projections = tmp_path / "load.csv"
    with open(load_projections, "w") as dst:
        dst.write(
            "Hello!\n"
            "This is a text file that I made for tests, what do you think?\n"
            "If I'm correct, I think that this needs 3 lines, at least, to fail."
        )

    config = {
        "grid": grid,
        "grid_priority": "suitability_score",
        "grid_baseline_load": "dc_capacity_mw_existing",
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    with pytest.raises(FileFormatError, match="Unable to parse text as CSV."):
        BaseDownscaleConfig(**config)


def test_validate_load_projections_csvreaderror(data_dir, tmp_path):
    """
    Test that BaseDownscaleConfig raises a CSVReadError when passed an input
    load_projections file that is not encoded in utf-8.
    """

    grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    src_projections = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )
    load_projections = tmp_path / "load.csv"
    load_df = pd.read_csv(src_projections)
    load_df.to_csv(load_projections, encoding="utf-16")

    config = {
        "grid": grid,
        "grid_priority": "suitability_score",
        "grid_baseline_load": "dc_capacity_mw_existing",
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    with pytest.raises(CSVReadError, match="Unable to parse input as 'utf-8' text"):
        BaseDownscaleConfig(**config)


@pytest.mark.parametrize("test_col", ["dc_load_mw", "year"])
def test_validate_load_projections_nonnumeric_attribute(data_dir, tmp_path, test_col):
    """
    Test that BaseDownscaleConfig raises a ValueError when a non-numeric column is
    specified for either the grid_priority or grid_baseline_load columns.
    """

    grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    src_projections = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )

    load_df = pd.read_csv(src_projections)
    load_df[test_col] = "a"
    load_projections = tmp_path / "projections.csv"
    load_df.to_csv(load_projections, header=True, index=False, quoting=QUOTE_NONNUMERIC)

    config = {
        "grid": grid,
        "grid_priority": "suitability_score",
        "grid_baseline_load": "dc_capacity_mw_existing",
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    with pytest.raises(
        ValueError, match="Specified load_projections attribute .* must be numeric"
    ):
        BaseDownscaleConfig(**config)


def test_validate_load_projections_predates_baseline_error(data_dir):
    """
    Test that BaseDownsaleConfig raises a ValueError when the load projections predate
    the baseline load year.
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
        "baseline_year": 2030,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    with pytest.raises(ValueError, match="First year in load_projections .* predates"):
        BaseDownscaleConfig(**config)


@pytest.mark.parametrize(
    "baseline_year",
    [2020, 2023],
)
@pytest.mark.parametrize("projection_resolution", ["total", "TOTAL"])
def test_totaldownscaleconfig_valid_inputs(
    data_dir, baseline_year, projection_resolution
):
    """
    Test that TotalDownsaleConfig can be instantiated with valid inputs.
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
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    TotalDownscaleConfig(**config)


def test_totaldownscaleconfig_resolution_error(data_dir):
    """
    Test that TotalDownsaleConfig raises a validation error when passed "total" as
    the projection_resolution instead of "regional".
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
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "regional",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    with pytest.raises(ValidationError, match="Input should be 'total"):
        TotalDownscaleConfig(**config)


def test_validate_load_projections_duplicate_years(data_dir, tmp_path):
    """
    Test that TotalDownscaleConfig raises a ValueError when there are duplicate
    years in the load_projections dataset.
    """

    grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    src_projections = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )

    load_df = pd.read_csv(src_projections)
    new_load_df = pd.concat([load_df.iloc[0:1], load_df], ignore_index=True)
    load_projections = tmp_path / "projections.csv"
    new_load_df.to_csv(load_projections, header=True, index=False)

    config = {
        "grid": grid,
        "grid_priority": "suitability_score",
        "grid_baseline_load": "dc_capacity_mw_existing",
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
    }

    with pytest.raises(
        ValueError, match="Input load_projections dataset has duplicate years"
    ):
        TotalDownscaleConfig(**config)


@pytest.mark.parametrize(
    "baseline_year",
    [2020, 2023],
)
@pytest.mark.parametrize("projection_resolution", ["regional", "REGIONAL"])
@pytest.mark.parametrize("regions_ext", ["gpkg", "parquet"])
def test_regionaldownscaleconfig_valid_inputs_load_regions(
    data_dir, baseline_year, projection_resolution, regions_ext
):
    """
    Test that RegionalDownsaleConfig can be instantiated with valid inputs, including
    regional load projections (specified via the load_regions parameter).
    """

    grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    load_projections = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_regional.csv"
    )
    regions = (
        data_dir / "downscale" / "inputs" / "regions" / f"eer_adp_zones.{regions_ext}"
    )
    config = {
        "grid": grid,
        "grid_priority": "suitability_score",
        "grid_baseline_load": "dc_capacity_mw_existing",
        "baseline_year": baseline_year,
        "load_projections": load_projections,
        "projection_resolution": projection_resolution,
        "load_value": "dc_load_mw",
        "load_year": "year",
        "load_regions": "zone",
        "regions": regions,
        "region_names": "emm_zone",
    }

    downscale_config = RegionalDownscaleConfig(**config)
    assert downscale_config.regions_ext == f".{regions_ext}"
    expected_regions_flavor = "ogr" if regions_ext == "gpkg" else "geoparquet"
    assert downscale_config.regions_flavor == expected_regions_flavor


def test_regionaldownscaleconfig_resolution_error(data_dir):
    """
    Test that RegionalDownsaleConfig raises a validation error when passed "regional"
    as the projection_resolution instead of "total".
    """

    grid = data_dir / "downscale" / "inputs" / "grid_char_weighted_scores.gpkg"
    load_projections = (
        data_dir
        / "downscale"
        / "inputs"
        / "load_growth_projections"
        / "eer_us-adp-2024-central_national.csv"
    )
    regions = data_dir / "downscale" / "inputs" / "regions" / "eer_adp_zones.gpkg"
    config = {
        "grid": grid,
        "grid_priority": "suitability_score",
        "grid_baseline_load": "dc_capacity_mw_existing",
        "baseline_year": 2022,
        "load_projections": load_projections,
        "projection_resolution": "total",
        "load_value": "dc_load_mw",
        "load_year": "year",
        "load_regions": "zone",
        "regions": regions,
        "region_names": "emm_zone",
    }

    with pytest.raises(ValidationError, match="Input should be 'regional"):
        RegionalDownscaleConfig(**config)


# TODO:
# Tests for RegionalDownscaleConfig:
# raises errors under validate_regions for each of:
#        1. non-polygon geometries
#        2. missing region_names column
#        3. CRS doesn't match grid CRS


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
