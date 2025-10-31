# -*- coding: utf-8 -*-
"""Tests for CLI"""
import json

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import pytest

from reVeal.cli.cli import main


def test_main(cli_runner):
    """Test main() CLI command."""
    result = cli_runner.invoke(main, "--help")
    assert result.exit_code == 0, f"Command failed with error {result.exception}"


def test_characterize(
    cli_runner,
    tmp_path,
    data_dir,
):
    """
    Happy path test for the characterize command. Tests that it produces the expected
    outputs for known inputs.
    """
    in_config_path = data_dir / "characterize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["execution_control"] = {"max_workers": 2}
    config_data["data_dir"] = (data_dir / "characterize").as_posix()
    config_data["grid"] = (
        data_dir / "characterize" / "grids" / "grid_2.gpkg"
    ).as_posix()
    for attr in config_data["characterizations"]:
        config_data["characterizations"][attr].update({"parallel": True})

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["characterize", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 0, f"Command failed with error {result.exception}"

    out_gpkg = tmp_path / "grid_char.gpkg"
    assert out_gpkg.exists(), "Output grid not created."

    out_df = gpd.read_file(out_gpkg)

    expected_gpkg = data_dir / "characterize" / "outputs" / "grid_char.gpkg"
    expected_df = gpd.read_file(expected_gpkg)

    assert_geodataframe_equal(expected_df, out_df)

    logs = list((tmp_path / "logs").glob("*_characterize.log"))
    assert len(logs) > 0, "No logs were created"

    log = logs[0]
    with open(log, "r") as f:
        log_content = f.read()

    assert (
        "UserWarning: NAs encountered in results dataframe" in log_content
    ), "Expected warning messages were not found in log file."
    assert (
        "Running characterization for output column" in log_content
    ), "Expected progress messages were not found in log file."


def test_characterize_invalid_config(
    cli_runner,
    tmp_path,
    data_dir,
):
    """
    Check for sane error message in log when an invalid configuration is passed.
    """
    in_config_path = data_dir / "characterize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["data_dir"] = (data_dir / "characterize").as_posix()
    config_data["grid"] = (
        data_dir / "characterize" / "grids" / "not-a-grid.gpkg"
    ).as_posix()

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["characterize", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 1

    log_paths = list((tmp_path / "logs").glob("*.log"))
    if len(log_paths) == 0:
        raise ValueError("Logs were not created by command.")
    log_path = log_paths[0]
    with open(log_path, "r") as f:
        log = f.read()
    expected_contents = (
        "Configuration did not pass validation. The following issues were identified:\n"
        "1 validation error for CharacterizeConfig"
    )
    assert expected_contents in log, "Expected error message not found in log file"


def test_normalize(
    cli_runner,
    tmp_path,
    data_dir,
):
    """
    Happy path test for the normalize command. Tests that it produces the
    expected outputs for known inputs.
    """
    in_config_path = data_dir / "normalize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (
        data_dir / "characterize" / "outputs" / "grid_char.gpkg"
    ).as_posix()

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["normalize", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 0, f"Command failed with error {result.exception}"

    out_gpkg = tmp_path / "grid_normalized.gpkg"
    assert out_gpkg.exists(), "Output grid not created."

    out_df = gpd.read_file(out_gpkg)

    expected_gpkg = data_dir / "normalize" / "outputs" / "grid_normalized.gpkg"
    expected_df = gpd.read_file(expected_gpkg)

    assert_geodataframe_equal(expected_df, out_df)

    logs = list((tmp_path / "logs").glob("*_normalize.log"))
    assert len(logs) > 0, "No logs were created"

    log = logs[0]
    with open(log, "r") as f:
        log_content = f.read()

    assert (
        "UserWarning: gid column already exists in self.dataframe" in log_content
    ), "Expected overwrite warning message was not found in log file."
    assert (
        "UserWarning: NAs encountered in results dataframe" in log_content
    ), "Expected NA warning message was not found in log file."
    assert (
        "Running normalization for output column" in log_content
    ), "Expected progress messages were not found in log file."


def test_normalize_invalid_config(
    cli_runner,
    tmp_path,
    data_dir,
):
    """
    Check for sane error message in log when an invalid configuration is passed.
    """
    in_config_path = data_dir / "normalize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (
        data_dir / "characterize" / "outputs" / "not-a-grid.gpkg"
    ).as_posix()

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["normalize", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 1

    log_paths = list((tmp_path / "logs").glob("*.log"))
    if len(log_paths) == 0:
        raise ValueError("Logs were not created by command.")
    log_path = log_paths[0]
    with open(log_path, "r") as f:
        log = f.read()
    expected_contents = (
        "Configuration did not pass validation. The following issues were identified:\n"
        "1 validation error for NormalizeConfig\n"
        "grid\n"
        "  Path does not point to a file"
    )
    assert expected_contents in log, "Expected error message not found in log file"


def test_score_weighted(
    cli_runner,
    tmp_path,
    data_dir,
):
    """
    Happy path test for the score-weighted command. Tests that it produces the
    expected outputs for known inputs.
    """
    in_config_path = data_dir / "score_weighted" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (
        data_dir / "normalize" / "outputs" / "grid_normalized.gpkg"
    ).as_posix()

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["score-weighted", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 0, f"Command failed with error {result.exception}"

    out_gpkg = tmp_path / "grid_scores.gpkg"
    assert out_gpkg.exists(), "Output grid not created."

    out_df = gpd.read_file(out_gpkg)

    expected_gpkg = data_dir / "score_weighted" / "outputs" / "grid_scores.gpkg"
    expected_df = gpd.read_file(expected_gpkg)

    assert_geodataframe_equal(expected_df, out_df)

    logs = list((tmp_path / "logs").glob("*_score_weighted.log"))
    assert len(logs) > 0, "No logs were created"

    log = logs[0]
    with open(log, "r") as f:
        log_content = f.read()

    assert (
        "UserWarning: gid column already exists in self.dataframe" in log_content
    ), "Expected overwrite warning message was not found in log file."
    assert (
        "Calculating weighted scores" in log_content
    ), "Expected progress messages were not found in log file."


def test_score_weighted_invalid_config(
    cli_runner,
    tmp_path,
    data_dir,
):
    """
    Check for sane error message in log when an invalid configuration is passed.
    """
    in_config_path = data_dir / "score_weighted" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (
        data_dir / "normalize" / "outputs" / "grid_normalized.gpkg"
    ).as_posix()
    for attribute in config_data["attributes"]:
        attribute["weight"] = 0.1

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["score-weighted", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 1

    log_paths = list((tmp_path / "logs").glob("*.log"))
    if len(log_paths) == 0:
        raise ValueError("Logs were not created by command.")
    log_path = log_paths[0]
    with open(log_path, "r") as f:
        log = f.read()
    expected_contents = (
        "Configuration did not pass validation. The following issues were identified:\n"
        "1 validation error for ScoreWeightedConfig\n"
        "  Value error, Weights of input attributes must sum to 1"
    )
    assert expected_contents in log, "Expected error message not found in log file"


@pytest.mark.parametrize(
    "config",
    ["config_total.json", "config_regional.json", "config_region_weights.json"],
)
def test_downscale(
    cli_runner,
    tmp_path,
    data_dir,
    config,
):
    """
    Happy path test for the downscale command. Tests that it produces the
    expected outputs for known inputs.
    """
    in_config_path = data_dir / "downscale" / config
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (data_dir / config_data["grid"]).as_posix()
    config_data["load_projections"] = (
        data_dir / config_data["load_projections"]
    ).as_posix()
    if "regions" in config_data:
        config_data["regions"] = (data_dir / config_data["regions"]).as_posix()

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["downscale", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 0, f"Command failed with error {result.exception}"

    out_gpkg = tmp_path / "grid_load_projections.gpkg"
    assert out_gpkg.exists(), "Output grid not created."

    out_df = gpd.read_file(out_gpkg)

    expected_gpkg_stub = config.replace("config_", "").replace(".json", "")
    expected_gpkg = (
        data_dir
        / "downscale"
        / "outputs"
        / f"grid_downscaled_{expected_gpkg_stub}_year_cap.gpkg"
    )
    expected_df = gpd.read_file(expected_gpkg)

    assert_geodataframe_equal(expected_df, out_df, check_like=True)

    logs = list((tmp_path / "logs").glob("*_downscale.log"))
    assert len(logs) > 0, "No logs were created"

    log = logs[0]
    with open(log, "r") as f:
        log_content = f.read()

    assert (
        "UserWarning: gid column already exists in self.dataframe" in log_content
    ), "Expected overwrite warning message was not found in log file."
    assert (
        "Running downscaling..." in log_content
    ), "Expected progress messages were not found in log file."


def test_downscale_invalid_config(
    cli_runner,
    tmp_path,
    data_dir,
):
    """
    Check for sane error message in log when an invalid configuration is passed.
    """
    in_config_path = data_dir / "downscale" / "config_total.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)
    config_data["grid"] = (data_dir / config_data["grid"]).as_posix()
    config_data["load_projections"] = (
        data_dir / config_data["load_projections"]
    ).as_posix()
    config_data["load_value"] = "MW"

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    result = cli_runner.invoke(
        main,
        ["downscale", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 1

    log_paths = list((tmp_path / "logs").glob("*.log"))
    if len(log_paths) == 0:
        raise ValueError("Logs were not created by command.")
    log_path = log_paths[0]
    with open(log_path, "r") as f:
        log = f.read()
    expected_contents = (
        "Configuration did not pass validation. The following issues were identified:\n"
        "1 validation error for BaseDownscaleConfig\n"
        "  Value error, Specified attribute MW does not exist in the input"
    )
    assert expected_contents in log, "Expected error message not found in log file"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
