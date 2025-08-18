# -*- coding: utf-8 -*-
"""Tests for CLI"""
import pytest

from loci.cli.cli import main


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
    config_path = data_dir / "characterize" / "config.json"
    result = cli_runner.invoke(
        main,
        ["characterize", "-c", config_path.as_posix()],
    )
    assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
