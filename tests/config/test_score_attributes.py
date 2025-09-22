# -*- coding: utf-8 -*-
"""
config.score_attributes module tests
"""
import pytest

import geopandas as gpd
from pydantic import ValidationError

from reVeal.config.score_attributes import (
    AttributeScoringMethodEnum,
    Attribute,
    ScoreAttributesConfig,
    GRID_IDX,
)


@pytest.mark.parametrize(
    "value,error_expected",
    [
        ("minmax", False),
        ("percentile", False),
        ("MINMAX", False),
        ("PERCENTILE", False),
        ("MinMax", False),
        ("PerCenTile", False),
        ("min-max", True),
        ("percentiles", True),
    ],
)
def test_attributescoringmethodenum(value, error_expected):
    """
    Test for AttributeScoringMethodEnum.
    """
    if error_expected:
        with pytest.raises(ValueError):
            AttributeScoringMethodEnum(value)
    else:
        AttributeScoringMethodEnum(value)


@pytest.mark.parametrize(
    "dset,attribute,score_method,invert",
    [
        ("characterize/outputs/grid_char.gpkg", "tline_length", "minmax", None),
        (
            "characterize/outputs/grid_char.gpkg",
            "fttp_average_speed",
            "percentile",
            True,
        ),
        ("characterize/outputs/grid_char.parquet", "generator_mwh", "minmax", False),
        (
            "characterize/outputs/grid_char.parquet",
            "developable_area",
            "percentile",
            True,
        ),
    ],
)
def test_attribute_valid_inputs(data_dir, attribute, score_method, invert, dset):
    """
    Test Attribute class with valid inputs and make sure dynamic attributes are set.
    """
    dset_src = data_dir / dset
    value = {
        "attribute": attribute,
        "score_method": score_method,
        "dset_src": dset_src,
        "invert": invert,
    }
    if invert is None:
        value.pop("invert")
    attribute_model = Attribute(**value)

    # check dynamic attributes are set
    assert attribute_model.dset_ext is not None, "dset_ext not set"
    assert attribute_model.dset_flavor is not None, "dset_flavor not set"
    if not invert:
        assert attribute_model.invert is False, "Unexpected value for invert"
    else:
        assert attribute_model.invert is True, "Unexpected value for invert"


@pytest.mark.parametrize(
    "attribute", ["generator_mwhs", "not-an-attribute", "some_value"]
)
def test_attribute_missing_attributes(data_dir, attribute):
    """
    Test that Attribute validation raises a ValueError when passed a non-existent
    attribute.
    """
    dset_src = data_dir / "characterize/outputs/grid_char.gpkg"
    value = {"attribute": attribute, "score_method": "minmax", "dset_src": dset_src}
    with pytest.raises(ValueError, match=f"Attribute {attribute} not found in"):
        Attribute(**value)


def test_attribute_nonnumeric_attributes(tmp_path, data_dir):
    """
    Test that Attribute validation raises a TypeError when passed a non-numeric
    attribute.
    """
    dset_raw_src = data_dir / "characterize/outputs/grid_char.gpkg"
    df = gpd.read_file(dset_raw_src)
    df["new_value"] = "foo"
    dset_src = tmp_path / "grid_char_mod.gpkg"
    df.to_file(dset_src)

    value = {"attribute": "new_value", "score_method": "minmax", "dset_src": dset_src}
    with pytest.raises(TypeError, match="Must be a numeric dtype"):
        Attribute(**value)


def test_attribute_nonexistent_dset():
    """
    Test that Attribute validation raises a ValidationError when passed a non-existent
    input dset_src.
    """

    value = {
        "attribute": "generator_mwh",
        "score_method": "minmax",
        "dset_src": "not-a-file.gpkg",
    }
    with pytest.raises(ValidationError, match="Path does not point to a file"):
        Attribute(**value)


def test_attribute_bad_method(data_dir):
    """
    Test that Attribute validation raises a ValidationError when passed an invalid
    score_method method
    """
    dset_src = data_dir / "characterize/outputs/grid_char.gpkg"

    value = {
        "attribute": "generator_mwh",
        "score_method": "magic",
        "dset_src": dset_src,
    }
    with pytest.raises(
        ValidationError, match="Input should be 'percentile' or 'minmax'"
    ):
        Attribute(**value)


def test_scoreattributesconfig_valid_inputs(data_dir):
    """
    Test that ScoreAttributesConfig builds successfully with valid inputs and check
    that dynamically derived properties are set.
    """

    grid = data_dir / "characterize/outputs/grid_char.gpkg"
    attributes = {
        "generator_mwh_score": {"attribute": "generator_mwh", "score_method": "minmax"},
        "fttp_average_speed_score": {
            "attribute": "fttp_average_speed",
            "score_method": "percentile",
            "invert": True,
        },
    }
    config_data = {
        "grid": grid,
        "attributes": attributes,
    }
    config = ScoreAttributesConfig(**config_data)

    # check dynamic attributes are set
    assert config.grid_ext is not None, "grid_ext not set"
    assert config.grid_flavor is not None, "grid_flavor not set"


def test_scoreattributesconfig_nonexistent_grid():
    """
    Test that ScoreAttributesConfig raises a ValidationError when passed a non-existent
    grid.
    """

    attributes = {
        "generator_mwh_score": {"attribute": "generator_mwh", "score_method": "minmax"},
        "fttp_average_speed_score": {
            "attribute": "fttp_average_speed",
            "score_method": "percentile",
        },
    }
    config = {
        "grid": "not-a-file.gpkg",
        "attributes": attributes,
    }
    with pytest.raises(ValidationError, match="Path does not point to a file"):
        ScoreAttributesConfig(**config)


@pytest.mark.parametrize(
    "attributes,err_msg",
    [
        # use "method" instead of "score_method"
        (
            {"scored": {"attribute": "fttp_average_speed", "method": "minmax"}},
            "Field required",
        ),
        # pass list instead of dictionary
        (
            [{"attribute": "fttp_average_speed", "score_method": "minmax"}],
            "should be a valid dictionary",
        ),
        # pass missing attribute
        (
            {"scored": {"attribute": "not-a-col", "score_method": "minmax"}},
            "Attribute not-a-col not found in",
        ),
    ],
)
def test_scoreattributesconfig_invalid_attributes(data_dir, attributes, err_msg):
    """
    Test that ScoreAttributesConfig raises a ValidationError when passed various
    types of invalid inputs for attributes
    """

    grid = data_dir / "characterize/outputs/grid_char.gpkg"
    config = {
        "grid": grid,
        "attributes": attributes,
    }
    with pytest.raises(ValidationError, match=err_msg):
        ScoreAttributesConfig(**config)


def test_scoreattributesconfig_no_attributes_or_scoremethod(data_dir):
    """
    Check that a ValidationError is raised when ScoreAttributesConfig() is
    initialized without either a score_method or attributes.
    """
    grid = data_dir / "characterize/outputs/grid_char.gpkg"
    config = {
        "grid": grid,
    }
    err_msg = "Either score_method or attributes must be specified"
    with pytest.raises(ValidationError, match=err_msg):
        ScoreAttributesConfig(**config)


@pytest.mark.parametrize("invert", [None, False, True])
def test_scoreattributesconfig_scoremethod_only(data_dir, invert):
    """
    Test that ScoreAttributesConfig correctly propagates attributes when top-level
    score_method is specified but attributes are not.
    """

    grid = data_dir / "characterize/outputs/grid_char.gpkg"
    score_method = "minmax"
    config_data = {"grid": grid, "score_method": score_method}
    if invert is None:
        expected_invert = False
    else:
        config_data["invert"] = invert
        expected_invert = invert

    config = ScoreAttributesConfig(**config_data)

    grid_df = gpd.read_file(grid)
    expected_attributes = {
        f"{c}_score" for c in grid_df.columns if c not in ("geometry", GRID_IDX)
    }
    actual_attributes = set(config.attributes.keys())
    attribute_diffs = actual_attributes.symmetric_difference(expected_attributes)
    assert len(attribute_diffs) == 0, "Propagated attributes do not match expected set"

    for attribute in config.attributes.values():
        assert (
            attribute.score_method == score_method
        ), "Unexpected score_method in propagated attribute"
        assert (
            attribute.invert == expected_invert
        ), "Unexpected invert in propagated attribute"


@pytest.mark.parametrize("invert", [None, False, True])
def test_scoreattributesconfig_scoremethod_backfill(data_dir, invert):
    """
    Test that ScoreAttributesConfig correctly backfills missing attributes when
    top-level score_method is specified but leaves specified attributes unchanged.
    """
    grid = data_dir / "characterize/outputs/grid_char.gpkg"
    score_method = "minmax"
    config_data = {
        "grid": grid,
        "score_method": score_method,
        "attributes": {
            "generator_mwh_score": {
                "attribute": "generator_mwh",
                "score_method": "percentile",
                "invert": True,
            }
        },
    }
    if invert is None:
        expected_invert = False
    else:
        config_data["invert"] = invert
        expected_invert = invert

    config = ScoreAttributesConfig(**config_data)

    grid_df = gpd.read_file(grid)
    expected_attributes = {
        f"{c}_score" for c in grid_df.columns if c not in ("geometry", GRID_IDX)
    }
    actual_attributes = set(config.attributes.keys())
    attribute_diffs = actual_attributes.symmetric_difference(expected_attributes)
    assert len(attribute_diffs) == 0, "Propagated attributes do not match expected set"

    for out_col, attribute in config.attributes.items():
        if out_col == "generator_mwh_score":
            expected_score_method = "percentile"
            expected_attr_invert = True
        else:
            expected_score_method = score_method
            expected_attr_invert = expected_invert
        assert (
            attribute.score_method == expected_score_method
        ), "Unexpected score_method in propagated attribute"
        assert (
            attribute.invert == expected_attr_invert
        ), "Unexpected invert in propgated attribute"


def test_scoreattributesconfig_scoremethod_propagate_warning(data_dir, tmp_path):
    """
    Test that ScoreAttributesConfig raises a warning when one of the propagated output
    columns already exists in the input dataset.
    """
    src_grid = data_dir / "characterize/outputs/grid_char.gpkg"
    grid_df = gpd.read_file(src_grid)
    grid_df["generator_mwh_score"] = 100.0
    grid = tmp_path / "grid_char_mod.gpkg"
    grid_df.to_file(grid)

    config_data = {"grid": grid, "score_method": "minmax"}

    warn_msg = "Output column generator_mwh_score exists in input grid"
    with pytest.warns(UserWarning, match=warn_msg):
        ScoreAttributesConfig(**config_data)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
