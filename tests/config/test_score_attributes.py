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
    # ScoreAttributesConfig,
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
    "dset,attribute,score_method",
    [
        ("characterize/outputs/grid_char.gpkg", "tline_length", "minmax"),
        ("characterize/outputs/grid_char.gpkg", "fttp_average_speed", "percentile"),
        ("characterize/outputs/grid_char.parquet", "generator_mwh", "minmax"),
        ("characterize/outputs/grid_char.parquet", "developable_area", "percentile"),
    ],
)
def test_attribute_valid_inputs(data_dir, attribute, score_method, dset):
    """
    Test Attribute class with valid inputs and make sure dynamic attributes are set.
    """
    dset_src = data_dir / dset
    value = {
        "attribute": attribute,
        "score_method": score_method,
        "dset_src": dset_src,
    }
    attribute = Attribute(**value)

    # check dynamic attributes are set
    assert attribute.dset_ext is not None, "dset_ext not set"
    assert attribute.dset_flavor is not None, "dset_flavor not set"


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


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
