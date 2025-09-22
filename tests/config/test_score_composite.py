# -*- coding: utf-8 -*-
"""
config.score_composite module tests
"""
import pytest

import geopandas as gpd
from pydantic import ValidationError

from reVeal.config.score_composite import Attribute


@pytest.mark.parametrize("attribute", ["tline_length_score", "generator_mwh_score"])
@pytest.mark.parametrize("weight", [0.01, 0.5, 0.99, 1.0])
def test_attribute_valid_inputs(data_dir, attribute, weight):
    """
    Test Attribute class with valid inputs. Make sure model is created and properties
    are correctly set.
    """

    dset_src = data_dir / "score_attributes" / "outputs" / "grid_char_attr_scores.gpkg"

    data = {"attribute": attribute, "weight": weight, "dset_src": dset_src}
    attribute_model = Attribute(**data)
    assert (
        attribute_model.attribute == attribute
    ), "Unexpected value for attribute property"
    assert attribute_model.weight == weight, "Unexpected value for weight property"
    assert (
        attribute_model.dset_src == dset_src
    ), "Unexpected value for dset_src property"


@pytest.mark.parametrize(
    "weight,err",
    [
        (2, "Input should be less than or equal to 1"),
        (0, "Input should be greater than 0"),
    ],
)
def test_attributes_invalid_weight(data_dir, weight, err):
    """
    Test that Attributes class raises ValidationError for invalid weights.
    """
    dset_src = data_dir / "score_attributes" / "outputs" / "grid_char_attr_scores.gpkg"

    data = {"attribute": "tline_length_score", "weight": weight, "dset_src": dset_src}
    with pytest.raises(ValidationError, match=err):
        Attribute(**data)


def test_attributes_invalid_attribute_missing(data_dir):
    """
    Test that Attribute raises a validation error when the specified attribute does
    not exist in the dataset.
    """

    dset_src = data_dir / "score_attributes" / "outputs" / "grid_char_attr_scores.gpkg"
    data = {"attribute": "not-a-col", "weight": 0.5, "dset_src": dset_src}
    with pytest.raises(ValidationError, match="Attribute not-a-col not found in"):
        Attribute(**data)


def test_attributes_invalid_attribute_nonnumeric(data_dir, tmp_path):
    """
    Test that Attribute raises a TypeError when passed a non-numeric attribute.
    """
    raw_dset_src = (
        data_dir / "score_attributes" / "outputs" / "grid_char_attr_scores.gpkg"
    )
    df = gpd.read_file(raw_dset_src)
    df["new-col"] = "foo"
    dset_src = tmp_path / "grid_char_attr_scores.gpkg"
    df.to_file(dset_src)

    data = {"attribute": "new-col", "weight": 0.5, "dset_src": dset_src}
    with pytest.raises(TypeError, match="Must be a numeric dtype."):
        Attribute(**data)


def test_attribute_invalid_dset(tmp_path):
    """
    Test that Attribute raises an OSError when passed a dataset that exist but is
    not a compatible vector dataset format.
    """

    dset_src = tmp_path / "mock.tif"
    dset_src.touch()

    data = {"attribute": "some-col", "weight": 0.5, "dset_src": dset_src}
    with pytest.raises(OSError, match="Unable to read input vector file"):
        Attribute(**data)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
