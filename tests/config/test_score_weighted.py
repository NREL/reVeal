# -*- coding: utf-8 -*-
"""
config.score_weighted module tests
"""
import pytest

import geopandas as gpd
from pydantic import ValidationError

from reVeal.config.score_weighted import Attribute, ScoreWeightedConfig


@pytest.mark.parametrize("attribute", ["tline_length_score", "generator_mwh_score"])
@pytest.mark.parametrize("weight", [0.01, 0.5, 0.99, 1.0])
def test_attribute_valid_inputs(data_dir, attribute, weight):
    """
    Test Attribute class with valid inputs. Make sure model is created and properties
    are correctly set.
    """

    dset_src = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"

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
    dset_src = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"

    data = {"attribute": "tline_length_score", "weight": weight, "dset_src": dset_src}
    with pytest.raises(ValidationError, match=err):
        Attribute(**data)


def test_attributes_invalid_attribute_missing(data_dir):
    """
    Test that Attribute raises a validation error when the specified attribute does
    not exist in the dataset.
    """

    dset_src = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    data = {"attribute": "not-a-col", "weight": 0.5, "dset_src": dset_src}
    with pytest.raises(ValidationError, match="Attribute not-a-col not found in"):
        Attribute(**data)


def test_attributes_invalid_attribute_nonnumeric(data_dir, tmp_path):
    """
    Test that Attribute raises a TypeError when passed a non-numeric attribute.
    """
    raw_dset_src = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
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


def test_scoreattributesconfig_valid_inputs(data_dir):
    """
    Test that ScoreWeightedConfig builds successfully with valid inputs.
    """

    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    attributes = [
        {"attribute": "generator_mwh_score", "weight": 0.25},
        {"attribute": "tline_length_score", "weight": 0.25},
        {"attribute": "fttp_average_speed_score", "weight": 0.25},
        {"attribute": "developable_area_score", "weight": 0.25},
    ]
    config_data = {
        "grid": grid,
        "attributes": attributes,
        "score_name": "composite_score",
    }
    config = ScoreWeightedConfig(**config_data)

    # check dynamic attributes are set
    assert config.grid_ext is not None, "grid_ext not set"
    assert config.grid_flavor is not None, "grid_flavor not set"


def test_scoreweightedconfig_nonexistent_grid():
    """
    Test that ScoreWeightedConfig raises a ValidationError when passed a non-existent
    grid.
    """

    attributes = [
        {"attribute": "generator_mwh_score", "weight": 0.25},
        {"attribute": "tline_length_score", "weight": 0.25},
        {"attribute": "fttp_average_speed_score", "weight": 0.25},
        {"attribute": "developable_area_score", "weight": 0.25},
    ]
    config = {
        "grid": "not-a-file.gpkg",
        "attributes": attributes,
        "score_name": "composite_score",
    }
    with pytest.raises(ValidationError, match="Path does not point to a file"):
        ScoreWeightedConfig(**config)


def test_scoreweightedconfig_bad_weight_sum(data_dir):
    """
    Test that ScoreWeightedConfig raises a ValidationError when the weights don't
    sum to 1.
    """
    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    attributes = [
        {"attribute": "generator_mwh_score", "weight": 0.20},
        {"attribute": "tline_length_score", "weight": 0.25},
        {"attribute": "fttp_average_speed_score", "weight": 0.25},
        {"attribute": "developable_area_score", "weight": 0.25},
    ]
    config_data = {
        "grid": grid,
        "attributes": attributes,
        "score_name": "composite_score",
    }
    with pytest.raises(
        ValidationError, match="Weights of input attributes must sum to 1"
    ):
        ScoreWeightedConfig(**config_data)


def test_scoreweightedconfig_missing_attribute(data_dir):
    """
    Test that ScoreWeightedConfig raises a ValidationError when passed an attribute
    that doesn't exist in the grid dataset.
    """

    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    attributes = [
        {"attribute": "generator_mwh_score", "weight": 0.25},
        {"attribute": "tline_length_score", "weight": 0.25},
        {"attribute": "fttp_average_speed_score", "weight": 0.25},
        {"attribute": "not-a-col", "weight": 0.25},
    ]
    config_data = {
        "grid": grid,
        "attributes": attributes,
        "score_name": "composite_score",
    }
    with pytest.raises(ValidationError, match="Attribute not-a-col not found"):
        ScoreWeightedConfig(**config_data)


@pytest.mark.parametrize(
    "weight,err",
    [
        (-1, "Input should be greater than 0"),
        (1.1, "Input should be less than or equal to 1"),
        ("ten", "Input should be a valid number"),
    ],
)
def test_scoreweightedconfig_invalid_weight(data_dir, weight, err):
    """
    Test that ScoreWeightedConfig raises the correct ValidationErrors when passed
    various invalid weights.
    """

    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    attributes = [
        {"attribute": "generator_mwh_score", "weight": weight},
        {"attribute": "tline_length_score", "weight": 0.25},
        {"attribute": "fttp_average_speed_score", "weight": 0.25},
        {"attribute": "developable_area_score", "weight": 0.25},
    ]
    config_data = {
        "grid": grid,
        "attributes": attributes,
        "score_name": "composite_score",
    }

    with pytest.raises(ValidationError, match=err):
        ScoreWeightedConfig(**config_data)


@pytest.mark.parametrize(
    "attributes",
    [
        [{"attribute": "generator_mwh_score"}],
        [{"weight": 0.5}],
    ],
)
def test_scoreweightedconfig_missing_attribute_fields(data_dir, attributes):
    """
    Test that ScoreWeightedConfig raises a ValidationError when passed attributes
     that are missing required fields.
    """
    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    config_data = {
        "grid": grid,
        "attributes": attributes,
        "score_name": "composite_score",
    }
    with pytest.raises(ValidationError, match="Field required"):
        ScoreWeightedConfig(**config_data)


def test_scoreweightedconfig_attribute_dict(data_dir):
    """
    Test that ScoreWeightedConfig raises a ValidationError when passed
    a dictionary for attributes instead of a list.
    """
    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    attributes = {
        "generator_mwh_score": 0.25,
        "tline_length_score": 0.25,
        "fttp_average_speed_score": 0.25,
        "developable_area_score": 0.25,
    }

    config_data = {
        "grid": grid,
        "attributes": attributes,
        "score_name": "composite_score",
    }
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        ScoreWeightedConfig(**config_data)


def test_scoreweightedconfig_no_score_name(data_dir):
    """
    Test that ScoreWeightedConfig raises a ValidationError when score_method
    is not provided.
    """
    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    attributes = [
        {"attribute": "generator_mwh_score", "weight": 0.25},
        {"attribute": "tline_length_score", "weight": 0.25},
        {"attribute": "fttp_average_speed_score", "weight": 0.25},
        {"attribute": "developable_area_score", "weight": 0.25},
    ]
    config_data = {
        "grid": grid,
        "attributes": attributes,
    }
    with pytest.raises(ValidationError, match="Field required"):
        ScoreWeightedConfig(**config_data)


def test_scoreweightedconfig_score_name_warning(data_dir):
    """
    Test that ScoreWeightedConfig raises a warning when the attribute specified by
    score_name is a column that already exists in the input dataset.
    """
    grid = data_dir / "normalize" / "outputs" / "grid_char_norm.gpkg"
    attributes = [
        {"attribute": "generator_mwh_score", "weight": 0.25},
        {"attribute": "tline_length_score", "weight": 0.25},
        {"attribute": "fttp_average_speed_score", "weight": 0.25},
        {"attribute": "developable_area_score", "weight": 0.25},
    ]
    config_data = {
        "grid": grid,
        "attributes": attributes,
        "score_name": "grid_id_score",
    }
    warn_msg = "Output column grid_id_score exists in input grid"
    with pytest.warns(UserWarning, match=warn_msg):
        ScoreWeightedConfig(**config_data)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
