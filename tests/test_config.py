# -*- coding: utf-8 -*-
"""
config module tests
"""
import pytest

from pydantic import ValidationError

from loci.config import (
    Characterization,
    VALID_CHARACTERIZATION_METHODS,
    CharacterizeConfig,
)

VALID_METHODS_AND_ATTRIBUTES = [
    (k, "a_field") if v["attribute_required"] else (k, None)
    for k, v in VALID_CHARACTERIZATION_METHODS.items()
]
METHODS_MISSING_ATTRIBUTES = [
    (k, None)
    for k, v in VALID_CHARACTERIZATION_METHODS.items()
    if v["attribute_required"]
]
METHODS_SUPERFLUOUS_ATTRIBUTES = [
    (k, "a_field")
    for k, v in VALID_CHARACTERIZATION_METHODS.items()
    if not v["attribute_required"]
]


@pytest.mark.parametrize("apply_exclusions", [None, True, False])
@pytest.mark.parametrize("neighbor_order", [None, 0, 1, 50.0])
@pytest.mark.parametrize("buffer_distance", [None, -100, 100])
def test_characterization_valid_optional_params(
    apply_exclusions, neighbor_order, buffer_distance
):
    """
    Test Characterization class with valid inputs for optional parameters.
    """

    value = {
        "dset": "test/dset.gpkg",
        "method": "sum area",
        "attribute": None,
        "apply_exclusions": apply_exclusions,
        "neighbor_order": neighbor_order,
        "buffer_distance": buffer_distance,
    }

    Characterization(**value)


@pytest.mark.parametrize("method,attribute", VALID_METHODS_AND_ATTRIBUTES)
def test_characterization_valid_methods_and_attributes(method, attribute):
    """
    Test Characterization class with valid combos of methods and attributes.
    """

    value = {
        "dset": "test/dset.gpkg",
        "method": method,
        "attribute": attribute,
    }

    Characterization(**value)


@pytest.mark.parametrize("method,attribute", METHODS_MISSING_ATTRIBUTES)
def test_characterization_invalid_methods_and_attributes(method, attribute):
    """
    Test Characterization class with invalid combos of methods and attributes.
    """

    value = {
        "dset": "test/dset.gpkg",
        "method": method,
        "attribute": attribute,
    }
    with pytest.raises(ValidationError):
        Characterization(**value)


@pytest.mark.parametrize("method,attribute", METHODS_SUPERFLUOUS_ATTRIBUTES)
def test_characterization_superfluous_methods_and_attributes(method, attribute):
    """
    Test Characterization class with invalid combos of methods and attributes.
    """

    value = {
        "dset": "test/dset.gpkg",
        "method": method,
        "attribute": attribute,
    }
    with pytest.warns(UserWarning):
        Characterization(**value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "not a valid method"),  # invalid entry
        ("apply_exclusions", "yes"),  # invalid entry
        ("neighbor_order", -1),  # invalid entry
        ("buffer_distance", "thirty"),  # invalid entry
        ("method", None),  # required field
        ("dset", None),  # required field
    ],
)
def test_characterization_invalid(field, value):
    """
    Test Characterization class with invalid inputs.
    """

    inputs = {
        "dset": "test/dset.gpkg",
        "method": "feature count",
    }
    inputs[field] = value
    with pytest.raises(ValidationError):
        Characterization(**inputs)


def test_characterization_extra():
    """
    Test Characterization class with extra fields.
    """

    inputs = {"dset": "test/dset.gpkg", "method": "feature count", "extra_field": 1}
    with pytest.raises(ValidationError):
        Characterization(**inputs)


@pytest.mark.parametrize("drop_expressions", [True, False])
def test_characterizationconfig_valid_inputs(tmp_path, drop_expressions):
    """
    Test CharacterizationConfig with valid inputs.
    """

    grid_path = tmp_path / "grid.gpkg"
    grid_path.touch()
    config = {
        "data_dir": tmp_path.as_posix(),
        "grid": grid_path.as_posix(),
        "characterizations": {
            "developable_area": {"dset": "rasters/developable.tif", "method": "area"}
        },
        "expressions": {"developable_sqkm": "developable_area / 1e6"},
    }
    if drop_expressions:
        config.pop("expressions")
    CharacterizeConfig(**config)


def test_characterizationconfig_nonexistent_datadir(tmp_path):
    """
    Test CharacterizationConfig with non-existent data_dir.
    """

    grid_path = tmp_path / "grid.gpkg"
    grid_path.touch()
    config = {
        "data_dir": "/data/directory",
        "grid": grid_path.as_posix(),
        "characterizations": {
            "developable_area": {"dset": "rasters/developable.tif", "method": "area"}
        },
        "expressions": {"developable_sqkm": "developable_area / 1e6"},
    }
    with pytest.raises(ValidationError):
        CharacterizeConfig(**config)


def test_characterizationconfig_nonexistent_grid(tmp_path):
    """
    Test CharacterizationConfig with non-existent grid.
    """

    grid_path = tmp_path / "grid.gpkg"
    config = {
        "data_dir": tmp_path.as_posix(),
        "grid": grid_path.as_posix(),
        "characterizations": {
            "developable_area": {"dset": "rasters/developable.tif", "method": "area"}
        },
        "expressions": {"developable_sqkm": "developable_area / 1e6"},
    }
    with pytest.raises(ValidationError):
        CharacterizeConfig(**config)


def test_characterizationconfig_invalid_characterizations(tmp_path):
    """
    Test CharacterizationConfig with invalid characterizations.
    """

    grid_path = tmp_path / "grid.gpkg"
    config = {
        "data_dir": tmp_path.as_posix(),
        "grid": grid_path.as_posix(),
        "characterizations": {
            "developable_area": {
                "dset": "rasters/developable.tif",
                "method": "not-a-method",
            }
        },
        "expressions": {"developable_sqkm": "developable_area / 1e6"},
    }
    with pytest.raises(ValidationError):
        CharacterizeConfig(**config)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
