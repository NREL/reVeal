# -*- coding: utf-8 -*-
"""
config module tests
"""
import json
from pathlib import Path

import pytest

from pydantic import ValidationError

from reVeal.config import (
    Characterization,
    VALID_CHARACTERIZATION_METHODS,
    CharacterizeConfig,
    DatasetFormatEnum,
    load_characterize_config,
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
NONWEIGHTS_METHODS = [
    k
    for k, v in VALID_CHARACTERIZATION_METHODS.items()
    if not v.get("supports_weights")
]


@pytest.mark.parametrize(
    "value,error_expected",
    [
        ("raster", False),
        ("point", False),
        ("line", False),
        ("polygon", False),
        ("RASTER", False),
        ("POINT", False),
        ("LINE", False),
        ("POLYGON", False),
        ("polygons", True),
        ("geometry", True),
        ("vector", True),
    ],
)
def test_datasetformatenum(value, error_expected):
    """
    Test for DatasetFormatEnum.
    """
    if error_expected:
        with pytest.raises(ValueError):
            DatasetFormatEnum(value)
    else:
        DatasetFormatEnum(value)


@pytest.mark.parametrize("weights_dset", [None, "characterize/rasters/developable.tif"])
@pytest.mark.parametrize("neighbor_order", [None, 0, 1, 50.0])
@pytest.mark.parametrize("buffer_distance", [None, -100, 100])
def test_characterization_valid_optional_params(
    data_dir,
    weights_dset,
    neighbor_order,
    buffer_distance,
):
    """
    Test Characterization class with valid inputs for optional parameters.
    """

    value = {
        "dset": "characterize/rasters/fiber_lines_onshore_proximity.tif",
        "data_dir": data_dir,
        "method": "mean",
        "attribute": None,
        "weights_dset": weights_dset,
        "neighbor_order": neighbor_order,
        "buffer_distance": buffer_distance,
    }

    Characterization(**value)


@pytest.mark.parametrize(
    "dset_name,geom_type,method,weights_dset",
    [
        ("characterize/vectors/generators.gpkg", "point", "feature count", None),
        ("characterize/vectors/tlines.gpkg", "line", "sum length", None),
        (
            "characterize/vectors/fiber_to_the_premises.parquet",
            "polygon",
            "sum area",
            None,
        ),
        (
            "characterize/rasters/fiber_lines_onshore_proximity.tif",
            "raster",
            "mean",
            "characterize/rasters/developable.tif",
        ),
    ],
)
def test_characterization_dynamic_attributes(
    data_dir, dset_name, geom_type, method, weights_dset
):
    """
    Test Characterization() class correctly populates dynamic properties.
    """
    value = {
        "dset": dset_name,
        "data_dir": data_dir,
        "method": method,
        "attribute": None,
        "weights_dset": weights_dset,
        "neighbor_order": 0,
        "buffer_distance": 0,
    }
    characterization = Characterization(**value)

    assert characterization.dset_src is not None, "dset_src property not set"
    assert characterization.dset_format is not None, "dset_format property not set"
    assert characterization.dset_ext is not None, "dset_ext property not set"
    assert characterization.crs is not None, "crs property not set"
    if weights_dset is not None:
        assert (
            characterization.weights_dset_src is not None
        ), "weights_dset_src property not set"

    assert (
        characterization.dset_src == data_dir / dset_name
    ), "Unexpected value for dset_src"
    assert characterization.dset_format == geom_type, "Unexpected value for dset_format"
    assert (
        characterization.dset_ext == Path(dset_name).suffix
    ), "Unexpected value for dset_suffix"
    assert characterization.crs == "EPSG:5070", "Unexpected value for CRS"
    if weights_dset is not None:
        assert (
            characterization.weights_dset_src == data_dir / weights_dset
        ), "Unexpected value for weights_dset_src"


@pytest.mark.parametrize("method,attribute", VALID_METHODS_AND_ATTRIBUTES)
def test_characterization_valid_methods_and_attributes(data_dir, method, attribute):
    """
    Test Characterization class with valid combos of methods and attributes.
    """

    geom_type = VALID_CHARACTERIZATION_METHODS.get(method).get("valid_inputs")[0]
    dset = None
    if geom_type == "point":
        dset = "characterize/vectors/generators.gpkg"
    elif geom_type == "line":
        dset = "characterize/vectors/tlines.gpkg"
    elif geom_type == "polygon":
        dset = "characterize/vectors/fiber_to_the_premises.gpkg"
    elif geom_type == "raster":
        dset = "characterize/rasters/fiber_lines_onshore_proximity.tif"
    else:
        raise ValueError("Unrecognized geom_type")

    value = {
        "dset": dset,
        "data_dir": data_dir,
        "method": method,
        "attribute": attribute,
    }

    Characterization(**value)


@pytest.mark.parametrize("method,attribute", METHODS_MISSING_ATTRIBUTES)
def test_characterization_invalid_methods_and_attributes(data_dir, method, attribute):
    """
    Test Characterization class with invalid combos of methods and attributes.
    """
    value = {
        "dset": "characterize/vectors/generators.gpkg",
        "data_dir": data_dir,
        "method": method,
        "attribute": attribute,
    }
    with pytest.raises(ValidationError, match="attribute was not provided*."):
        Characterization(**value)


@pytest.mark.parametrize("method,attribute", METHODS_SUPERFLUOUS_ATTRIBUTES)
def test_characterization_superfluous_methods_and_attributes(
    data_dir, method, attribute
):
    """
    Test Characterization class with invalid combos of methods and attributes.
    """
    geom_type = VALID_CHARACTERIZATION_METHODS.get(method).get("valid_inputs")[0]

    dset = None
    if geom_type == "point":
        dset = "characterize/vectors/generators.gpkg"
    elif geom_type == "line":
        dset = "characterize/vectors/tlines.gpkg"
    elif geom_type == "polygon":
        dset = "characterize/vectors/fiber_to_the_premises.gpkg"
    elif geom_type == "raster":
        dset = "characterize/rasters/fiber_lines_onshore_proximity.tif"
    else:
        raise ValueError("Unrecognized geom_type")

    value = {
        "dset": dset,
        "data_dir": data_dir,
        "method": method,
        "attribute": attribute,
    }
    with pytest.warns(
        UserWarning, match="attribute specified but will not be applied.*"
    ):
        Characterization(**value)


@pytest.mark.parametrize("method", NONWEIGHTS_METHODS)
def test_characterization_superfluous_weights_dset(data_dir, method):
    """
    Test Characterization class raises warning when weights_dset is specified but
    not applicable to the method.
    """
    geom_type = VALID_CHARACTERIZATION_METHODS.get(method).get("valid_inputs")[0]
    if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
        attribute = "a_field"
    else:
        attribute = None

    dset = None
    if geom_type == "point":
        dset = "characterize/vectors/generators.gpkg"
    elif geom_type == "line":
        dset = "characterize/vectors/tlines.gpkg"
    elif geom_type == "polygon":
        dset = "characterize/vectors/fiber_to_the_premises.gpkg"
    elif geom_type == "raster":
        dset = "characterize/rasters/fiber_lines_onshore_proximity.tif"
    else:
        raise ValueError("Unrecognized geom_type")

    value = {
        "dset": dset,
        "data_dir": data_dir,
        "method": method,
        "attribute": attribute,
        "weights_dset": "characterize/rasters/developable.tif",
    }
    with pytest.warns(
        UserWarning, match="weights_dset specified but will not be applied.*"
    ):
        Characterization(**value)


@pytest.mark.parametrize(
    "field,value,err",
    [
        ("method", "not a valid method", "Invalid method specified*."),
        ("weights_dset", "weights.tif", "Path does not point to a file*."),
        ("neighbor_order", -1, "Input should be greater than or equal to 0*."),
        ("buffer_distance", "thirty", "Input should be a valid number*."),
        ("method", None, "Input should be a valid string*."),
        ("dset", None, "Field required.*"),
    ],
)
def test_characterization_invalid(data_dir, field, value, err):
    """
    Test Characterization class with invalid inputs.
    """

    inputs = {
        "dset": "characterize/vectors/generators.gpkg",
        "data_dir": data_dir,
        "method": "feature count",
    }
    inputs[field] = value
    with pytest.raises(ValidationError, match=err):
        Characterization(**inputs)


def test_characterization_extra():
    """
    Test Characterization class with extra fields.
    """

    inputs = {"dset": "test/dset.gpkg", "method": "feature count", "extra_field": 1}
    with pytest.raises(ValidationError, match="Extra inputs.*"):
        Characterization(**inputs)


@pytest.mark.parametrize("drop_expressions", [True, False])
def test_characterizationconfig_valid_inputs(data_dir, drop_expressions):
    """
    Test CharacterizationConfig with valid inputs.
    """

    grid_path = data_dir / "characterize" / "grids" / "grid_1.gpkg"
    grid_path.touch()
    config = {
        "data_dir": data_dir.as_posix(),
        "grid": grid_path.as_posix(),
        "characterizations": {
            "developable_area": {
                "dset": "characterize/rasters/developable.tif",
                "method": "area",
            }
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


@pytest.mark.parametrize("from_dict", [True, False])
def test_load_characterize_config(data_dir, from_dict):
    """
    test that load_charactrize_config() works when passed either a dict or
    CharacterizeConfig input.
    """

    in_config_path = data_dir / "characterize" / "config.json"
    with open(in_config_path, "r") as f:
        config_data = json.load(f)

    config_data["data_dir"] = (data_dir / "characterize").as_posix()
    config_data["grid"] = (
        data_dir / "characterize" / "grids" / "grid_1.gpkg"
    ).as_posix()

    if from_dict:
        config = load_characterize_config(config_data)
    else:
        config = load_characterize_config(CharacterizeConfig(**config_data))

    assert isinstance(config, CharacterizeConfig)


def test_load_characterize_config_typerror():
    """
    Test that laod_characterize_config() raises a TypeError when passed an unsupported
    input.
    """

    with pytest.raises(TypeError, match="Invalid input for characterize config.*"):
        load_characterize_config("string input")


def test_characterizationconfig_crs_mismatch(
    data_dir,
):
    """
    Test that CharacterizationConfig raises an error when passed a grid and
    characterizations with mismatched CRSs.
    """

    grid_path = data_dir / "characterize" / "grids" / "grid_3.gpkg"
    grid_path.touch()
    config = {
        "data_dir": data_dir.as_posix(),
        "grid": grid_path.as_posix(),
        "characterizations": {
            "developable_area": {
                "dset": "characterize/rasters/developable.tif",
                "method": "area",
            }
        },
    }
    with pytest.raises(ValidationError, match="CRS of input dataset*."):
        CharacterizeConfig(**config)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
