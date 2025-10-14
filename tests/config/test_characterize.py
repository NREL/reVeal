# -*- coding: utf-8 -*-
"""
config.characterize module tests
"""
import json
from pathlib import Path

import pytest

from pydantic import ValidationError

from reVeal.config.config import load_config
from reVeal.config.characterize import (
    Characterization,
    VALID_CHARACTERIZATION_METHODS,
    CharacterizeConfig,
    DatasetFormatEnum,
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
NONPARALLEL_METHODS = [
    k
    for k, v in VALID_CHARACTERIZATION_METHODS.items()
    if not v.get("supports_parallel")
]
NONWHERE_METHODS = [
    k for k, v in VALID_CHARACTERIZATION_METHODS.items() if not v.get("supports_where")
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


@pytest.mark.parametrize(
    "dset,method,weights_dset,where,parallel",
    [
        (
            "rasters/fiber_lines_onshore_proximity.tif",
            "mean",
            "characterize/rasters/developable.tif",
            None,
            True,
        ),
        ("rasters/fiber_lines_onshore_proximity.tif", "mean", None, None, False),
        ("vectors/generators.gpkg", "feature count", None, "value > 1", False),
        ("vectors/generators.gpkg", "feature count", None, None, False),
    ],
)
@pytest.mark.parametrize("neighbor_order", [0, 1, 50.0])
@pytest.mark.parametrize("buffer_distance", [0, -100, 100])
def test_characterization_valid_optional_params(
    data_dir,
    dset,
    method,
    weights_dset,
    neighbor_order,
    buffer_distance,
    where,
    parallel,
):
    """
    Test Characterization class with valid inputs for optional parameters.
    """

    value = {
        "dset": f"characterize/{dset}",
        "data_dir": data_dir,
        "method": method,
        "attribute": None,
        "weights_dset": weights_dset,
        "neighbor_order": neighbor_order,
        "buffer_distance": buffer_distance,
        "where": where,
        "parallel": parallel,
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
        "parallel": False,
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
        if attribute:
            attribute = "net_generation_megawatthours"
    elif geom_type == "line":
        dset = "characterize/vectors/tlines.gpkg"
        if attribute:
            attribute = "VOLTAGE"
    elif geom_type == "polygon":
        dset = "characterize/vectors/fiber_to_the_premises.gpkg"
        if attribute:
            attribute = "max_advertised_upload_speed"
    elif geom_type == "raster":
        dset = "characterize/rasters/fiber_lines_onshore_proximity.tif"
    else:
        raise ValueError("Unrecognized geom_type")

    supports_parallel = VALID_CHARACTERIZATION_METHODS.get(method).get(
        "supports_parallel"
    )

    value = {
        "dset": dset,
        "data_dir": data_dir,
        "method": method,
        "attribute": attribute,
        "parallel": supports_parallel,
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
        "parallel": False,
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
    attribute = None

    dset = None
    if geom_type == "point":
        dset = "characterize/vectors/generators.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "net_generation_megawatthours"
    elif geom_type == "line":
        dset = "characterize/vectors/tlines.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "VOLTAGE"
    elif geom_type == "polygon":
        dset = "characterize/vectors/fiber_to_the_premises.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "max_advertised_upload_speed"
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
        "parallel": False,
    }
    with pytest.warns(
        UserWarning, match="weights_dset specified but will not be applied.*"
    ):
        Characterization(**value)


@pytest.mark.parametrize("method", NONPARALLEL_METHODS)
@pytest.mark.parametrize(
    "parallel,max_workers",
    [
        (True, None),
        (False, 2),
        (True, 2),
    ],
)
def test_characterization_superfluous_parallel(data_dir, method, parallel, max_workers):
    """
    Test Characterization class raises warning when parallel=True is specified but
    not applicable to the method.
    """
    geom_type = VALID_CHARACTERIZATION_METHODS.get(method).get("valid_inputs")[0]
    attribute = None

    dset = None
    if geom_type == "point":
        dset = "characterize/vectors/generators.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "net_generation_megawatthours"
    elif geom_type == "line":
        dset = "characterize/vectors/tlines.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "VOLTAGE"
    elif geom_type == "polygon":
        dset = "characterize/vectors/fiber_to_the_premises.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "max_advertised_upload_speed"
    elif geom_type == "raster":
        dset = "characterize/rasters/fiber_lines_onshore_proximity.tif"
    else:
        raise ValueError("Unrecognized geom_type")

    value = {
        "dset": dset,
        "data_dir": data_dir,
        "method": method,
        "attribute": attribute,
        "parallel": parallel,
        "max_workers": max_workers,
    }
    with pytest.warns(
        UserWarning, match="parallel specified as True and/or max_workers provided"
    ):
        Characterization(**value)


@pytest.mark.parametrize("method", NONWHERE_METHODS)
def test_characterization_superfluous_where(data_dir, method):
    """
    Test Characterization class raises warning when where is specified but
    not applicable to the method.
    """
    geom_type = VALID_CHARACTERIZATION_METHODS.get(method).get("valid_inputs")[0]
    attribute = None

    dset = None
    if geom_type == "point":
        dset = "characterize/vectors/generators.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "net_generation_megawatthours"
    elif geom_type == "line":
        dset = "characterize/vectors/tlines.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "VOLTAGE"
    elif geom_type == "polygon":
        dset = "characterize/vectors/fiber_to_the_premises.gpkg"
        if VALID_CHARACTERIZATION_METHODS.get(method).get("attribute_required"):
            attribute = "max_advertised_upload_speed"
    elif geom_type == "raster":
        dset = "characterize/rasters/fiber_lines_onshore_proximity.tif"
    else:
        raise ValueError("Unrecognized geom_type")

    value = {
        "dset": dset,
        "data_dir": data_dir,
        "method": method,
        "attribute": attribute,
        "where": "value > 1",
    }
    with pytest.warns(UserWarning, match="where specified but will not be applied.*"):
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


@pytest.mark.parametrize(
    "dset,attribute,err",
    [
        ("characterize/vectors/generators", "volts", ValueError),
        ("characterize/vectors/generators", "primary_fuel_type", TypeError),
    ],
)
@pytest.mark.parametrize("dset_ext", ["gpkg", "parquet"])
def test_characterization_invalid_attributes(data_dir, dset, dset_ext, attribute, err):
    """
    Check that invalid attributes -- either missing from the input dataset or not
    a numeric data type -- are caught.
    """

    value = {
        "dset": f"{dset}.{dset_ext}",
        "data_dir": data_dir,
        "method": "sum attribute",
        "attribute": attribute,
    }
    if err is ValueError:
        err_msg = f"Attribute {attribute} not found in"
    else:
        err_msg = "Must be a numeric dtype"
    with pytest.raises(err, match=err_msg):
        Characterization(**value)


@pytest.mark.parametrize(
    "bad_where",
    [
        "@pd.compat.os.system('echo foo)",
        "os.system('echo foo')",
        "print(sys.executable)",
        "import time",
    ],
)
def test_characterization_where_injection(data_dir, bad_where):
    """
    Test that validation catches questionable inputs for where.
    """

    value = {
        "dset": "characterize/vectors/generators.gpkg",
        "data_dir": data_dir,
        "method": "feature count",
        "where": bad_where,
    }
    with pytest.raises(ValueError, match="Will not eval().*"):
        Characterization(**value)


@pytest.mark.parametrize("max_workers", [2, None])
def test_characterization_valid_max_workers(data_dir, max_workers):
    """
    Test Characterization class with valid inputs for max_workers.
    """
    value = {
        "dset": "characterize/rasters/fiber_lines_onshore_proximity.tif",
        "data_dir": data_dir,
        "method": "mean",
        "parallel": True,
        "max_workers": max_workers,
    }
    Characterization(**value)


@pytest.mark.parametrize("max_workers", [0, -1])
def test_characterization_invalid_max_workers(data_dir, max_workers):
    """
    Test Characterization class with valid inputs for max_workers.
    """
    value = {
        "dset": "characterize/rasters/fiber_lines_onshore_proximity.tif",
        "data_dir": data_dir,
        "method": "mean",
        "parallel": True,
        "max_workers": max_workers,
    }
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        Characterization(**value)


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


@pytest.mark.parametrize(
    "characterizations,err_msg",
    [
        (
            {"dev_area": {"dset": "rasters/developable.tif", "method": "not-a-method"}},
            "Invalid method specified",
        ),
        (
            [{"dset": "rasters/developable.tif", "method": "not-a-method"}],
            "Input should be a valid dictionary",
        ),
    ],
)
def test_characterizationconfig_invalid_characterizations(
    data_dir, characterizations, err_msg
):
    """
    Test CharacterizationConfig with invalid characterizations.
    """

    grid_path = data_dir / "characterize" / "grids" / "grid_2.gpkg"
    config = {
        "data_dir": data_dir / "characterize",
        "grid": grid_path.as_posix(),
        "characterizations": characterizations,
        "expressions": {"developable_sqkm": "developable_area / 1e6"},
    }
    with pytest.raises(ValidationError, match=err_msg):
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
        config = load_config(config_data, CharacterizeConfig)
    else:
        config = load_config(CharacterizeConfig(**config_data), CharacterizeConfig)

    assert isinstance(config, CharacterizeConfig)


def test_load_characterize_config_typerror():
    """
    Test that laod_characterize_config() raises a TypeError when passed an unsupported
    input.
    """

    with pytest.raises(TypeError, match="Invalid input for config.*"):
        load_config("string input", CharacterizeConfig)


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


@pytest.mark.parametrize(
    "bad_expression",
    [
        "@pd.compat.os.system('echo foo)",
        "os.system('echo foo')",
        "print(sys.executable)",
        "import time",
    ],
)
def test_characterizationconfig_expression_injection(data_dir, bad_expression):
    """
    Test that validation catches questionable inputs for expressions.
    """

    grid_path = data_dir / "characterize" / "grids" / "grid_1.gpkg"
    grid_path.touch()
    config = {
        "data_dir": data_dir.as_posix(),
        "grid": grid_path.as_posix(),
        "characterizations": {},
        "expressions": {"bad_actor": bad_expression},
    }
    with pytest.raises(ValueError, match="Will not eval().*"):
        CharacterizeConfig(**config)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
