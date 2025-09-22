# -*- coding: utf-8 -*-
"""
io module tests
"""
import pytest
import numpy as np
from pandas.errors import UndefinedVariableError

from reVeal.fileio import (
    get_geom_info_parquet,
    get_geom_type_parquet,
    get_geom_type_pyogrio,
    get_crs_raster,
    get_crs_pyogrio,
    get_crs_parquet,
    read_vectors,
    get_attributes_parquet,
    get_attributes_pyogrio,
    attribute_is_numeric,
)


def test_get_geom_info_parquet(data_dir):
    """
    Happy path test for get_geom_info_parquet(). Test that it returns dict with
    expected keys for a valid geoparquet file.
    """
    dset_src = data_dir / "characterize" / "vectors" / "fiber_to_the_premises.parquet"
    geom_info = get_geom_info_parquet(dset_src)

    expected_keys = ["encoding", "crs", "geometry_types", "bbox", "covering"]
    assert (
        list(geom_info.keys()) == expected_keys
    ), "Geometry info dictionary does not have expected keys"


def test_get_geom_info_valueerror(data_dir):
    """
    Test that get_geom_info_valueerror() raises a ValueError for a Parquet file
    without geometry columns.
    """
    dset_src = data_dir / "edge_case_inputs" / "no_geometry.parquet"
    with pytest.raises(ValueError):
        get_geom_info_parquet(dset_src)


@pytest.mark.parametrize(
    "dset_name,expected_geom_type",
    [
        ("characterize/vectors/fiber_to_the_premises.parquet", "polygon"),
        ("characterize/vectors/generators.parquet", "point"),
        ("characterize/vectors/tlines.parquet", "line"),
        ("edge_case_inputs/combo_lines_multilines.parquet", "line"),
    ],
)
def test_get_geom_type_parquet(data_dir, dset_name, expected_geom_type):
    """
    Test that get_geom_type_parquet() returns correct type for known inputs.
    """
    dset_src = data_dir / dset_name
    geom_type = get_geom_type_parquet(dset_src)
    assert geom_type == expected_geom_type, "Unexpected geometry type identified"


@pytest.mark.parametrize(
    "dset_name,expected_geom_type",
    [
        ("characterize/vectors/fiber_to_the_premises.gpkg", "polygon"),
        ("characterize/vectors/generators.gpkg", "point"),
        ("characterize/vectors/tlines.gpkg", "line"),
        ("edge_case_inputs/combo_lines_multilines.gpkg", "line"),
    ],
)
def test_get_geom_type_pyogrio(data_dir, dset_name, expected_geom_type):
    """
    Test that get_geom_type_pyogrio() returns correct type for known inputs.
    """
    dset_src = data_dir / dset_name
    geom_type = get_geom_type_pyogrio(dset_src)
    assert geom_type == expected_geom_type, "Unexpected geometry type identified"


@pytest.mark.parametrize("dset_ext", ["parquet", "gpkg"])
def test_get_attributes(data_dir, dset_ext):
    """
    Test that get_attributes_parquet() and get_attributes_pyogrio() return the same
    expected results.
    """
    dset_src = data_dir / f"characterize/vectors/fiber_to_the_premises.{dset_ext}"
    if dset_ext == "parquet":
        attributes = get_attributes_parquet(dset_src)
    else:
        attributes = get_attributes_pyogrio(dset_src)
    expected_attributes = {
        "h3_res8_id": np.object_,
        "max_advertised_download_speed": np.int64,
        "max_advertised_upload_speed": np.int64,
        "low_latency": np.int64,
    }
    assert attributes == expected_attributes, "Attributes do not match expected result"


@pytest.mark.parametrize("in_format", ["parquet", "pyogrio"])
def test_get_geom_type_error_multipoint(data_dir, in_format):
    """
    Test that get_geom_type_parquet() and get_geom_type_pyogrio() both raise a
    ValueError when passed an input dataset that is MultiPoint.
    """
    if in_format == "parquet":
        dset_src = data_dir / "edge_case_inputs" / "multipoint.parquet"
        with pytest.raises(ValueError, match="Unsupported geometry type*."):
            get_geom_type_parquet(dset_src)
    elif in_format == "pyogrio":
        dset_src = data_dir / "edge_case_inputs" / "multipoint.gpkg"
        with pytest.raises(ValueError, match="Unsupported geometry type*."):
            get_geom_type_pyogrio(dset_src)


def test_get_crs_raster(data_dir):
    """
    Test for get_crs_raster()
    """
    dset_src = data_dir / "characterize" / "rasters" / "developable.tif"
    crs = get_crs_raster(dset_src)
    assert crs == "EPSG:5070", "Unexpected CRS value"


def test_get_crs_pyogrio(data_dir):
    """
    Test for get_crs_pyogrio()
    """
    dset_src = data_dir / "characterize" / "vectors" / "generators.gpkg"
    crs = get_crs_pyogrio(dset_src)
    assert crs == "EPSG:5070", "Unexpected CRS value"


def test_get_crs_parquet(data_dir):
    """
    Test for get_crs_parquet()
    """
    dset_src = data_dir / "characterize" / "vectors" / "generators.parquet"
    crs = get_crs_parquet(dset_src)
    assert crs == "EPSG:5070", "Unexpected CRS value"


@pytest.mark.parametrize(
    "vector_src,error_expected,where,row_count",
    [
        ("rasters/developable.tif", True, None, None),
        ("vectors/generators.gpkg", False, None, 12),
        ("vectors/generators.parquet", False, None, 12),
        ("vectors/generators.gpkg", False, "capacity_factor < 0.5", 9),
        ("vectors/generators.parquet", False, "capacity_factor < 0.05", 7),
    ],
)
def test_read_vectors(data_dir, vector_src, error_expected, where, row_count):
    """
    Test for read_vectors() for different input file formats.
    """
    vector_src_path = data_dir / "characterize" / vector_src
    if error_expected:
        with pytest.raises(IOError, match="Unable to read vectors from input file.*"):
            read_vectors(vector_src_path, where=where)
    else:
        df = read_vectors(vector_src_path, where=where)
        assert len(df) == row_count, "Unexpected row count in GeoDataFrame"


@pytest.mark.parametrize(
    "bad_expression,err",
    [
        ("@gpd.pd.compat.os.system('echo foo')", UndefinedVariableError),
        ("os.system('echo foo')", UndefinedVariableError),
        ("import os", NotImplementedError),
    ],
)
def test_read_vectors_with_expression_injection(data_dir, capfd, bad_expression, err):
    """
    Unit test that ensures that attempts to inject system level commmands using
    where clause does not work.
    """
    vector_src_path = data_dir / "characterize/vectors/generators.gpkg"

    with pytest.raises(err):
        read_vectors(vector_src_path, where=bad_expression)

    captured_stdout = capfd.readouterr().out
    assert (
        captured_stdout == ""
    ), "stdout is not empty. Injection occurred via dataframe.eval()."


@pytest.mark.parametrize(
    "attribute,expected_result,err",
    [
        ("net_generation_megawatthours", True, False),
        ("plant_code", True, False),
        ("primary_fuel_type", False, False),
        ("not-a-col", None, True),
    ],
)
@pytest.mark.parametrize("dset_ext", ["gpkg", "parquet", "tif"])
def test_attribute_is_numeric(data_dir, attribute, expected_result, err, dset_ext):
    """
    Unit tests for attribute_is_numeric(). Check that it returns the expected results
    and/or raises the expected errors.
    """

    vector_src = data_dir / "characterize" / "vectors" / f"generators.{dset_ext}"
    if dset_ext == "tif":
        with pytest.raises(OSError, match="Unable to read input vector file"):
            attribute_is_numeric(vector_src, attribute)
    elif err:
        with pytest.raises(ValueError, match=f"Attribute {attribute} not found"):
            attribute_is_numeric(vector_src, attribute)
    else:
        result = attribute_is_numeric(vector_src, attribute)
        assert result == expected_result, "Unexpected result for attribute_is_numeric"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-k", "test_attribute_is_numeric"])
