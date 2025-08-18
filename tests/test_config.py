# -*- coding: utf-8 -*-
"""
config module tests
"""
import pytest

from pydantic import ValidationError

from loci.config import (
    Characterization,
    VALID_CHARACTERIZATION_METHODS,
)


@pytest.mark.parametrize("method", list(VALID_CHARACTERIZATION_METHODS.keys()))
@pytest.mark.parametrize("attribute", [None, "out_attr"])
@pytest.mark.parametrize("apply_exclusions", [None, True, False])
@pytest.mark.parametrize("neighbor_order", [None, 0, 1, 50.0])
@pytest.mark.parametrize("buffer_distance", [None, -100, 100])
def test_characterization_valid(
    method, attribute, apply_exclusions, neighbor_order, buffer_distance
):
    """
    Test Characterization class with valid inputs.
    """

    value = {
        "dset": "test/dset.gpkg",
        "method": method,
        "attribute": attribute,
        "apply_exclusions": apply_exclusions,
        "neighbor_order": neighbor_order,
        "buffer_distance": buffer_distance,
    }

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


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
