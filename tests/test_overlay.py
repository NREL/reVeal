# -*- coding: utf-8 -*-
"""
overlay module tests
"""
import inspect

import pytest
from geopandas.testing import assert_geodataframe_equal
import pandas as pd
import geopandas as gpd

from reVeal.overlay import (
    calc_feature_count,
    calc_sum_attribute,
    calc_sum_length,
    calc_sum_attribute_length,
    calc_sum_area,
    calc_percent_covered,
    calc_area_weighted_average,
    calc_area_apportioned_sum,
    zonal_statistic,
    calc_median,
    calc_mean,
    calc_sum,
    calc_area,
)
from reVeal.grid import get_method_from_members, OVERLAY_METHODS
from reVeal.config.characterize import VALID_CHARACTERIZATION_METHODS


@pytest.mark.parametrize("where", [None, "capacity_factor < 0.5"])
def test_calc_feature_count(data_dir, base_grid, where):
    """
    Unit tests for calc_feature_count().
    """
    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / "generators.gpkg"
    results = calc_feature_count(zones_df, dset_src, where=where)

    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    if where:
        expected_results_src = (
            data_dir / "overlays" / "feature_count_results_filtered.gpkg"
        )
    else:
        expected_results_src = data_dir / "overlays" / "feature_count_results.gpkg"

    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "attribute,where,exception_type",
    [
        ("net_generation_megawatthours", None, None),
        ("utility_name", None, TypeError),
        ("not_a_column", None, KeyError),
        ("net_generation_megawatthours", "capacity_factor < 0.5", None),
    ],
)
def test_calc_sum_attribute(data_dir, base_grid, attribute, where, exception_type):
    """
    Unit tests for calc_sum_attribute().
    """
    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / "generators.gpkg"

    if exception_type:
        with pytest.raises(exception_type):
            calc_sum_attribute(zones_df, dset_src, attribute, where=where)
    else:
        results = calc_sum_attribute(zones_df, dset_src, attribute, where=where)

        results_df = pd.concat([zones_df, results], axis=1)
        results_df.reset_index(inplace=True)

        if where:
            expected_results_src = (
                data_dir / "overlays" / "feature_sum_results_filtered.gpkg"
            )
        else:
            expected_results_src = data_dir / "overlays" / "feature_sum_results.gpkg"
        expected_df = gpd.read_file(expected_results_src)

        assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "dset_name,where",
    [
        ("tlines", None),
        ("generators", None),
        ("fiber_to_the_premises", None),
        ("tlines", "(VOLTAGE == 345) & (OWNER == 'CENTERPOINT ENERGY')"),
        ("generators", "capacity_factor < 0.5"),
        ("fiber_to_the_premises", "max_advertised_upload_speed < 50"),
    ],
)
def test_calc_sum_length(data_dir, base_grid, dset_name, where):
    """
    Unit tests for calc_sum_length().
    """

    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / f"{dset_name}.gpkg"

    results = calc_sum_length(zones_df, dset_src, where=where)

    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    if where:
        expected_results_src = (
            data_dir / "overlays" / f"sum_length_results_{dset_name}_filtered.gpkg"
        )
    else:
        expected_results_src = (
            data_dir / "overlays" / f"sum_length_results_{dset_name}.gpkg"
        )

    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "attribute,where,exception_type",
    [
        ("VOLTAGE", None, None),
        ("INFERRED", None, TypeError),
        ("not_a_column", None, KeyError),
        ("VOLTAGE", "(VOLTAGE == 345) & (OWNER == 'CENTERPOINT ENERGY')", None),
    ],
)
def test_calc_sum_attribute_length(
    data_dir, base_grid, attribute, where, exception_type
):
    """
    Unit tests for calc_sum_attribute().
    """
    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / "tlines.gpkg"

    if exception_type:
        with pytest.raises(exception_type):
            calc_sum_attribute_length(zones_df, dset_src, attribute, where=where)
    else:
        results = calc_sum_attribute_length(zones_df, dset_src, attribute, where=where)

        results_df = pd.concat([zones_df, results], axis=1)
        results_df.reset_index(inplace=True)

        if where:
            expected_results_src = (
                data_dir / "overlays" / "sum_attribute_length_results_filtered.gpkg"
            )
        else:
            expected_results_src = (
                data_dir / "overlays" / "sum_attribute_length_results.gpkg"
            )
        expected_df = gpd.read_file(expected_results_src)

        assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "dset_name,where,all_zeros",
    [
        ("fiber_to_the_premises.gpkg", None, False),
        ("tlines.gpkg", None, True),
        ("generators.gpkg", None, True),
        ("fiber_to_the_premises.gpkg", "max_advertised_upload_speed < 50", False),
    ],
)
def test_calc_sum_area(data_dir, base_grid, dset_name, where, all_zeros):
    """
    Unit tests for calc_sum_area().
    """

    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / dset_name

    results = calc_sum_area(zones_df, dset_src, where=where)
    if all_zeros:
        assert (results["value"] == 0).all(), "Results are not all zero as expected"
    else:
        results_df = pd.concat([zones_df, results], axis=1)
        results_df.reset_index(inplace=True)

        if where:
            expected_results_src = (
                data_dir / "overlays" / "sum_area_results_filtered.gpkg"
            )
        else:
            expected_results_src = data_dir / "overlays" / "sum_area_results.gpkg"

        expected_df = gpd.read_file(expected_results_src)

        assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "dset_name,where,all_zeros",
    [
        ("fiber_to_the_premises.gpkg", None, False),
        ("tlines.gpkg", None, True),
        ("generators.gpkg", None, True),
        ("fiber_to_the_premises.gpkg", "max_advertised_upload_speed < 50", False),
    ],
)
def test_calc_percent_covered(data_dir, base_grid, dset_name, where, all_zeros):
    """
    Unit tests for calc_percent_covered().
    """

    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / dset_name

    results = calc_percent_covered(zones_df, dset_src, where=where)
    if all_zeros:
        assert (results["value"] == 0).all(), "Results are not all zero as expected"
    else:
        results_df = pd.concat([zones_df, results], axis=1)
        results_df.reset_index(inplace=True)

        if where:
            expected_results_src = (
                data_dir / "overlays" / "percent_covered_results_filtered.gpkg"
            )
        else:
            expected_results_src = (
                data_dir / "overlays" / "percent_covered_results.gpkg"
            )
        expected_df = gpd.read_file(expected_results_src)

        assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "dset_name,attribute,where,exception_type,all_nans",
    [
        (
            "fiber_to_the_premises.gpkg",
            "max_advertised_upload_speed",
            None,
            None,
            False,
        ),
        ("tlines.gpkg", "VOLTAGE", None, None, True),
        ("generators.gpkg", "net_generation_megawatthours", None, None, True),
        ("fiber_to_the_premises.gpkg", "h3_res8_id", None, TypeError, False),
        ("fiber_to_the_premises.gpkg", "not_a_column", None, KeyError, False),
        (
            "fiber_to_the_premises.gpkg",
            "max_advertised_upload_speed",
            "max_advertised_upload_speed <= 50",
            None,
            False,
        ),
    ],
)
def test_calc_area_weighted_average(
    data_dir, base_grid, dset_name, attribute, where, exception_type, all_nans
):
    """
    Unit tests for calc_area_weighted_average().
    """

    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / dset_name

    if exception_type:
        with pytest.raises(exception_type):
            calc_area_weighted_average(zones_df, dset_src, attribute, where=where)
    else:
        results = calc_area_weighted_average(zones_df, dset_src, attribute, where=where)
        if all_nans:
            assert (results["value"].isna()).all(), "Results are not all NA as expected"
        else:
            results_df = pd.concat([zones_df, results], axis=1)
            results_df.reset_index(inplace=True)

            if where:
                expected_results_src = (
                    data_dir
                    / "overlays"
                    / "area_weighted_average_results_filtered.gpkg"
                )
            else:
                expected_results_src = (
                    data_dir / "overlays" / "area_weighted_average_results.gpkg"
                )
            expected_df = gpd.read_file(expected_results_src)

            assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "dset_name,attribute,where,exception_type,all_zeros",
    [
        (
            "fiber_to_the_premises.gpkg",
            "max_advertised_upload_speed",
            None,
            None,
            False,
        ),
        ("tlines.gpkg", "VOLTAGE", None, None, True),
        ("generators.gpkg", "net_generation_megawatthours", None, None, True),
        ("fiber_to_the_premises.gpkg", "h3_res8_id", None, TypeError, False),
        ("fiber_to_the_premises.gpkg", "not_a_column", None, KeyError, False),
        (
            "fiber_to_the_premises.gpkg",
            "max_advertised_upload_speed",
            "max_advertised_upload_speed <= 50",
            None,
            False,
        ),
    ],
)
def test_calc_area_apportioned_sum(
    data_dir, base_grid, dset_name, attribute, where, exception_type, all_zeros
):
    """
    Unit tests for calc_area_weighted_average().
    """

    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "vectors" / dset_name

    if exception_type:
        with pytest.raises(exception_type):
            calc_area_apportioned_sum(zones_df, dset_src, attribute, where=where)
    else:
        results = calc_area_apportioned_sum(zones_df, dset_src, attribute, where=where)
        if all_zeros:
            assert (results["value"] == 0).all(), "Results are not all zero as expected"
        else:
            results_df = pd.concat([zones_df, results], axis=1)
            results_df.reset_index(inplace=True)

            if where:
                expected_results_src = (
                    data_dir / "overlays" / "area_apportioned_sum_results_filtered.gpkg"
                )
            else:
                expected_results_src = (
                    data_dir / "overlays" / "area_apportioned_sum_results.gpkg"
                )
            expected_df = gpd.read_file(expected_results_src)

            assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize(
    "stat,weighted",
    [
        ("median", False),
        ("count", False),
        ("mean", False),
        ("sum", False),
        ("mean", True),
        ("sum", True),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_zonal_statistic(data_dir, base_grid, stat, weighted, parallel):
    """
    Unit tests for zonal_statistic().
    """

    zones_df = base_grid.df
    dset_src = (
        data_dir / "characterize" / "rasters" / "fiber_lines_onshore_proximity.tif"
    )
    if weighted:
        weights_src = data_dir / "characterize" / "rasters" / "developable.tif"
    else:
        weights_src = None

    results = zonal_statistic(
        zones_df, dset_src, stat=stat, weights_dset_src=weights_src, parallel=parallel
    )
    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = (
        data_dir / "overlays" / f"zonal_{stat}_weighted_{weighted}.gpkg"
    )
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


def test_calc_median(data_dir, base_grid):
    """
    Unit tests for calc_median().
    """

    zones_df = base_grid.df
    dset_src = (
        data_dir / "characterize" / "rasters" / "fiber_lines_onshore_proximity.tif"
    )
    results = calc_median(zones_df, dset_src)
    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = data_dir / "overlays" / "zonal_median_weighted_False.gpkg"
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize("weighted", [True, False])
def test_calc_mean(data_dir, base_grid, weighted):
    """
    Unit tests for calc_mean().
    """

    zones_df = base_grid.df
    dset_src = (
        data_dir / "characterize" / "rasters" / "fiber_lines_onshore_proximity.tif"
    )
    if weighted:
        weights_src = data_dir / "characterize" / "rasters" / "developable.tif"
    else:
        weights_src = None

    results = calc_mean(zones_df, dset_src, weights_dset_src=weights_src)
    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = (
        data_dir / "overlays" / f"zonal_mean_weighted_{weighted}.gpkg"
    )
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize("weighted", [True, False])
def test_calc_sum(data_dir, base_grid, weighted):
    """
    Unit tests for calc_sum().
    """

    zones_df = base_grid.df
    dset_src = (
        data_dir / "characterize" / "rasters" / "fiber_lines_onshore_proximity.tif"
    )
    if weighted:
        weights_src = data_dir / "characterize" / "rasters" / "developable.tif"
    else:
        weights_src = None

    results = calc_sum(zones_df, dset_src, weights_dset_src=weights_src)
    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = data_dir / "overlays" / f"zonal_sum_weighted_{weighted}.gpkg"
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


@pytest.mark.parametrize("weighted", [True, False])
def test_calc_area(data_dir, base_grid, weighted):
    """
    Unit tests for calc_area().
    """

    zones_df = base_grid.df
    dset_src = data_dir / "characterize" / "rasters" / "fiber_availability_binary.tif"
    if weighted:
        weights_src = data_dir / "characterize" / "rasters" / "developable.tif"
    else:
        weights_src = None

    results = calc_area(zones_df, dset_src, weights_dset_src=weights_src)
    results_df = pd.concat([zones_df, results], axis=1)
    results_df.reset_index(inplace=True)

    expected_results_src = (
        data_dir / "overlays" / f"area_results_weighted_{weighted}.gpkg"
    )
    expected_df = gpd.read_file(expected_results_src)

    assert_geodataframe_equal(results_df, expected_df, check_like=True)


def test_check_where():
    """
    Check that overlay that are specified in the config as supporting use of a where
    clause have a where argument in their function signature. Since there logic of
    the where clause is applied within each applicable overlay method, this is
    effectively a backstop to ensure that, at a minimum, the function accepts it as
    an input. Note that it does not actually ensure that the `where` argument is
    actually passed to read_vectors(), which is where it is actually applied.
    """
    where_methods = [
        m for m, i in VALID_CHARACTERIZATION_METHODS.items() if i.get("supports_where")
    ]
    for method in where_methods:
        overlay_method = get_method_from_members(method, OVERLAY_METHODS)
        args = inspect.getfullargspec(overlay_method).args
        assert "where" in args, (
            f"Function calc_{overlay_method} is identified as supporting where clause "
            "but does not have a where argument in its signature"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
