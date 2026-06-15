# -*- coding: utf-8 -*-
"""
config.learn_weights module tests
"""
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from pydantic import ValidationError

from reVeal.config.learn_weights import LearnWeightsConfig


@pytest.fixture
def grid_and_labels(tmp_path):
    """Create a synthetic grid and labels saved as GPKG files."""
    rng = np.random.default_rng(42)
    n_cells = 50

    geoms = [box(i % 10, i // 10, (i % 10) + 1, (i // 10) + 1) for i in range(n_cells)]
    data = {
        "gid": list(range(n_cells)),
        "feature_a_score": rng.uniform(0, 1, n_cells),
        "feature_b_score": rng.uniform(0, 1, n_cells),
        "geometry": geoms,
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    grid_path = tmp_path / "grid.gpkg"
    gdf.to_file(grid_path, driver="GPKG")

    points = [Point(0.5, 0.5), Point(1.5, 0.5), Point(2.5, 0.5)]
    labels = gpd.GeoDataFrame({"id": [0, 1, 2], "geometry": points}, crs="EPSG:4326")
    labels_path = tmp_path / "labels.gpkg"
    labels.to_file(labels_path, driver="GPKG")

    return grid_path, labels_path


class TestLearnWeightsConfigValid:
    """Tests for valid LearnWeightsConfig creation."""

    def test_minimal_config(self, grid_and_labels):
        """Test creating config with only required fields."""
        grid_path, labels_path = grid_and_labels
        config = LearnWeightsConfig(grid=grid_path, labels=labels_path)
        assert config.n_estimators == 500
        assert config.attributes is None
        assert config.exclude_attributes is None
        assert config.tune is False

    def test_with_attributes(self, grid_and_labels):
        """Test config with explicit attributes."""
        grid_path, labels_path = grid_and_labels
        config = LearnWeightsConfig(
            grid=grid_path,
            labels=labels_path,
            attributes=["feature_a_score", "feature_b_score"],
        )
        assert config.attributes == ["feature_a_score", "feature_b_score"]

    def test_with_exclude_attributes(self, grid_and_labels):
        """Test config with exclude_attributes."""
        grid_path, labels_path = grid_and_labels
        config = LearnWeightsConfig(
            grid=grid_path,
            labels=labels_path,
            exclude_attributes=["feature_b_score"],
        )
        assert config.exclude_attributes == ["feature_b_score"]

    def test_custom_parameters(self, grid_and_labels):
        """Test config with custom model parameters."""
        grid_path, labels_path = grid_and_labels
        config = LearnWeightsConfig(
            grid=grid_path,
            labels=labels_path,
            n_estimators=100,
            class_prior=0.3,
            background_samples=500,
            test_size=0.3,
            n_jobs=4,
            random_state=123,
            tune=True,
            n_trials=50,
            tuning_metric="tpr",
        )
        assert config.n_estimators == 100
        assert config.class_prior == 0.3
        assert config.background_samples == 500
        assert config.test_size == 0.3
        assert config.n_jobs == 4
        assert config.random_state == 123
        assert config.tune is True
        assert config.n_trials == 50
        assert config.tuning_metric == "tpr"


class TestLearnWeightsConfigInvalid:
    """Tests for invalid LearnWeightsConfig inputs."""

    def test_nonexistent_grid(self, tmp_path):
        """Test that a nonexistent grid raises ValidationError."""
        labels_path = tmp_path / "labels.gpkg"
        gpd.GeoDataFrame(
            {"id": [0], "geometry": [Point(0, 0)]}, crs="EPSG:4326"
        ).to_file(labels_path, driver="GPKG")

        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=tmp_path / "no_such_grid.gpkg",
                labels=labels_path,
            )

    def test_nonexistent_labels(self, grid_and_labels):
        """Test that a nonexistent labels file raises ValidationError."""
        grid_path, _ = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path,
                labels="/no/such/labels.gpkg",
            )

    def test_mutual_exclusion(self, grid_and_labels):
        """Test that both attributes and exclude_attributes raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError, match="Only one of"):
            LearnWeightsConfig(
                grid=grid_path,
                labels=labels_path,
                attributes=["a_score"],
                exclude_attributes=["b_score"],
            )

    def test_n_estimators_too_low(self, grid_and_labels):
        """Test that n_estimators < 10 raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, n_estimators=5
            )

    def test_n_estimators_too_high(self, grid_and_labels):
        """Test that n_estimators > 10000 raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, n_estimators=20000
            )

    def test_class_prior_out_of_range(self, grid_and_labels):
        """Test that class_prior outside (0,1) raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, class_prior=0.0
            )
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, class_prior=1.0
            )

    def test_test_size_out_of_range(self, grid_and_labels):
        """Test that test_size outside (0,1) raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, test_size=0.0
            )
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, test_size=1.0
            )

    def test_background_samples_too_low(self, grid_and_labels):
        """Test that background_samples < 100 raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, background_samples=50
            )

    def test_invalid_tuning_metric(self, grid_and_labels):
        """Test that invalid tuning_metric raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, tuning_metric="f1"
            )

    def test_n_trials_too_low(self, grid_and_labels):
        """Test that n_trials < 1 raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, n_trials=0
            )

    def test_n_jobs_too_low(self, grid_and_labels):
        """Test that n_jobs < 1 raises."""
        grid_path, labels_path = grid_and_labels
        with pytest.raises(ValidationError):
            LearnWeightsConfig(
                grid=grid_path, labels=labels_path, n_jobs=0
            )
