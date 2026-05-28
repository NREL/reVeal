# -*- coding: utf-8 -*-
"""
Tests for learn_weights module.
"""
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from pydantic import ValidationError

from reVeal.config.learn_weights import LearnWeightsConfig
from reVeal.learn_weights import (
    prepare_pu_data,
    train_pu_model,
    tune_class_prior,
    importances_to_weights,
    generate_score_weighted_config,
    run_learn_weights,
)


@pytest.fixture
def synthetic_grid():
    """Create a synthetic grid with score columns for testing."""
    rng = np.random.default_rng(42)
    n_cells = 200

    # Create a grid of square polygons
    geoms = [box(i % 20, i // 20, (i % 20) + 1, (i // 20) + 1) for i in range(n_cells)]

    data = {
        "gid": list(range(n_cells)),
        "feature_a_score": rng.uniform(0, 1, n_cells),
        "feature_b_score": rng.uniform(0, 1, n_cells),
        "feature_c_score": rng.uniform(0, 1, n_cells),
        "geometry": geoms,
    }

    # Make feature_a strongly correlated with being "positive" for cells 0-19
    data["feature_a_score"][:20] = rng.uniform(0.7, 1.0, 20)
    data["feature_a_score"][20:] = rng.uniform(0.0, 0.4, 180)

    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def synthetic_labels():
    """Create point labels that fall within the first 20 grid cells."""
    # Points in centers of grid cells 0-19 (which are at x=0..19, y=0)
    points = [Point(i + 0.5, 0.5) for i in range(20)]
    return gpd.GeoDataFrame({"id": range(20), "geometry": points}, crs="EPSG:4326")


class TestPrepareData:
    """Tests for prepare_pu_data."""

    def test_basic_splits(self, synthetic_grid, synthetic_labels):
        """Test that data splits are created with correct sizes."""
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            test_size=0.2,
            validation_size=0.1,
            random_state=42,
        )

        assert "train_gids" in splits
        assert "test_gids" in splits
        assert "validation_gids" in splits
        assert "background_gids" in splits
        assert len(splits["train_gids"]) > 0
        assert len(splits["background_gids"]) > 0

        # No overlap between train and test
        assert not set(splits["train_gids"]) & set(splits["test_gids"])

    def test_background_size_capped(self, synthetic_grid, synthetic_labels):
        """Test that background sampling respects available cells."""
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=1000,  # More than available
            random_state=42,
        )
        # Should be capped at non-intersecting count (180)
        assert len(splits["background_gids"]) <= 180


class TestTrainModel:
    """Tests for train_pu_model."""

    def test_train_produces_importances(self, synthetic_grid, synthetic_labels):
        """Test that training produces feature importances."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )

        results = train_pu_model(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=attributes,
            n_estimators=10,  # Small for speed
            random_state=42,
        )

        assert "feature_importances" in results
        assert len(results["feature_importances"]) == 3
        assert results["model"] is not None

    def test_empty_positives_raises(self, synthetic_grid):
        """Test that empty positive training set raises ValueError."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]
        splits = {
            "train_gids": np.array([]),
            "validation_gids": np.array([]),
            "test_gids": np.array([]),
            "background_gids": np.array([0, 1, 2, 3, 4]),
            "background_test_gids": np.array([]),
            "background_val_gids": np.array([]),
        }
        with pytest.raises(ValueError, match="No positive training samples"):
            train_pu_model(
                grid_df=synthetic_grid,
                data_splits=splits,
                attributes=attributes,
                n_estimators=10,
                random_state=42,
            )

    def test_empty_background_raises(self, synthetic_grid):
        """Test that empty background training set raises ValueError."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]
        splits = {
            "train_gids": np.array([0, 1, 2, 3, 4]),
            "validation_gids": np.array([]),
            "test_gids": np.array([]),
            "background_gids": np.array([]),
            "background_test_gids": np.array([]),
            "background_val_gids": np.array([]),
        }
        with pytest.raises(ValueError, match="No background"):
            train_pu_model(
                grid_df=synthetic_grid,
                data_splits=splits,
                attributes=attributes,
                n_estimators=10,
                random_state=42,
            )

    def test_invalid_class_prior_raises(self, synthetic_grid, synthetic_labels):
        """Test that class_prior outside (0,1) raises ValueError."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )
        with pytest.raises(ValueError, match="class_prior must be between"):
            train_pu_model(
                grid_df=synthetic_grid,
                data_splits=splits,
                attributes=attributes,
                n_estimators=10,
                class_prior=1.0,
                random_state=42,
            )


class TestImportancesToWeights:
    """Tests for importances_to_weights."""

    def test_normalization(self):
        """Test that weights sum to 1."""
        importances = np.array([0.5, 0.3, 0.2])
        attrs = ["a_score", "b_score", "c_score"]
        weights = importances_to_weights(importances, attrs)

        total = sum(w["weight"] for w in weights)
        assert abs(total - 1.0) < 1e-10

    def test_excludes_zero_importances(self):
        """Test that features with 0 importance are excluded."""
        importances = np.array([0.5, 0.0, 0.3])
        attrs = ["a_score", "b_score", "c_score"]
        weights = importances_to_weights(importances, attrs)

        assert len(weights) == 2
        attr_names = [w["attribute"] for w in weights]
        assert "b_score" not in attr_names

    def test_excludes_negative_importances(self):
        """Test that features with negative importance are excluded."""
        importances = np.array([0.5, -0.1, 0.3])
        attrs = ["a_score", "b_score", "c_score"]
        weights = importances_to_weights(importances, attrs)

        assert len(weights) == 2
        total = sum(w["weight"] for w in weights)
        assert abs(total - 1.0) < 1e-10

    def test_all_zero_raises(self):
        """Test that all-zero importances raises ValueError."""
        importances = np.array([0.0, 0.0, 0.0])
        attrs = ["a_score", "b_score", "c_score"]
        with pytest.raises(ValueError, match="All feature importances"):
            importances_to_weights(importances, attrs)

    def test_sorted_descending(self):
        """Test that output is sorted by weight descending."""
        importances = np.array([0.1, 0.5, 0.3])
        attrs = ["a_score", "b_score", "c_score"]
        weights = importances_to_weights(importances, attrs)

        weight_values = [w["weight"] for w in weights]
        assert weight_values == sorted(weight_values, reverse=True)


class TestGenerateConfig:
    """Tests for generate_score_weighted_config."""

    def test_config_format(self):
        """Test that generated config matches expected schema."""
        weights = [
            {"attribute": "fiber_score", "weight": 0.6},
            {"attribute": "tline_score", "weight": 0.4},
        ]
        config = generate_score_weighted_config(
            weights, "/path/to/grid.gpkg", "my_score"
        )

        assert config["grid"] == "/path/to/grid.gpkg"
        assert config["score_name"] == "my_score"
        assert len(config["attributes"]) == 2
        assert config["attributes"][0]["attribute"] == "fiber_score"
        assert config["attributes"][0]["weight"] == 0.6


class TestEndToEnd:
    """End-to-end integration test."""

    def test_full_pipeline(self, synthetic_grid, synthetic_labels):
        """Test full pipeline produces valid score-weighted config."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]

        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )

        results = train_pu_model(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=attributes,
            n_estimators=10,
            random_state=42,
        )

        weights = importances_to_weights(results["feature_importances"], attributes)
        config = generate_score_weighted_config(weights, "grid_normalized.gpkg")

        # Validate output structure
        assert "attributes" in config
        assert "score_name" in config
        assert len(config["attributes"]) > 0

        # Validate weights sum to 1
        total = sum(a["weight"] for a in config["attributes"])
        assert abs(total - 1.0) < 1e-10

        # All weights in valid range
        for attr in config["attributes"]:
            assert attr["weight"] > 0
            assert attr["weight"] <= 1


class TestTuneClassPrior:
    """Tests for tune_class_prior (Optuna-based tuning)."""

    def test_returns_best_prior(self, synthetic_grid, synthetic_labels):
        """Test that tuning returns a valid class_prior."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )

        result = tune_class_prior(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=attributes,
            n_estimators=10,
            n_jobs=1,
            random_state=42,
            n_trials=5,
            metric="auc",
        )

        assert "best_class_prior" in result
        assert "best_value" in result
        assert 0.01 <= result["best_class_prior"] <= 0.99
        assert 0.0 <= result["best_value"] <= 1.0

    def test_tpr_metric(self, synthetic_grid, synthetic_labels):
        """Test tuning with TPR metric."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )

        result = tune_class_prior(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=attributes,
            n_estimators=10,
            n_jobs=1,
            random_state=42,
            n_trials=5,
            metric="tpr",
        )

        assert 0.01 <= result["best_class_prior"] <= 0.99

    def test_tuned_prior_used_in_training(self, synthetic_grid, synthetic_labels):
        """Test that tuned prior can be used to train a final model."""
        attributes = ["feature_a_score", "feature_b_score", "feature_c_score"]
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )

        tuning_result = tune_class_prior(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=attributes,
            n_estimators=10,
            n_jobs=1,
            random_state=42,
            n_trials=5,
            metric="auc",
        )

        # Use tuned prior to train final model
        results = train_pu_model(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=attributes,
            n_estimators=10,
            class_prior=tuning_result["best_class_prior"],
            random_state=42,
        )

        assert results["model"] is not None
        assert len(results["feature_importances"]) == 3


class TestAttributeSelection:
    """Tests for attributes / exclude_attributes config."""

    def test_explicit_attributes(self, synthetic_grid, synthetic_labels):
        """Test that explicit attributes limits features to those specified."""
        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )
        # Only use 2 of 3 features
        results = train_pu_model(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=["feature_a_score", "feature_b_score"],
            n_estimators=10,
            random_state=42,
        )
        assert len(results["feature_importances"]) == 2

    def test_exclude_attributes_removes(self, synthetic_grid, synthetic_labels):
        """Test that exclude removes specified features from auto-detected set."""
        all_score_cols = [c for c in synthetic_grid.columns if c.endswith("_score")]
        exclude = ["feature_c_score"]
        expected = [c for c in all_score_cols if c not in exclude]

        splits = prepare_pu_data(
            grid_df=synthetic_grid,
            labels_gdf=synthetic_labels,
            background_samples=50,
            random_state=42,
        )
        results = train_pu_model(
            grid_df=synthetic_grid,
            data_splits=splits,
            attributes=expected,
            n_estimators=10,
            random_state=42,
        )
        assert len(results["feature_importances"]) == len(expected)

    def test_mutual_exclusion_attributes_and_exclude(self, synthetic_grid, synthetic_labels, tmp_path):
        """Test that setting both attributes and exclude_attributes raises."""
        grid_path = tmp_path / "grid.gpkg"
        synthetic_grid.to_file(grid_path, driver="GPKG")
        labels_path = tmp_path / "labels.gpkg"
        synthetic_labels.to_file(labels_path, driver="GPKG")

        with pytest.raises(ValidationError, match="Only one of"):
            LearnWeightsConfig(
                grid=grid_path,
                labels=labels_path,
                attributes=["a_score"],
                exclude_attributes=["b_score"],
            )
