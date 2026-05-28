# -*- coding: utf-8 -*-
"""
Tests for feature_analysis module.
"""
import json

import numpy as np
import pandas as pd
import pytest

from reVeal.feature_analysis import (
    compute_correlation_matrix,
    compute_feature_clusters,
    plot_dendrogram,
    plot_correlation_heatmap,
    suggest_exclusions,
    save_analysis_outputs,
)


@pytest.fixture
def simple_data():
    """Create simple correlated data for testing."""
    rng = np.random.default_rng(42)
    n = 100
    # a and b are highly correlated; c is independent
    a = rng.uniform(0, 1, n)
    b = a + rng.normal(0, 0.05, n)  # strongly correlated with a
    c = rng.uniform(0, 1, n)  # independent
    X = np.column_stack([a, b, c])
    columns = ["feat_a_score", "feat_b_score", "feat_c_score"]
    return X, columns


@pytest.fixture
def five_feature_data():
    """Create 5-feature data with known cluster structure."""
    rng = np.random.default_rng(123)
    n = 200
    # Cluster 1: a, b (correlated)
    a = rng.uniform(0, 1, n)
    b = a * 0.9 + rng.normal(0, 0.1, n)
    # Cluster 2: c, d (correlated)
    c = rng.uniform(0, 1, n)
    d = c * 0.85 + rng.normal(0, 0.12, n)
    # Standalone: e
    e = rng.uniform(0, 1, n)
    X = np.column_stack([a, b, c, d, e])
    columns = ["a_score", "b_score", "c_score", "d_score", "e_score"]
    return X, columns


class TestComputeCorrelationMatrix:
    """Tests for compute_correlation_matrix."""

    def test_spearman_output_shape(self, simple_data):
        """Test that output is a square DataFrame with correct shape."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)
        assert list(corr.columns) == columns
        assert list(corr.index) == columns

    def test_pearson_output_shape(self, simple_data):
        """Test pearson method also works."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="pearson")
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self, simple_data):
        """Test that diagonal elements are 1.0."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        for i in range(len(columns)):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-10

    def test_symmetry(self, simple_data):
        """Test that correlation matrix is symmetric."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_high_correlation_detected(self, simple_data):
        """Test that correlated features have high correlation."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        # a and b should be highly correlated
        assert abs(corr.loc["feat_a_score", "feat_b_score"]) > 0.9
        # c should be less correlated
        assert abs(corr.loc["feat_a_score", "feat_c_score"]) < 0.5

    def test_invalid_method_raises(self, simple_data):
        """Test that invalid method raises ValueError."""
        X, columns = simple_data
        with pytest.raises(ValueError, match="Unknown correlation method"):
            compute_correlation_matrix(X, columns, method="kendall")

    def test_constant_column_dropped(self):
        """Test that constant columns are dropped."""
        X = np.array([[1, 2, 5], [1, 3, 6], [1, 4, 7]], dtype=float)
        columns = ["const", "vary1", "vary2"]
        corr = compute_correlation_matrix(X, columns, method="spearman")
        assert "const" not in corr.columns
        assert corr.shape == (2, 2)


class TestComputeFeatureClusters:
    """Tests for compute_feature_clusters."""

    def test_returns_expected_keys(self, simple_data):
        """Test that output has expected structure."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        result = compute_feature_clusters(corr, threshold=0.7)
        assert "clusters" in result
        assert "linkage" in result
        assert "dendrogram" in result

    def test_correlated_features_cluster_together(self, simple_data):
        """Test that highly correlated features end up in same cluster."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        result = compute_feature_clusters(corr, threshold=0.7)
        clusters = result["clusters"]

        # Find which cluster contains feat_a and feat_b
        a_cluster = None
        b_cluster = None
        for cid, features in clusters.items():
            if "feat_a_score" in features:
                a_cluster = cid
            if "feat_b_score" in features:
                b_cluster = cid
        assert a_cluster == b_cluster

    def test_all_features_assigned(self, five_feature_data):
        """Test that all features appear in exactly one cluster."""
        X, columns = five_feature_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        result = compute_feature_clusters(corr, threshold=0.7)
        clusters = result["clusters"]

        all_features = []
        for features in clusters.values():
            all_features.extend(features)
        assert sorted(all_features) == sorted(columns)

    def test_low_threshold_more_clusters(self, five_feature_data):
        """Test that lower threshold produces more clusters."""
        X, columns = five_feature_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        result_low = compute_feature_clusters(corr, threshold=0.3)
        result_high = compute_feature_clusters(corr, threshold=1.5)
        assert len(result_low["clusters"]) >= len(result_high["clusters"])


class TestPlotDendrogram:
    """Tests for plot_dendrogram."""

    def test_creates_file(self, simple_data, tmp_path):
        """Test that a PNG file is created."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        result = compute_feature_clusters(corr, threshold=0.7)
        save_path = tmp_path / "dendro.png"
        plot_dendrogram(corr, result["linkage"], save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap."""

    def test_creates_file(self, simple_data, tmp_path):
        """Test that a PNG file is created."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        result = compute_feature_clusters(corr, threshold=0.7)
        save_path = tmp_path / "heatmap.png"
        plot_correlation_heatmap(corr, result["dendrogram"], save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestSuggestExclusions:
    """Tests for suggest_exclusions."""

    def test_suggests_most_redundant(self):
        """Test that the most redundant feature in a cluster is suggested for drop."""
        # Build a correlation matrix where b_score is most correlated with others
        corr_data = np.array([
            [1.0, 0.9, 0.3],
            [0.9, 1.0, 0.8],
            [0.3, 0.8, 1.0],
        ])
        columns = ["a_score", "b_score", "c_score"]
        corr_matrix = pd.DataFrame(corr_data, index=columns, columns=columns)

        clusters = {
            "cluster_1": ["a_score", "b_score", "c_score"],
            "cluster_2": ["d_score"],
        }

        suggestions = suggest_exclusions(clusters, corr_matrix)

        # Only cluster_1 has >1 feature
        assert "cluster_1" in suggestions
        assert "cluster_2" not in suggestions
        # b_score has highest avg correlation to others: (0.9+0.8)/2=0.85
        assert suggestions["cluster_1"]["drop"] == "b_score"
        assert "a_score" in suggestions["cluster_1"]["keep"]
        assert "c_score" in suggestions["cluster_1"]["keep"]

    def test_empty_clusters_no_suggestions(self):
        """Test that single-feature clusters produce no suggestions."""
        corr_data = np.array([[1.0, 0.2], [0.2, 1.0]])
        columns = ["a_score", "b_score"]
        corr_matrix = pd.DataFrame(corr_data, index=columns, columns=columns)

        clusters = {
            "cluster_1": ["a_score"],
            "cluster_2": ["b_score"],
        }

        suggestions = suggest_exclusions(clusters, corr_matrix)
        assert suggestions == {}

    def test_multiple_multi_feature_clusters(self):
        """Test with multiple clusters needing suggestions."""
        corr_data = np.array([
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.7],
            [0.1, 0.1, 0.7, 1.0],
        ])
        columns = ["a_score", "b_score", "c_score", "d_score"]
        corr_matrix = pd.DataFrame(corr_data, index=columns, columns=columns)

        clusters = {
            "cluster_1": ["a_score", "b_score"],
            "cluster_2": ["c_score", "d_score"],
        }

        suggestions = suggest_exclusions(clusters, corr_matrix)
        assert len(suggestions) == 2
        # In 2-feature clusters, both have the same avg corr to the other,
        # so either could be picked (argmax picks first in ties)
        assert suggestions["cluster_1"]["drop"] in ["a_score", "b_score"]
        assert suggestions["cluster_2"]["drop"] in ["c_score", "d_score"]


class TestSaveAnalysisOutputs:
    """Tests for save_analysis_outputs."""

    def test_saves_all_files(self, simple_data, tmp_path):
        """Test that all expected output files are created."""
        X, columns = simple_data
        corr = compute_correlation_matrix(X, columns, method="spearman")
        cluster_result = compute_feature_clusters(corr, threshold=0.7)

        paths = save_analysis_outputs(corr, cluster_result, tmp_path)

        assert (tmp_path / "correlation_matrix.csv").exists()
        assert (tmp_path / "feature_clusters.json").exists()
        assert (tmp_path / "dendrogram.png").exists()
        assert (tmp_path / "correlation_heatmap.png").exists()

        # Verify JSON is valid
        with open(tmp_path / "feature_clusters.json") as f:
            clusters = json.load(f)
        assert isinstance(clusters, dict)

        # Verify CSV is valid
        df = pd.read_csv(tmp_path / "correlation_matrix.csv", index_col=0)
        assert df.shape[0] == df.shape[1]
