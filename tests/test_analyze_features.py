# -*- coding: utf-8 -*-
"""
Tests for analyze_features CLI and config.
"""
import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pydantic import ValidationError

from reVeal.config.analyze_features import AnalyzeFeaturesConfig


@pytest.fixture
def synthetic_grid(tmp_path):
    """Create a synthetic grid with score columns saved as GPKG."""
    rng = np.random.default_rng(42)
    n_cells = 200

    geoms = [box(i % 20, i // 20, (i % 20) + 1, (i // 20) + 1) for i in range(n_cells)]

    data = {
        "gid": list(range(n_cells)),
        "feature_a_score": rng.uniform(0, 1, n_cells),
        "feature_b_score": rng.uniform(0, 1, n_cells),
        "feature_c_score": rng.uniform(0, 1, n_cells),
        "geometry": geoms,
    }

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    grid_path = tmp_path / "grid.gpkg"
    gdf.to_file(grid_path, driver="GPKG")
    return grid_path


class TestAnalyzeFeaturesConfig:
    """Tests for AnalyzeFeaturesConfig validation."""

    def test_valid_config(self, synthetic_grid):
        """Test creating a valid config."""
        config = AnalyzeFeaturesConfig(grid=synthetic_grid)
        assert config.correlation_method == "spearman"
        assert config.cluster_threshold == 0.7

    def test_pearson_method(self, synthetic_grid):
        """Test pearson correlation method is accepted."""
        config = AnalyzeFeaturesConfig(
            grid=synthetic_grid, correlation_method="pearson"
        )
        assert config.correlation_method == "pearson"

    def test_custom_threshold(self, synthetic_grid):
        """Test custom cluster threshold."""
        config = AnalyzeFeaturesConfig(
            grid=synthetic_grid, cluster_threshold=1.5
        )
        assert config.cluster_threshold == 1.5

    def test_invalid_threshold_zero(self, synthetic_grid):
        """Test that threshold=0 is rejected."""
        with pytest.raises(ValidationError):
            AnalyzeFeaturesConfig(grid=synthetic_grid, cluster_threshold=0)

    def test_invalid_threshold_over_2(self, synthetic_grid):
        """Test that threshold > 2 is rejected."""
        with pytest.raises(ValidationError):
            AnalyzeFeaturesConfig(grid=synthetic_grid, cluster_threshold=2.5)

    def test_mutual_exclusion_attributes(self, synthetic_grid):
        """Test that specifying both attribute options raises error."""
        with pytest.raises(ValidationError, match="Only one of"):
            AnalyzeFeaturesConfig(
                grid=synthetic_grid,
                attributes=["a_score"],
                exclude_attributes=["b_score"],
            )


class TestAnalyzeFeaturesRun:
    """Tests for the analyze-features run function."""

    def test_full_analysis_run(self, synthetic_grid, tmp_path):
        """Test that run() produces expected output files."""
        from reVeal.cli.analyze_features import run

        out_dir = tmp_path / "output"
        run(
            grid=str(synthetic_grid),
            out_dir=str(out_dir),
            correlation_method="spearman",
            cluster_threshold=0.7,
            _local=False,
        )

        analysis_dir = out_dir / "analysis"
        assert analysis_dir.exists()
        assert (analysis_dir / "correlation_matrix.csv").exists()
        assert (analysis_dir / "feature_clusters.json").exists()
        assert (analysis_dir / "dendrogram.png").exists()
        assert (analysis_dir / "correlation_heatmap.png").exists()
        assert (analysis_dir / "suggested_exclusions.json").exists()

    def test_explicit_attributes(self, synthetic_grid, tmp_path):
        """Test that attributes filters features."""
        from reVeal.cli.analyze_features import run

        out_dir = tmp_path / "output"
        run(
            grid=str(synthetic_grid),
            out_dir=str(out_dir),
            attributes=["feature_a_score", "feature_b_score"],
            _local=False,
        )

        analysis_dir = out_dir / "analysis"
        # Correlation matrix should only have 2 features
        import pandas as pd
        corr = pd.read_csv(analysis_dir / "correlation_matrix.csv", index_col=0)
        assert len(corr.columns) == 2

    def test_exclude_attributes(self, synthetic_grid, tmp_path):
        """Test that exclude_attributes filters features."""
        from reVeal.cli.analyze_features import run

        out_dir = tmp_path / "output"
        run(
            grid=str(synthetic_grid),
            out_dir=str(out_dir),
            exclude_attributes=["feature_c_score"],
            _local=False,
        )

        analysis_dir = out_dir / "analysis"
        import pandas as pd
        corr = pd.read_csv(analysis_dir / "correlation_matrix.csv", index_col=0)
        assert "feature_c_score" not in corr.columns

    def test_pearson_method(self, synthetic_grid, tmp_path):
        """Test pearson correlation method in run."""
        from reVeal.cli.analyze_features import run

        out_dir = tmp_path / "output"
        run(
            grid=str(synthetic_grid),
            out_dir=str(out_dir),
            correlation_method="pearson",
            _local=False,
        )

        assert (out_dir / "analysis" / "correlation_matrix.csv").exists()

    def test_no_score_columns_raises(self, tmp_path):
        """Test that a grid without score columns raises ValueError."""
        from reVeal.cli.analyze_features import run

        # Create a grid with no _score columns
        geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
        gdf = gpd.GeoDataFrame(
            {"gid": [0, 1], "other_col": [1.0, 2.0], "geometry": geoms},
            crs="EPSG:4326",
        )
        grid_path = tmp_path / "no_scores.gpkg"
        gdf.to_file(grid_path, driver="GPKG")

        out_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="No attributes found"):
            run(grid=str(grid_path), out_dir=str(out_dir), _local=False)

    def test_suggested_exclusions_content(self, synthetic_grid, tmp_path):
        """Test that suggested_exclusions.json has valid structure."""
        from reVeal.cli.analyze_features import run

        out_dir = tmp_path / "output"
        run(
            grid=str(synthetic_grid),
            out_dir=str(out_dir),
            _local=False,
        )

        exclusions_path = out_dir / "analysis" / "suggested_exclusions.json"
        with open(exclusions_path) as f:
            data = json.load(f)

        # Should be a dict (possibly empty if no multi-feature clusters)
        assert isinstance(data, dict)
        for cluster_id, suggestion in data.items():
            assert "drop" in suggestion
            assert "keep" in suggestion
            assert isinstance(suggestion["keep"], list)
