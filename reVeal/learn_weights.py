"""
learn_weights module - Core logic for PU learning-based weight derivation.

Trains a PUExtraTrees model on a normalized grid using point labels as positive
samples and auto-sampled background cells as unlabeled samples. Produces a
score-weighted configuration with feature importances normalized as weights.
"""
import logging

import geopandas as gpd
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from reVeal.pu import PUExtraTrees

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_pu_data(
    grid_df,
    labels_gdf,
    background_samples=10000,
    test_size=0.2,
    validation_size=0.1,
    random_state=42,
):
    """
    Prepare positive and unlabeled data splits from a grid and point labels.

    Performs a spatial join of labels onto the grid to identify positive cells,
    then samples background (unlabeled) cells from non-intersecting cells.

    Parameters
    ----------
    grid_df : gpd.GeoDataFrame
        Normalized grid with score attributes and a 'gid' column.
    labels_gdf : gpd.GeoDataFrame
        Point geometries representing positive label locations.
    background_samples : int
        Number of background (unlabeled) cells to sample.
    test_size : float
        Fraction of positive cells to hold out for testing.
    validation_size : float
        Fraction of remaining positive cells for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys: 'train_gids', 'validation_gids', 'test_gids',
        'background_gids', 'background_test_gids', 'background_val_gids'.
    """
    if "gid" not in grid_df.columns:
        grid_df = grid_df.copy()
        grid_df["gid"] = grid_df.index

    joined = gpd.sjoin(grid_df, labels_gdf, how="left", predicate="intersects")
    intersecting_mask = ~joined["index_right"].isna()
    all_positive_gids = joined[intersecting_mask]["gid"].unique()
    non_intersecting_gids = grid_df["gid"][
        ~grid_df["gid"].isin(all_positive_gids)
    ].values

    logger.info(
        f"Found {len(all_positive_gids)} positive cells, "
        f"{len(non_intersecting_gids)} non-intersecting cells."
    )

    rng = np.random.default_rng(random_state)

    # Split positives into train / test
    shuffled = rng.permutation(all_positive_gids)
    test_count = int(len(shuffled) * test_size)
    test_gids = shuffled[:test_count]
    remaining = shuffled[test_count:]

    # Split remaining positives into train / validation
    val_count = int(len(remaining) * validation_size)
    validation_gids = remaining[:val_count]
    train_gids = remaining[val_count:]

    # Sample background
    remaining_bg = non_intersecting_gids.copy()
    rng.shuffle(remaining_bg)

    bg_total = min(background_samples, len(remaining_bg))
    background_gids = remaining_bg[:bg_total]
    remaining_bg = remaining_bg[bg_total:]

    # Background for test and validation (equal to positive counts)
    bg_test_count = min(len(test_gids), len(remaining_bg))
    background_test_gids = remaining_bg[:bg_test_count]
    remaining_bg = remaining_bg[bg_test_count:]

    bg_val_count = min(len(validation_gids), len(remaining_bg))
    background_val_gids = remaining_bg[:bg_val_count]

    logger.info(
        f"Splits: train={len(train_gids)}, val={len(validation_gids)}, "
        f"test={len(test_gids)}, bg={len(background_gids)}, "
        f"bg_test={len(background_test_gids)}, bg_val={len(background_val_gids)}"
    )

    return {
        "train_gids": train_gids,
        "validation_gids": validation_gids,
        "test_gids": test_gids,
        "background_gids": background_gids,
        "background_test_gids": background_test_gids,
        "background_val_gids": background_val_gids,
    }


def train_pu_model(grid_df, data_splits, attributes, n_estimators=500,
                   class_prior=None, n_jobs=1, random_state=42):
    """
    Train a PUExtraTrees model and extract feature importances.

    Parameters
    ----------
    grid_df : gpd.GeoDataFrame
        Grid with feature columns.
    data_splits : dict
        Output from prepare_pu_data with gid arrays.
    attributes : list of str
        Feature column names.
    n_estimators : int
        Number of trees in the forest.
    class_prior : float or None
        Prior probability of positive class. If None, auto-computed.
    n_jobs : int
        Number of parallel jobs for training.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys: 'model', 'feature_importances', 'metrics'.
    """
    g = grid_df

    # Build training arrays
    p_tr = g[g["gid"].isin(data_splits["train_gids"])][attributes].to_numpy()
    u_tr = g[g["gid"].isin(data_splits["background_gids"])][attributes].to_numpy()

    if len(p_tr) == 0:
        raise ValueError(
            "No positive training samples found. Ensure that the labels "
            "spatially intersect the grid."
        )
    if len(u_tr) == 0:
        raise ValueError(
            "No background (unlabeled) training samples found. Ensure that "
            "the grid has cells that do not intersect the labels."
        )

    if class_prior is None:
        class_prior = len(p_tr) / (len(p_tr) + len(u_tr))
        logger.info(f"Auto-computed class_prior: {class_prior:.4f}")

    if not (0 < class_prior < 1):
        raise ValueError(
            f"class_prior must be between 0 and 1 (exclusive), got {class_prior}. "
            "This can happen when all grid cells are labeled as positive or "
            "no positive samples are found."
        )

    # Train model
    model = PUExtraTrees(
        n_estimators=n_estimators,
        risk_estimator="nnPU",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    logger.info(f"Training PUExtraTrees with {n_estimators} trees...")
    model.fit(P=p_tr, U=u_tr, pi=class_prior)
    logger.info("Training complete.")

    # Feature importances
    importances = model.feature_importances()

    # Evaluate on test set if available
    metrics = {}
    if (len(data_splits["test_gids"]) > 0
            and len(data_splits["background_test_gids"]) > 0):
        p_test = g[g["gid"].isin(data_splits["test_gids"])][attributes].to_numpy()
        u_test = g[g["gid"].isin(data_splits["background_test_gids"])][
            attributes
        ].to_numpy()
        x_test = np.vstack((p_test, u_test))
        y_test = np.concatenate((
            np.ones(len(p_test), dtype=int),
            np.zeros(len(u_test), dtype=int),
        ))
        y_pred = (model.predict(x_test) == 1).astype(int)
        y_score = model.predict_proba(x_test)
        metrics["auc"] = float(roc_auc_score(y_test, y_score))
        tpr = y_pred[y_test == 1].sum() / y_test.sum()
        metrics["tpr"] = float(tpr)
        logger.info(f"Test metrics: AUC={metrics['auc']:.4f}, TPR={metrics['tpr']:.4f}")

    return {
        "model": model,
        "feature_importances": importances,
        "metrics": metrics,
    }


def _compute_metric(grid_df, data_splits, attributes, class_prior, n_estimators,
                    n_jobs, random_state, metric="auc"):
    """
    Train a model with the given class_prior and return the specified metric
    evaluated on the validation set.

    Parameters
    ----------
    grid_df : gpd.GeoDataFrame
        Grid with feature columns.
    data_splits : dict
        Output from prepare_pu_data.
    attributes : list of str
        Feature column names.
    class_prior : float
        Prior probability of positive class.
    n_estimators : int
        Number of trees.
    n_jobs : int
        Parallel jobs.
    random_state : int
        Random seed.
    metric : str
        Metric to compute: 'auc' or 'tpr'.

    Returns
    -------
    float
        Metric value on validation set.
    """
    g = grid_df
    p_tr = g[g["gid"].isin(data_splits["train_gids"])][attributes].to_numpy()
    u_tr = g[g["gid"].isin(data_splits["background_gids"])][attributes].to_numpy()

    model = PUExtraTrees(
        n_estimators=n_estimators,
        risk_estimator="nnPU",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(P=p_tr, U=u_tr, pi=class_prior)

    # Evaluate on validation set
    p_val = g[g["gid"].isin(data_splits["validation_gids"])][attributes].to_numpy()
    u_val = g[g["gid"].isin(data_splits["background_val_gids"])][attributes].to_numpy()

    if len(p_val) == 0 or len(u_val) == 0:
        return 0.0

    x_val = np.vstack((p_val, u_val))
    y_val = np.concatenate((
        np.ones(len(p_val), dtype=int),
        np.zeros(len(u_val), dtype=int),
    ))
    y_pred = (model.predict(x_val) == 1).astype(int)

    if metric == "auc":
        y_score = model.predict_proba(x_val)
        return float(roc_auc_score(y_val, y_score))
    elif metric == "tpr":
        return float(y_pred[y_val == 1].sum() / y_val.sum())
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'auc' or 'tpr'.")


def tune_class_prior(grid_df, data_splits, attributes, n_estimators=500,
                     n_jobs=1, random_state=42, n_trials=20, metric="auc"):
    """
    Use Optuna to find the optimal class_prior for PUExtraTrees.

    Parameters
    ----------
    grid_df : gpd.GeoDataFrame
        Grid with feature columns.
    data_splits : dict
        Output from prepare_pu_data.
    attributes : list of str
        Feature column names.
    n_estimators : int
        Number of trees per trial.
    n_jobs : int
        Parallel jobs for each model training.
    random_state : int
        Random seed.
    n_trials : int
        Number of Optuna trials.
    metric : str
        Metric to maximize: 'auc' or 'tpr'.

    Returns
    -------
    dict
        Dictionary with 'best_class_prior' and 'best_value'.
    """
    logger.info(
        f"Tuning class_prior with {n_trials} trials, optimizing '{metric}'..."
    )

    def objective(trial):
        class_prior = trial.suggest_float("class_prior", 0.01, 0.99)
        return _compute_metric(
            grid_df=grid_df,
            data_splits=data_splits,
            attributes=attributes,
            class_prior=class_prior,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            metric=metric,
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_prior = study.best_params["class_prior"]
    best_value = study.best_value
    logger.info(
        f"Tuning complete. Best class_prior={best_prior:.4f}, "
        f"best {metric}={best_value:.4f}"
    )

    return {"best_class_prior": best_prior, "best_value": best_value}


def importances_to_weights(feature_importances, attributes):
    """
    Convert raw feature importances to normalized weights (sum=1).

    Features with importance <= 0 are excluded.

    Parameters
    ----------
    feature_importances : np.ndarray
        Raw importances from PUExtraTrees.feature_importances().
    attributes : list of str
        Feature column names (same order as importances).

    Returns
    -------
    list of dict
        List of {"attribute": name, "weight": float} sorted by weight descending.
        Only includes features with positive importance.
    """
    mask = feature_importances > 0
    valid_attrs = [a for a, m in zip(attributes, mask) if m]
    valid_importances = feature_importances[mask]

    total = valid_importances.sum()
    if total == 0:
        raise ValueError(
            "All feature importances are <= 0. Cannot derive weights. "
            "Check that the label set intersects the grid and features vary."
        )

    weights = valid_importances / total

    result = [
        {"attribute": attr, "weight": float(w)}
        for attr, w in zip(valid_attrs, weights)
    ]
    result.sort(key=lambda x: x["weight"], reverse=True)
    return result


def generate_score_weighted_config(weights, grid_path, score_name="suitability_score"):
    """
    Generate a score-weighted configuration dictionary.

    Parameters
    ----------
    weights : list of dict
        Output from importances_to_weights.
    grid_path : str
        Path to the normalized grid (for the output config).
    score_name : str
        Name for the output score column.

    Returns
    -------
    dict
        Configuration dictionary compatible with ScoreWeightedConfig.
    """
    return {
        "grid": str(grid_path),
        "attributes": weights,
        "score_name": score_name,
    }


def run_learn_weights(config):
    """
    Full pipeline: read data, train PU model, output weight config.

    Parameters
    ----------
    config : LearnWeightsConfig
        Validated configuration object.

    Returns
    -------
    dict
        Dictionary with keys: 'config' (score_weighted config dict),
        'metrics' (model evaluation metrics), 'model' (trained PUExtraTrees).
    """
    # Read grid
    logger.info(f"Reading grid from {config.grid}...")
    grid_df = gpd.read_file(config.grid, engine="pyogrio", use_arrow=True)
    grid_df.fillna(0, inplace=True)

    # Read labels
    logger.info(f"Reading labels from {config.labels}...")
    labels_gdf = gpd.read_file(config.labels, engine="pyogrio", use_arrow=True)

    # Ensure matching CRS
    grid_crs = grid_df.crs
    if labels_gdf.crs != grid_crs:
        logger.info(f"Reprojecting labels to grid CRS ({grid_crs})...")
        labels_gdf = labels_gdf.to_crs(grid_crs)

    # Resolve attributes
    attributes = config.attributes
    if attributes is None:
        attributes = [c for c in grid_df.columns if c.endswith("_score")]
        logger.info(f"Auto-detected {len(attributes)} score attributes.")

    if not attributes:
        raise ValueError(
            "No attributes found. Provide 'attributes' in the config or ensure "
            "the grid has columns ending with '_score'."
        )

    # Ensure gid column
    if "gid" not in grid_df.columns:
        grid_df["gid"] = grid_df.index

    # Filter to cells with valid developable area if column exists
    if "developable_area_m2_score" in grid_df.columns:
        grid_df = grid_df[grid_df["developable_area_m2_score"] > 0]
        logger.info(
            f"Filtered to {len(grid_df)} cells with developable_area_m2_score > 0."
        )

    # Prepare data
    data_splits = prepare_pu_data(
        grid_df=grid_df,
        labels_gdf=labels_gdf,
        background_samples=config.background_samples,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    # Tune class_prior if requested
    class_prior = config.class_prior
    tuning_results = None
    if config.tune and class_prior is None:
        tuning_results = tune_class_prior(
            grid_df=grid_df,
            data_splits=data_splits,
            attributes=attributes,
            n_estimators=config.n_estimators,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            n_trials=config.n_trials,
            metric=config.tuning_metric,
        )
        class_prior = tuning_results["best_class_prior"]

    # Train model
    results = train_pu_model(
        grid_df=grid_df,
        data_splits=data_splits,
        attributes=attributes,
        n_estimators=config.n_estimators,
        class_prior=class_prior,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )

    # Convert to weights
    weights = importances_to_weights(results["feature_importances"], attributes)
    logger.info(f"Derived {len(weights)} non-zero weights from {len(attributes)} features.")

    # Generate output config
    score_config = generate_score_weighted_config(
        weights=weights,
        grid_path=str(config.grid),
        score_name=config.score_name,
    )

    return {
        "config": score_config,
        "metrics": results["metrics"],
        "model": results["model"],
        "tuning": tuning_results,
    }
