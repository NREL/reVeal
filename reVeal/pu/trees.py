# PU ExtraTrees - A Random Forest Classifier for PU Learning
# Adapted from https://github.com/jonathanwilton/PUExtraTrees
from .tree import PUExtraTree
from joblib import Parallel, delayed
import scipy
import numpy as np

class PUExtraTrees:
    def __init__(self, n_estimators = 100,
                 risk_estimator = 'nnPU',
                 loss = 'quadratic',
                 max_depth = None,
                 min_samples_leaf = 1,
                 max_features = 'sqrt',
                 max_candidates = 1,
                 n_jobs = 1,
                 random_state = 42):
        """
        An extra-trees binary classifier that can be trained using only positive and unlabeled samples, or positive and negative samples.
        
        Parameters
        ----------
        random_state : int, default=42
            Controls the randomness of the estimator for reproducible results.
        risk_estimator : {"PN", "uPU", "nnPU"}, default='nnPU'
            PU data based risk estimator. Supports supervised (PN) learning, unbiased PU (uPU) learning and nonnegative PU (nnPU) learning.
        loss : {"quadratic", "logistic"}, default='quadratic'
            The function to measure the cost of making an incorrect prediction. Supported loss functions are:
            "quadratic" l(v,y) = (1-vy)^2 and 
            "logistic" l(v,y) = ln(1+exp(-vy)).
        max_depth : int or None, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_leaf samples. 
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node. The default is 1.
        max_features : int or {"sqrt", "all"}, default="sqrt"
            The number of features to consider when looking for the best split. If "sqrt", then max_features = ceil(sqrt(n_features)). If "all", then max_features = n_features. 
        max_candidates : int, default=1
            Number of randomly chosen split points to consider for each candidate feature. 
        n_jobs : int, default=1
            The number of jobs to run in parallel. fit and predict are all parallelized over the trees.
         
        Returns
        -------
        None.

        """
        
        self.n_estimators = n_estimators
        self.risk_estimator = risk_estimator
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_candidates = max_candidates
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.leaf_count = 0
        self.current_max_depth = 0
        self.is_trained = False # indicate if tree empty/trained
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_estimators': self.n_estimators,
            'risk_estimator': self.risk_estimator,
            'loss': self.loss,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'max_candidates': self.max_candidates,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def train_tree(self, tree_idx, P = None, U = None, N = None, pi = None):
        """
        Train a single decision tree.

        Parameters
        ----------
        tree_idx : int
            Index of the tree being trained (for deterministic seeding)
        P : array-like of shape (n_p, n_features), default=None
            Training samples from the positive class. 
        U : array-like of shape (n_u, n_features), default=None
            Unlabelled training samples.
        N : array-like of shape (n_n, n_features), default=None
            Training samples from the negative class if performing supervised (PN) learning.
        pi : float
            Prior probability that an example belongs to the positive class.

        Returns
        -------
        g : ET classifier
            An instance of the single tree RF classifier.

        """
        # Create deterministic seed for this specific tree
        tree_seed = self.random_state + tree_idx if self.random_state is not None else None
        
        g = PUExtraTree(risk_estimator = self.risk_estimator,
                        loss = self.loss,
                        max_depth = self.max_depth, 
                        min_samples_leaf = self.min_samples_leaf, 
                        max_features = self.max_features, 
                        max_candidates = self.max_candidates,
                        random_state = tree_seed)
        g.fit(P = P, U = U, N = N, pi = pi)
        return g
    
    def predict_tree(self, g, X):
        """
        Predict classes for examples in X using the single DT g.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns
        -------
        preds : array of shape (n_samples,)
            The predicted classes.

        """
        return g.predict(X)
    
    def fit(self, P = None, U = None, N = None, pi = None):
        """
        Train the random forest.

        Parameters
        ----------
        pi : float
            Prior probability that an example belongs to the positive class. 
        P : array-like of shape (n_p, n_features), default=None
            Training samples from the positive class.
        U : array-like of shape (n_u, n_features), default=None
            Unlabeled training samples.
        N : array-like of shape (n_n, n_features), default=None
            Training samples from the negative class if performing PN learning.

        Returns
        -------
        self
            Returns instance of self.

        """
        # Use sequential processing for determinism when n_jobs=1
        if self.n_jobs == 1:
            self.gs = []
            for i in range(self.n_estimators):
                tree = self.train_tree(tree_idx=i, P=P, U=U, N=N, pi=pi)
                self.gs.append(tree)
        else:
            # For parallel processing, pass tree indices for deterministic seeding
            self.gs = Parallel(n_jobs = min(self.n_jobs, self.n_estimators), prefer="threads")(
                delayed(self.train_tree)(tree_idx=i, P=P, U=U, N=N, pi=pi) 
                for i in range(self.n_estimators)
            )
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Predict classes for examples in X.
        The predicted class of an input sample is the majority vote by the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns
        -------
        preds : array of shape (n_samples,)
            The predicted classes.

        """
        # Use sequential processing for determinism when n_jobs=1
        if self.n_jobs == 1:
            preds = []
            for g in self.gs:
                preds.append(self.predict_tree(g, X))
            self.preds = preds
        else:
            self.preds = Parallel(n_jobs = min(self.n_jobs, self.n_estimators), prefer="threads")(
                delayed(self.predict_tree)(g, X) for g in self.gs
            )
        
        return scipy.stats.mode(np.array(self.preds), axis = 0, keepdims = False)[0]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for examples in X.
        The predicted class probabilities of an input sample is the mean predicted class probabilities by the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns
        -------
        proba : array of shape (n_samples, 2)
            The class probabilities of the input samples. proba[i, j] is the probability that sample i belongs to class j.

        """
        # Keep original logic - using sequential processing for determinism when n_jobs=1
        if self.n_jobs == 1:
            preds = []
            for g in self.gs:
                preds.append(self.predict_tree(g, X))
            self.preds = preds
        else:
            self.preds = Parallel(n_jobs = min(self.n_jobs, self.n_estimators), prefer="threads")(
                delayed(self.predict_tree)(g, X) for g in self.gs
            )
        
        preds_array = np.array(self.preds)
        preds_array[preds_array < 1] = 0
        return np.mean(preds_array, axis = 0)

    def n_leaves(self, tree):
        """
        Get the number of leaf nodes in a specified tree

        Parameters
        ----------
        tree : int
            The index of the tree.

        Returns
        -------
        Number of leaf nodes in the specified tree.

        """
        
        return self.gs[tree].n_leaves()
    
    def get_depth(self, tree):
        """
        Get the depth of a specified tree in the forest.

        Parameters
        ----------
        tree : int
            The index of the tree.

        Returns
        -------
        Depth of the specified tree.

        """
        
        return self.gs[tree].get_depth()
    
    def get_max_depth(self):
        """
        Return the depth of the deepest tree in the forest.

        Returns
        -------
        Maximum depth : int

        """
        
        depths = []
        for tree in self.gs:
            depths += [tree.get_depth()]
        return np.max(depths)
    
    def feature_importances(self):
        """
        Get the risk reduction feature importances.

        Returns
        -------
        importances : array of shape (n_features,)
            The risk reduction feature importances.

        """
        importances = np.zeros([self.gs[0].d])
        for tree in self.gs:
            importances += tree.feature_importances()/self.n_estimators
        
        return importances

    def extract_cutpoints(self, feature_idx):
        """
        Extract all cutpoints (split thresholds) for a specific feature across all trees in the forest.
        
        Parameters
        ----------
        feature_idx : int
            Index of the feature to extract cutpoints for.
        
        Returns
        -------
        cutpoints : np.ndarray
            Sorted array of unique cutpoints for the specified feature.
        """
        cutpoints = []
        
        # Iterate through all trees in the forest
        for tree in self.gs:
            # Iterate through all nodes in the tree
            for node_key, node_data in tree.nodes.items():
                # Check if this node splits on the feature of interest
                if node_data['j'] == feature_idx and node_data['xi'] is not None:
                    cutpoints.append(node_data['xi'])
        
        # Return sorted unique cutpoints
        if len(cutpoints) == 0:
            return np.array([])
        
        return np.unique(cutpoints)


    def create_grid(self, X, conditioning_features):
        """
        Create a grid that partitions the feature space based on cutpoints from the trees.
        
        The grid divides each conditioning feature's range into intervals defined by
        the split points (cutpoints) that the trees use for that feature.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to create grid for (typically test set).
        conditioning_features : list of int
            Indices of features to condition on (the Z variables).
        
        Returns
        -------
        grid : dict
            Dictionary mapping feature indices to their grid boundaries.
            grid[feature_idx] = array of boundaries defining intervals
        """
        if not self.is_trained:
            raise ValueError("Forest must be trained before creating grid")
        
        grid = {}
        
        for feature_idx in conditioning_features:
            # Extract cutpoints from trees
            cutpoints = self.extract_cutpoints(feature_idx)
            
            # Get min and max values from the data
            feature_min = X[:, feature_idx].min()
            feature_max = X[:, feature_idx].max()
            
            # Create grid boundaries: [min, cutpoint1, cutpoint2, ..., max]
            # Only include cutpoints that fall within the data range
            valid_cutpoints = cutpoints[(cutpoints > feature_min) & (cutpoints < feature_max)]
            
            # Combine min, cutpoints, and max to form complete boundaries
            boundaries = np.concatenate([[feature_min], valid_cutpoints, [feature_max]])
            boundaries = np.unique(boundaries)  # Ensure uniqueness and sort
            
            grid[feature_idx] = boundaries
        
        return grid


    def assign_to_grid_cells(self, X, grid):
        """
        Assign each sample in X to a grid cell based on the grid boundaries.
        
        For each conditioning feature, determines which interval the sample falls into.
        The combination of intervals across all conditioning features defines the grid cell.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to assign to grid cells.
        grid : dict
            Grid boundaries from create_grid().
        
        Returns
        -------
        grid_cells : np.ndarray of shape (n_samples, n_conditioning_features)
            Grid cell indices for each sample. grid_cells[i, j] indicates which
            interval sample i falls into for conditioning feature j.
        """
        n_samples = X.shape[0]
        n_conditioning = len(grid)
        
        # Initialize array to store grid cell assignments
        grid_cells = np.zeros((n_samples, n_conditioning), dtype=np.int32)
        
        # For each conditioning feature
        for idx, (feature_idx, boundaries) in enumerate(grid.items()):
            # Use searchsorted to find which interval each sample falls into
            # searchsorted returns the index where value would be inserted
            # This gives us the interval: [boundaries[i-1], boundaries[i]]
            feature_values = X[:, feature_idx]
            cell_indices = np.searchsorted(boundaries, feature_values, side='right') - 1
            
            # Ensure indices are within bounds (handles edge cases)
            cell_indices = np.clip(cell_indices, 0, len(boundaries) - 2)
            
            grid_cells[:, idx] = cell_indices
        
        return grid_cells


    def get_samples_in_cell(self, grid_cells, target_cell):
        """
        Get indices of samples that belong to a specific grid cell.
        
        Parameters
        ----------
        grid_cells : np.ndarray of shape (n_samples, n_conditioning_features)
            Grid cell assignments from assign_to_grid_cells().
        target_cell : tuple or np.ndarray
            The grid cell to find samples for. Length must match n_conditioning_features.
        
        Returns
        -------
        sample_indices : np.ndarray
            Indices of samples in the target grid cell.
        """
        # Convert target_cell to array if it's a tuple
        target_cell = np.array(target_cell)
        
        # Find samples where all conditioning features match the target cell
        matches = np.all(grid_cells == target_cell, axis=1)
        
        return np.where(matches)[0]
