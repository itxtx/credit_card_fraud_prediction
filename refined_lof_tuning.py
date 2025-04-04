import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import credit_card_fraud_utils as ccf

# Custom scorer for AUPRC
def auprc_scorer(y_true, y_score):
    """Calculate Area Under Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

# Wrapper class for LOF to work with GridSearchCV (enhanced)
class RefinedLOFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=20, contamination=0.01, algorithm='auto', 
                 leaf_size=30, metric='minkowski', p=2):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.model = None
        self.classes_ = np.array([0, 1])  # Required for compatibility with GridSearchCV
    
    def fit(self, X, y=None):
        """Fit the model according to the given training data."""
        # Check that X and y have correct shape
        if y is not None:
            X, y = check_X_y(X, y)
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            novelty=True  # Required for prediction on new data
        )
        self.model.fit(X)
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        check_is_fitted(self, 'model')
        X = check_array(X)
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)
    
    def score_samples(self, X):
        """Return the anomaly score of the samples."""
        check_is_fitted(self, 'model')
        X = check_array(X)
        return -self.model.score_samples(X)
    
    def decision_function(self, X):
        """Compute the decision function of the samples."""
        check_is_fitted(self, 'model')
        X = check_array(X)
        return -self.model.score_samples(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        check_is_fitted(self, 'model')
        X = check_array(X)
        scores = self.decision_function(X)
        # Convert scores to probabilities using sigmoid
        proba = 1 / (1 + np.exp(-scores))
        # Return probabilities for both classes
        return np.column_stack([1 - proba, proba])
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_neighbors': self.n_neighbors,
            'contamination': self.contamination,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'metric': self.metric,
            'p': self.p
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

def tune_lof_refined(X, y, best_resampling_strategy, best_sampling_ratio, config, sample_size=None):
    """
    Enhanced LOF hyperparameter tuning with refined parameter grid and optimal resampling.
    
    Parameters:
    -----------
    X : DataFrame
        Feature dataset
    y : Series
        Target variable
    best_resampling_strategy : str
        Best resampling strategy from previous evaluation
    best_sampling_ratio : float or 'auto'
        Best sampling ratio from previous evaluation
    config : Config object
        Configuration parameters
    sample_size : int or None
        Size of sample to use for tuning, None uses all data
        
    Returns:
    --------
    tuple
        (Best parameters, Grid search object)
    """
    print(f"Starting refined LOF tuning with {best_resampling_strategy} resampling (ratio: {best_sampling_ratio})")
    
    # Apply resampling first if needed
    if best_resampling_strategy != 'none':
        config_copy = config
        config_copy.USE_RESAMPLING = True
        config_copy.RESAMPLING_STRATEGY = best_resampling_strategy
        config_copy.SAMPLING_RATIO = best_sampling_ratio
        
        X_resampled, y_resampled = ccf.apply_resampling(X, y, best_resampling_strategy, config_copy)
    else:
        X_resampled, y_resampled = X, y
    
    # Sample a smaller subset for cross-validation if needed
    if sample_size is not None and sample_size < len(X_resampled):
        from sklearn.model_selection import train_test_split
        
        X_sampled, _, y_sampled, _ = train_test_split(
            X_resampled, y_resampled,
            train_size=sample_size,
            stratify=y_resampled,
            random_state=config.RANDOM_STATE
        )
        print(f"Using {sample_size} samples for tuning (class distribution: {dict(sorted(pd.Series(y_sampled).value_counts().items()))})")
    else:
        X_sampled, y_sampled = X_resampled, y_resampled
        print(f"Using all {len(X_sampled)} samples for tuning")
    
    # Define refined parameter grid based on domain knowledge for fraud detection
    param_grid = {
        'n_neighbors': [5, 10, 15, 20, 25, 30, 40, 50],  # More values around common optimum
        'contamination': [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05],  # Refined around fraud rate
        'algorithm': ['auto'],  # Usually 'auto' is best, but could include 'ball_tree', 'kd_tree', 'brute'
        'metric': ['minkowski'],  # Could include 'euclidean', 'manhattan' for further tuning
    }
    
    # Create the custom AUPRC scorer
    auprc_scorer_fn = make_scorer(auprc_scorer, needs_threshold=True, greater_is_better=True)
    
    # Set up stratified cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Initialize and fit grid search with JobLib parallel backend for faster computation
    print("Running grid search (this may take a while)...")
    from joblib import parallel_backend
    
    with parallel_backend('threading', n_jobs=-1):
        grid_search = GridSearchCV(
            RefinedLOFClassifier(),
            param_grid,
            scoring=auprc_scorer_fn,
            cv=cv,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_sampled, y_sampled)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best LOF parameters: {best_params}")
    print(f"Best AUPRC score: {best_score:.4f}")
    
    return best_params, grid_search

def plot_grid_search_results_enhanced(grid_search, param_name, title, top_n=3):
    """
    Enhanced plot of grid search results for a specific parameter.
    
    Parameters:
    -----------
    grid_search : GridSearchCV object
        Fitted grid search object
    param_name : str
        Parameter name to analyze (e.g. 'n_neighbors')
    title : str
        Plot title
    top_n : int
        Number of top parameter combinations to highlight
    """
    # Extract results into a DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Get all unique values for this parameter
    param_values = sorted(results[f'param_{param_name}'].unique())
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Calculate mean scores for each parameter value
    mean_scores = []
    std_scores = []
    for value in param_values:
        mask = results[f'param_{param_name}'] == value
        mean_scores.append(results.loc[mask, 'mean_test_score'].mean())
        std_scores.append(results.loc[mask, 'mean_test_score'].std())
    
    # Plot line with confidence interval
    ax1.plot(param_values, mean_scores, 'o-', linewidth=2, color='#2C7BB6')
    ax1.fill_between(param_values, 
                    [m - s for m, s in zip(mean_scores, std_scores)],
                    [m + s for m, s in zip(mean_scores, std_scores)],
                    alpha=0.2, color='#2C7BB6')
    
    # Find and mark the best value
    best_idx = np.argmax(mean_scores)
    best_value = param_values[best_idx]
    best_score = mean_scores[best_idx]
    
    ax1.scatter([best_value], [best_score], s=200, c='#D7191C', marker='*', 
               label=f'Best: {best_value} (Score: {best_score:.4f})')
    
    # Highlight top performing configurations
    top_configs = results.sort_values('mean_test_score', ascending=False).head(top_n)
    
    # Create summary table for top configurations
    top_params_list = []
    for _, row in top_configs.iterrows():
        param_dict = {}
        for p in grid_search.param_grid.keys():
            param_dict[p] = row[f'param_{p}']
        param_dict['score'] = row['mean_test_score']
        top_params_list.append(param_dict)
    
    top_params_df = pd.DataFrame(top_params_list)
    
    # Style the top parameters table
    cell_text = []
    for _, row in top_params_df.iterrows():
        cell_text.append([row[p] if p != 'score' else f"{row[p]:.4f}" for p in top_params_df.columns])
    
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=cell_text, colLabels=top_params_df.columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title and labels
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Mean AUPRC Score')
    ax1.set_title(title)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Add annotations for each point
    for x, y in zip(param_values, mean_scores):
        ax1.annotate(f'{y:.4f}', 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    return plt

def analyze_lof_hyperparameters(grid_search):
    """
    Perform a comprehensive analysis of LOF hyperparameter interactions.
    
    Parameters:
    -----------
    grid_search : GridSearchCV object
        Fitted grid search object
        
    Returns:
    --------
    plt : matplotlib plot
        Interaction plot showing how different parameters affect each other
    """
    # Extract results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Check which parameters were tuned
    param_names = [name for name in grid_search.param_grid.keys() 
                  if len(grid_search.param_grid[name]) > 1]
    
    # We need at least 2 parameters for an interaction plot
    if len(param_names) < 2:
        print("Need at least 2 tuned parameters for interaction analysis")
        return None
    
    # Choose the top 2 most important parameters
    # We'll use n_neighbors and contamination if available
    if 'n_neighbors' in param_names and 'contamination' in param_names:
        param1 = 'n_neighbors'
        param2 = 'contamination'
    else:
        param1 = param_names[0]
        param2 = param_names[1]
    
    # Create interaction plot
    plt.figure(figsize=(12, 8))
    
    # Get unique values for each parameter
    param1_values = sorted(results[f'param_{param1}'].unique())
    param2_values = sorted(results[f'param_{param2}'].unique())
    
    # Create a 2D grid of scores
    score_grid = np.zeros((len(param1_values), len(param2_values)))
    
    # Fill in the grid
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            mask = (results[f'param_{param1}'] == val1) & (results[f'param_{param2}'] == val2)
            if any(mask):
                score_grid[i, j] = results.loc[mask, 'mean_test_score'].mean()
            else:
                score_grid[i, j] = np.nan
    
    # Create heatmap
    sns.heatmap(score_grid, annot=True, fmt='.4f', cmap='viridis',
               xticklabels=param2_values, yticklabels=param1_values)
    
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(f'Interaction between {param1} and {param2} on AUPRC Score')
    
    # Find optimal combination
    max_i, max_j = np.unravel_index(np.nanargmax(score_grid), score_grid.shape)
    best_param1 = param1_values[max_i]
    best_param2 = param2_values[max_j]
    best_score = score_grid[max_i, max_j]
    
    plt.tight_layout()
    
    print(f"Optimal combination: {param1}={best_param1}, {param2}={best_param2}, Score={best_score:.4f}")
    
    return plt