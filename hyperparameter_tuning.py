import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Custom scorer for AUPRC
def auprc_scorer(y_true, y_score):
    """Calculate Area Under Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

# Wrapper class for Isolation Forest to work with GridSearchCV
class IsolationForestClassifier:
    def __init__(self, contamination=0.01, n_estimators=100, max_samples='auto', random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state
        )
    
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    
    def predict(self, X):
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)
    
    def score_samples(self, X):
        return -self.model.score_samples(X)  # Negative because lower = more anomalous
    
    def decision_function(self, X):
        # For compatibility with GridSearchCV
        return -self.model.score_samples(X)

# Wrapper class for LOF to work with GridSearchCV
class LOFClassifier:
    def __init__(self, n_neighbors=20, contamination=0.01):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True
        )
    
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    
    def predict(self, X):
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)
    
    def score_samples(self, X):
        return -self.model.score_samples(X)
    
    def decision_function(self, X):
        # For compatibility with GridSearchCV
        return -self.model.score_samples(X)

def tune_isolation_forest(X, y, best_resampling_strategy, best_sampling_ratio, config):
    """
    Tune Isolation Forest hyperparameters.
    
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
        
    Returns:
    --------
    tuple
        (Best parameters, Grid search object)
    """
    # Apply resampling first if needed
    if best_resampling_strategy != 'none':
        config_copy = config
        config_copy.USE_RESAMPLING = True
        config_copy.RESAMPLING_STRATEGY = best_resampling_strategy
        config_copy.SAMPLING_RATIO = best_sampling_ratio
        
        X_resampled, y_resampled = ccf.apply_resampling(X, y, best_resampling_strategy, config_copy)
    else:
        X_resampled, y_resampled = X, y
    
    # Define parameter grid
    param_grid = {
        'contamination': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        'n_estimators': [50, 100, 200, 300],
        'max_samples': ['auto', 0.5, 0.7, 1.0]
    }
    
    # Create the custom scorer
    auprc_scorer_fn = make_scorer(auprc_scorer, needs_threshold=True, greater_is_better=True)
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Initialize and fit grid search
    print("Starting Isolation Forest hyperparameter tuning...")
    grid_search = GridSearchCV(
        IsolationForestClassifier(random_state=config.RANDOM_STATE),
        param_grid,
        scoring=auprc_scorer_fn,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_resampled, y_resampled)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best Isolation Forest parameters: {best_params}")
    print(f"Best AUPRC score: {best_score:.4f}")
    
    return best_params, grid_search

def tune_lof(X, y, best_resampling_strategy, best_sampling_ratio, config):
    """
    Tune Local Outlier Factor hyperparameters.
    
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
        
    Returns:
    --------
    tuple
        (Best parameters, Grid search object)
    """
    # Apply resampling first if needed
    if best_resampling_strategy != 'none':
        config_copy = config
        config_copy.USE_RESAMPLING = True
        config_copy.RESAMPLING_STRATEGY = best_resampling_strategy
        config_copy.SAMPLING_RATIO = best_sampling_ratio
        
        X_resampled, y_resampled = ccf.apply_resampling(X, y, best_resampling_strategy, config_copy)
    else:
        X_resampled, y_resampled = X, y
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [5, 10, 20, 30, 50, 100],
        'contamination': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    }
    
    # Create the custom scorer
    auprc_scorer_fn = make_scorer(auprc_scorer, needs_threshold=True, greater_is_better=True)
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Initialize and fit grid search
    print("Starting LOF hyperparameter tuning...")
    grid_search = GridSearchCV(
        LOFClassifier(),
        param_grid,
        scoring=auprc_scorer_fn,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_resampled, y_resampled)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best LOF parameters: {best_params}")
    print(f"Best AUPRC score: {best_score:.4f}")
    
    return best_params, grid_search

def plot_grid_search_results(grid_search, param_name, title):
    """
    Plot grid search results for a specific parameter.
    
    Parameters:
    -----------
    grid_search : GridSearchCV object
        Fitted grid search object
    param_name : str
        Parameter name to analyze (e.g. 'contamination')
    title : str
        Plot title
    """
    # Extract results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Get all unique values for this parameter
    param_values = sorted(results[f'param_{param_name}'].unique())
    
    # Group by parameter value and calculate mean scores
    mean_scores = []
    for value in param_values:
        mask = results[f'param_{param_name}'] == value
        mean_scores.append(results.loc[mask, 'mean_test_score'].mean())
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, mean_scores, 'o-', linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel('Mean AUPRC Score')
    plt.title(title)
    plt.grid(True)
    
    # Find optimal value
    best_idx = np.argmax(mean_scores)
    best_value = param_values[best_idx]
    best_score = mean_scores[best_idx]
    
    plt.scatter([best_value], [best_score], s=200, c='red', marker='*', 
                label=f'Best: {best_value} (Score: {best_score:.4f})')
    plt.legend()
    
    return plt