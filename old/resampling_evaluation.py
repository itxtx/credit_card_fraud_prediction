import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
import credit_card_fraud_utils as ccf

def evaluate_resampling_strategy(X, y, strategy, config, n_splits=5, sampling_ratio='auto'):
    """
    Evaluate a specific resampling strategy using cross-validation.
    
    Parameters:
    -----------
    X : DataFrame
        Feature dataset
    y : Series
        Target variable
    strategy : str
        Resampling strategy name
    config : Config object
        Configuration parameters
    n_splits : int
        Number of cross-validation folds
    sampling_ratio : float or 'auto'
        Ratio for resampling
        
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    # Configure resampling
    config_copy = config
    config_copy.USE_RESAMPLING = True if strategy != 'none' else False
    config_copy.RESAMPLING_STRATEGY = strategy
    config_copy.SAMPLING_RATIO = sampling_ratio
    
    # Lists to store performance metrics
    cv_scores_if_auprc = []
    cv_scores_lof_auprc = []
    cv_scores_if_roc = []
    cv_scores_lof_roc = []
    
    # Implement k-fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Track progress
    fold_num = 1
    
    for train_idx, val_idx in kf.split(X, y):
        print(f"  Processing fold {fold_num}/{n_splits}...")
        
        # Get train/validation split
        X_train_cv = X.iloc[train_idx].copy()
        X_val_cv = X.iloc[val_idx].copy()
        y_train_cv = y.iloc[train_idx].copy()
        y_val_cv = y.iloc[val_idx].copy()
        
        # Apply resampling to training data only
        if config_copy.USE_RESAMPLING:
            X_train_cv, y_train_cv = ccf.apply_resampling(X_train_cv, y_train_cv, strategy, config_copy)
        
        # Train models
        if_model = ccf.train_isolation_forest(X_train_cv, config_copy)
        if_preds, if_scores = ccf.get_model_predictions(if_model, X_val_cv)
        
        lof_model = ccf.train_lof(X_train_cv, config_copy)
        lof_preds, lof_scores = ccf.get_model_predictions(lof_model, X_val_cv, is_isolation_forest=False)
        
        # Calculate metrics
        # AUPRC for both models
        precision_if, recall_if, _ = precision_recall_curve(y_val_cv, -if_scores)
        auprc_if = auc(recall_if, precision_if)
        
        precision_lof, recall_lof, _ = precision_recall_curve(y_val_cv, lof_scores)
        auprc_lof = auc(recall_lof, precision_lof)
        
        # ROC AUC for both models
        fpr_if, tpr_if, _ = ccf.roc_curve(y_val_cv, -if_scores)
        roc_auc_if = auc(fpr_if, tpr_if)
        
        fpr_lof, tpr_lof, _ = ccf.roc_curve(y_val_cv, lof_scores)
        roc_auc_lof = auc(fpr_lof, tpr_lof)
        
        # Store metrics
        cv_scores_if_auprc.append(auprc_if)
        cv_scores_lof_auprc.append(auprc_lof)
        cv_scores_if_roc.append(roc_auc_if)
        cv_scores_lof_roc.append(roc_auc_lof)
        
        fold_num += 1
    
    # Return average metrics
    return {
        'strategy': strategy,
        'sampling_ratio': sampling_ratio,
        'IF_AUPRC_mean': np.mean(cv_scores_if_auprc),
        'IF_AUPRC_std': np.std(cv_scores_if_auprc),
        'LOF_AUPRC_mean': np.mean(cv_scores_lof_auprc),
        'LOF_AUPRC_std': np.std(cv_scores_lof_auprc),
        'IF_ROC_AUC_mean': np.mean(cv_scores_if_roc),
        'IF_ROC_AUC_std': np.std(cv_scores_if_roc),
        'LOF_ROC_AUC_mean': np.mean(cv_scores_lof_roc),
        'LOF_ROC_AUC_std': np.std(cv_scores_lof_roc)
    }

def evaluate_all_resampling_strategies(X, y, config, sample_size=None):
    """
    Evaluate all resampling strategies.
    
    Parameters:
    -----------
    X : DataFrame
        Feature dataset
    y : Series
        Target variable
    config : Config object
        Configuration parameters
    sample_size : int or None
        If specified, randomly sample this many instances while maintaining class distribution
        If None, use the full dataset
        
    Returns:
    --------
    DataFrame
        Results of all strategies
    """
    # If sample_size is specified, take a stratified sample
    if sample_size is not None:
        print(f"Taking a stratified sample of {sample_size} instances...")
        from sklearn.model_selection import train_test_split
        
        # Calculate class distribution
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        # Ensure we have at least 100 samples of the minority class
        min_minority_samples = 100
        min_majority_samples = min_minority_samples * 5  # Keep 5:1 ratio
        
        # Calculate required sample size based on class distribution
        minority_ratio = class_counts[minority_class] / len(y)
        required_minority = max(min_minority_samples, int(sample_size * minority_ratio))
        required_majority = min(required_minority * 5, int(sample_size * (1 - minority_ratio)))
        
        required_size = required_minority + required_majority
        
        if sample_size < required_size:
            print(f"Warning: sample_size {sample_size} is too small. Using {required_size} instead.")
            sample_size = required_size
        
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=config.RANDOM_STATE
        )
        print(f"Sample class distribution: {dict(sorted(y_sample.value_counts().items()))}")
    else:
        X_sample, y_sample = X, y
    
    # Define strategies to evaluate
    strategies = ['none', 'ros', 'rus', 'smote', 'adasyn', 'nearmiss', 'tomek', 'enn', 'smote-tomek', 'smote-enn']
    
    # Define sampling ratios based on strategy type
    over_sampling_ratios = [0.1, 0.2, 'auto']  # For over-sampling methods
    under_sampling_ratios = [0.001, 0.0005, 0.0001]  # For under-sampling methods (more conservative)
    
    # Store results
    results = []
    
    # Evaluate each strategy
    for strategy in strategies:
        print(f"Evaluating strategy: {strategy}")
        
        if strategy in ['none', 'tomek', 'enn']:  # These don't use sampling_ratio or use default
            result = evaluate_resampling_strategy(X_sample, y_sample, strategy, config)
            results.append(result)
        elif strategy in ['rus', 'nearmiss']:  # Under-sampling methods
            for ratio in under_sampling_ratios:
                print(f"  With sampling ratio: {ratio}")
                result = evaluate_resampling_strategy(X_sample, y_sample, strategy, config, sampling_ratio=ratio)
                results.append(result)
        else:  # Over-sampling methods
            for ratio in over_sampling_ratios:
                print(f"  With sampling ratio: {ratio}")
                result = evaluate_resampling_strategy(X_sample, y_sample, strategy, config, sampling_ratio=ratio)
                results.append(result)
    
    # Convert results to DataFrame for analysis
    return pd.DataFrame(results)

def plot_resampling_results(results_df, metric='AUPRC'):
    """
    Plot results from resampling strategy evaluation.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from evaluate_all_resampling_strategies
    metric : str
        Which metric to plot ('AUPRC' or 'ROC_AUC')
    """
    # Create a more readable dataset for plotting
    if metric == 'AUPRC':
        plot_df = pd.DataFrame({
            'Strategy': [f"{row['strategy']} ({row['sampling_ratio']})" if row['sampling_ratio'] != 'auto' else f"{row['strategy']} (auto)" 
                        for _, row in results_df.iterrows()],
            'Isolation Forest': results_df[f'IF_{metric}_mean'],
            'IF Error': results_df[f'IF_{metric}_std'],
            'LOF': results_df[f'LOF_{metric}_mean'],
            'LOF Error': results_df[f'LOF_{metric}_std']
        })
    else:  # ROC_AUC
        plot_df = pd.DataFrame({
            'Strategy': [f"{row['strategy']} ({row['sampling_ratio']})" if row['sampling_ratio'] != 'auto' else f"{row['strategy']} (auto)" 
                        for _, row in results_df.iterrows()],
            'Isolation Forest': results_df[f'IF_{metric}_mean'],
            'IF Error': results_df[f'IF_{metric}_std'],
            'LOF': results_df[f'LOF_{metric}_mean'],
            'LOF Error': results_df[f'LOF_{metric}_std']
        })
    
    # Sort by Isolation Forest performance (typically the better model)
    plot_df = plot_df.sort_values('Isolation Forest', ascending=False)
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(plot_df))
    width = 0.35
    
    plt.bar(x - width/2, plot_df['Isolation Forest'], width, 
            yerr=plot_df['IF Error'], 
            label='Isolation Forest')
    
    plt.bar(x + width/2, plot_df['LOF'], width, 
            yerr=plot_df['LOF Error'], 
            label='Local Outlier Factor')
    
    plt.ylabel(f'{metric} Score')
    plt.title(f'Performance by Resampling Strategy ({metric})')
    plt.xticks(x, plot_df['Strategy'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    return plt