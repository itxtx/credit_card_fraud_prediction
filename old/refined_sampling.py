import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
import credit_card_fraud_utils as ccf

def evaluate_refined_sampling_ratios(X, y, config, sample_size=10000):
    """
    Evaluate a more refined set of sampling ratios based on initial findings.
    
    Parameters:
    -----------
    X : DataFrame
        Feature dataset
    y : Series
        Target variable
    config : Config object
        Configuration parameters
    sample_size : int
        Number of samples to use for evaluation
        
    Returns:
    --------
    DataFrame
        Results of all strategies with refined sampling ratios
    """
    # Take a stratified sample if specified
    if sample_size is not None and sample_size < len(X):
        from sklearn.model_selection import train_test_split
        
        X_sample, _, y_sample, _ = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=config.RANDOM_STATE
        )
        print(f"Using stratified sample of {sample_size} instances")
        print(f"Sample class distribution: {dict(sorted(y_sample.value_counts().items()))}")
    else:
        X_sample, y_sample = X, y
    
    # Define strategies to evaluate with refined sampling ratios
    evaluation_configs = [
        # For oversampling methods around 0.1
        {'strategy': 'ros', 'ratios': [0.05, 0.075, 0.1, 0.125, 0.15]},
        {'strategy': 'smote', 'ratios': [0.05, 0.075, 0.1, 0.125, 0.15]},
        {'strategy': 'adasyn', 'ratios': [0.05, 0.075, 0.1, 0.125, 0.15]},
        
        # For undersampling methods around 0.001
        {'strategy': 'rus', 'ratios': [0.0008, 0.001, 0.0012, 0.0015]},
        {'strategy': 'nearmiss', 'ratios': [0.0008, 0.001, 0.0012, 0.0015]},
        
        # Combined methods with refined ratios
        {'strategy': 'smote-tomek', 'ratios': [0.075, 0.1, 0.125]},
        {'strategy': 'smote-enn', 'ratios': [0.075, 0.1, 0.125]},
        
        # Baseline (no resampling)
        {'strategy': 'none', 'ratios': [None]}
    ]
    
    # Store results
    results = []
    
    # Evaluate each strategy with its refined set of ratios
    for config_item in evaluation_configs:
        strategy = config_item['strategy']
        ratios = config_item['ratios']
        
        print(f"Evaluating strategy: {strategy}")
        
        for ratio in ratios:
            if ratio is not None:
                print(f"  With sampling ratio: {ratio}")
            
            # Evaluate current strategy with the specified ratio
            result = evaluate_resampling_strategy(X_sample, y_sample, strategy, config, 
                                                 n_splits=3, sampling_ratio=ratio)
            results.append(result)
    
    # Convert results to DataFrame for analysis
    return pd.DataFrame(results)

def evaluate_resampling_strategy(X, y, strategy, config, n_splits=3, sampling_ratio=None):
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
    sampling_ratio : float or None
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

def plot_refined_results(results_df, metric='AUPRC', model='IF'):
    """
    Plot results from refined sampling ratio evaluation with a focus on one model.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from evaluate_refined_sampling_ratios
    metric : str
        Which metric to plot ('AUPRC' or 'ROC_AUC')
    model : str
        Which model to focus on ('IF' for Isolation Forest or 'LOF')
    """
    plt.figure(figsize=(14, 10))
    
    # Group strategies and get unique ones
    unique_strategies = results_df['strategy'].unique()
    
    # Create color palette
    colors = sns.color_palette("husl", len(unique_strategies))
    
    # Plot for each strategy
    for i, strategy in enumerate(unique_strategies):
        strategy_data = results_df[results_df['strategy'] == strategy]
        
        # Skip if only one data point (like 'none')
        if len(strategy_data) <= 1:
            if strategy == 'none':
                # Add horizontal line for baseline
                plt.axhline(y=strategy_data[f'{model}_{metric}_mean'].values[0], 
                          color='black', linestyle='--', 
                          label=f'Baseline (no resampling): {strategy_data[f"{model}_{metric}_mean"].values[0]:.4f}')
            continue
            
        # Sort by sampling ratio
        strategy_data = strategy_data.sort_values('sampling_ratio')
        
        # Extract data for plotting
        x = strategy_data['sampling_ratio']
        y = strategy_data[f'{model}_{metric}_mean']
        yerr = strategy_data[f'{model}_{metric}_std']
        
        # Plot line with error bars
        plt.errorbar(x, y, yerr=yerr, marker='o', label=strategy, color=colors[i])
    
    plt.xlabel('Sampling Ratio')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Performance by Sampling Ratio - {model} {metric}')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value annotations
    for strategy in unique_strategies:
        if strategy == 'none':
            continue
            
        strategy_data = results_df[results_df['strategy'] == strategy].sort_values('sampling_ratio')
        for x, y in zip(strategy_data['sampling_ratio'], strategy_data[f'{model}_{metric}_mean']):
            plt.annotate(f'{y:.4f}', 
                        (x, y), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center')
    
    plt.tight_layout()
    return plt

def find_best_configuration(results_df, metric='AUPRC', model='IF'):
    """
    Find the best resampling configuration based on specified metric and model.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from evaluate_refined_sampling_ratios
    metric : str
        Which metric to use ('AUPRC' or 'ROC_AUC')
    model : str
        Which model to focus on ('IF' for Isolation Forest or 'LOF')
        
    Returns:
    --------
    dict
        Best configuration details
    """
    # Sort by the specified metric for the specified model
    sorted_results = results_df.sort_values(f'{model}_{metric}_mean', ascending=False)
    
    # Get the best configuration
    best_config = sorted_results.iloc[0]
    
    # Create detailed results
    best_details = {
        'strategy': best_config['strategy'],
        'sampling_ratio': best_config['sampling_ratio'],
        f'{model}_{metric}_score': best_config[f'{model}_{metric}_mean'],
        'std_dev': best_config[f'{model}_{metric}_std'],
        'improvement_over_baseline': None  # Will calculate this next
    }
    
    # Find the baseline (no resampling) score
    baseline = results_df[results_df['strategy'] == 'none']
    if not baseline.empty:
        baseline_score = baseline.iloc[0][f'{model}_{metric}_mean']
        best_details['improvement_over_baseline'] = (best_details[f'{model}_{metric}_score'] - baseline_score) / baseline_score * 100
    
    return best_details