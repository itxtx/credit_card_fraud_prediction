import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_curve, 
                             auc, roc_curve, average_precision_score) # Added average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Import resampling modules
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imblearn not installed. Resampling techniques will not be available. Install with: pip install imbalanced-learn")

# Configuration parameters
class Config:
    # File paths
    DATA_PATH = 'creditcard.csv'  # You need to download this from Kaggle
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Isolation Forest parameters
    IF_CONTAMINATION = 0.01  # Initial estimate, can be updated based on data or tuned
    IF_N_ESTIMATORS = 100
    IF_MAX_SAMPLES = 'auto'
    
    # LOF parameters
    LOF_N_NEIGHBORS = 20
    LOF_CONTAMINATION = 0.01  # Initial estimate, can be updated or tuned
    LOF_NOVELTY = True # Set to True to use predict/score_samples on new data
    
    # Resampling parameters
    USE_RESAMPLING = False
    RESAMPLING_STRATEGY = 'none'  # Options: 'none', 'ros', 'rus', 'smote', 'adasyn', 'nearmiss', 'tomek', 'enn', 'smote-tomek', 'smote-enn'
    # For oversamplers: ratio of minority to majority AFTER resampling.
    # For undersamplers: ratio of minority to majority AFTER resampling (applied to majority class).
    # 'auto' typically means balancing the classes.
    SAMPLING_RATIO = 0.1  
    
    # Visualization
    FIGSIZE = (12, 8)
    SUBPLOT_FIGSIZE = (12,12) # For plots with subplots
    HEATMAP_FIGSIZE = (15,12) # For correlation heatmap

def load_data(config):
    """Load and return the credit card fraud dataset."""
    try:
        df = pd.read_csv(config.DATA_PATH)
        return df
    except FileNotFoundError:
        print(f"Error: The data file was not found at {config.DATA_PATH}")
        print("Please ensure 'creditcard.csv' is in the correct directory or update config.DATA_PATH.")
        return None

def get_data_summary(df):
    """Get a summary of the dataset."""
    if df is None:
        return None
    
    # Check class distribution
    if 1 not in df['Class'].value_counts() or 0 not in df['Class'].value_counts():
        print("Warning: One or both classes (0 or 1) are missing in the 'Class' column.")
        fraud_count = df['Class'].value_counts().get(1, 0)
        normal_count = df['Class'].value_counts().get(0, 0)
    else:
        fraud_count = df['Class'].value_counts()[1]
        normal_count = df['Class'].value_counts()[0]

    if df.shape[0] > 0 :
        fraud_percentage = (fraud_count / df.shape[0]) * 100
    else:
        fraud_percentage = 0
    
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'fraud_count': fraud_count,
        'normal_count': normal_count,
        'fraud_percentage': fraud_percentage
    }
    
    return summary

def plot_class_distribution(df, config):
    """Plot the class distribution."""
    if df is None:
        return None
    plt.figure(figsize=config.FIGSIZE)
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    return plt

def plot_amount_distribution(df, config):
    """Plot the transaction amount distribution by class."""
    if df is None:
        return None
    # Removed redundant plt.figure() call here
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.SUBPLOT_FIGSIZE)
    
    # Plot amount distribution by class
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax1)
    ax1.set_title('Transaction Amount Distribution by Class')
    ax1.set_yscale('log') # Use log scale for better visualization of outliers
    ax1.set_xticklabels(['Normal', 'Fraud'])
    
    # Compare normal vs fraudulent transaction amounts
    amount_fraud = df[df['Class'] == 1]['Amount']
    amount_normal = df[df['Class'] == 0]['Amount']
    
    sns.histplot(amount_fraud, color='red', label='Fraud', ax=ax2, kde=True, bins=50, stat="density")
    sns.histplot(amount_normal, color='blue', label='Normal', ax=ax2, kde=True, bins=50, alpha=0.6, stat="density")
    ax2.set_title('Transaction Amount Distribution: Normal vs Fraud')
    ax2.set_xscale('log') # Use log scale for better visualization
    ax2.legend()
    
    plt.tight_layout()
    return plt

def plot_correlation_matrix(df, config):
    """Plot the correlation matrix of features."""
    if df is None:
        return None
    plt.figure(figsize=config.HEATMAP_FIGSIZE)
    # Exclude 'Time' if it's not relevant or scale it first for meaningful correlation
    correlation_matrix = df.drop(columns=['Time'], errors='ignore').corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title('Feature Correlation Matrix (excluding Time)')
    plt.tight_layout()
    return plt, correlation_matrix

def get_top_correlations(correlation_matrix, feature='Class', top_n=10):
    """Get top correlated features with the target feature."""
    if correlation_matrix is None or feature not in correlation_matrix:
        return None
    correlations = correlation_matrix[feature].sort_values(ascending=False)
    # Exclude self-correlation (which is 1) and take top_n
    return correlations.drop(feature, errors='ignore').head(top_n)


def apply_resampling(X, y, strategy, config):
    """Apply the specified resampling strategy to the data."""
    if not IMBLEARN_AVAILABLE:
        print("Warning: imblearn not installed. Skipping resampling.")
        return X, y
    
    if strategy == 'none' or strategy is None:
        return X, y
    
    print(f"Applying {strategy} resampling strategy...")
    print(f"Original class distribution: {dict(sorted(y.value_counts().items()))}")

    minority_class_label = 1 
    majority_class_label = 0
    
    original_minority_count = y.value_counts().get(minority_class_label, 0)
    original_majority_count = y.value_counts().get(majority_class_label, 0)

    resampler = None
    current_sampling_strategy = config.SAMPLING_RATIO

    if strategy in ['ros', 'smote', 'adasyn', 'smote-tomek', 'smote-enn']: # Oversampling or combined
        if isinstance(current_sampling_strategy, float):
            # Target number of minority samples = majority_count * ratio
            # imblearn expects ratio of minority to majority AFTER resampling
            # So if ratio is 0.5, minority will be 50% of majority.
            # If original minority is less than target, it will oversample.
            pass # Float ratio is directly usable
        elif current_sampling_strategy == 'auto':
            current_sampling_strategy = 'minority' # Upsample the minority class
        else: # Dictionary
            pass


    elif strategy in ['rus', 'nearmiss']: # Undersampling
        if isinstance(current_sampling_strategy, float):
            # Target number of majority samples = minority_count / ratio
            # imblearn expects ratio of minority to majority AFTER resampling.
            # If ratio is 0.5, majority will be 2 * minority.
            pass # Float ratio is directly usable
        elif current_sampling_strategy == 'auto':
             current_sampling_strategy = 'majority' # Downsample the majority class
        else: # Dictionary
            pass


    elif strategy in ['tomek', 'enn']: # Cleaning
        current_sampling_strategy = 'all' # Clean both classes, or 'majority'

    # Initialize the appropriate resampler
    if strategy == 'ros':
        resampler = RandomOverSampler(sampling_strategy=current_sampling_strategy, random_state=config.RANDOM_STATE)
    elif strategy == 'rus':
        resampler = RandomUnderSampler(sampling_strategy=current_sampling_strategy, random_state=config.RANDOM_STATE)
    elif strategy == 'smote':
        resampler = SMOTE(sampling_strategy=current_sampling_strategy, random_state=config.RANDOM_STATE, k_neighbors=5) # k_neighbors might need adjustment
    elif strategy == 'adasyn':
        resampler = ADASYN(sampling_strategy=current_sampling_strategy, random_state=config.RANDOM_STATE, n_neighbors=5) # n_neighbors might need adjustment
    elif strategy == 'nearmiss':
        resampler = NearMiss(sampling_strategy=current_sampling_strategy, version=1) # version can be 1, 2, or 3
    elif strategy == 'tomek':
        resampler = TomekLinks(sampling_strategy=current_sampling_strategy) # 'all' or 'majority'
    elif strategy == 'enn':
        resampler = EditedNearestNeighbours(sampling_strategy=current_sampling_strategy, kind_sel='all', n_neighbors=3) # kind_sel 'all' or 'mode'
    elif strategy == 'smote-tomek':
        resampler = SMOTETomek(sampling_strategy=current_sampling_strategy, random_state=config.RANDOM_STATE, smote=SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5))
    elif strategy == 'smote-enn':
        resampler = SMOTEENN(sampling_strategy=current_sampling_strategy, random_state=config.RANDOM_STATE, smote=SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5))
    else:
        print(f"Unknown resampling strategy: {strategy}. Using original data.")
        return X, y
    
    try:
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        print(f"Class distribution after resampling: {dict(sorted(pd.Series(y_resampled).value_counts().items()))}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error during resampling with {strategy} and sampling_strategy '{current_sampling_strategy}': {e}")
        print("Using original data.")
        return X, y


def preprocess_data(df, config):
    """Preprocess the data and return train/test splits."""
    if df is None:
        return None, None, None, None
        
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    features_to_scale = ['Time', 'Amount']
    
    # Split data into training and testing sets FIRST to prevent data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y  # Ensure both sets have a similar proportion of fraud cases
    )
    
    # Create copies to avoid SettingWithCopyWarning when scaling
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    # Initialize and apply scaling
    scaler = StandardScaler()
    
    # Fit scaler ONLY on training data and transform both train and test
    X_train[features_to_scale] = scaler.fit_transform(X_train_raw[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test_raw[features_to_scale])
    
    # Apply resampling to the training data if configured
    if config.USE_RESAMPLING and config.RESAMPLING_STRATEGY not in ['none', None]:
        X_train, y_train = apply_resampling(X_train, y_train, config.RESAMPLING_STRATEGY, config)
    
    return X_train, X_test, y_train, y_test

def train_isolation_forest(X_train, config):
    """Train and return an Isolation Forest model."""
    if X_train is None: return None
    model = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        max_samples=config.IF_MAX_SAMPLES,
        contamination=config.IF_CONTAMINATION, # This is an estimate of the proportion of outliers
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train) # Unsupervised, so y_train is not used for fitting
    return model

def train_lof(X_train, config):
    """Train and return a Local Outlier Factor model."""
    if X_train is None: return None
    model = LocalOutlierFactor(
        n_neighbors=config.LOF_N_NEIGHBORS,
        contamination=config.LOF_CONTAMINATION, # Proportion of outliers
        novelty=config.LOF_NOVELTY # Must be True to use predict, decision_function and score_samples on new data
    )
    model.fit(X_train) # Unsupervised
    return model

def get_model_predictions(model, X, is_isolation_forest=True):
    """Get predictions and anomaly scores from a model."""
    if model is None or X is None: return None, None

    predictions = model.predict(X) # For IF and LOF (novelty=True): -1 for outliers, 1 for inliers.
    
    # score_samples returns the opposite of the anomaly score. Higher value = more normal.
    # We want scores where higher = more anomalous for consistent evaluation.
    if is_isolation_forest:
        # For IF, score_samples() returns higher for inliers. Negative makes lower for inliers (more anomalous).
        # To make higher = more anomalous, we use -score_samples()
        scores = -model.score_samples(X) 
    else: # LOF with novelty=True
        # For LOF (novelty=True), score_samples() returns higher for inliers.
        # Negative of decision_function or score_samples is often used as anomaly score.
        # Here, -score_samples() makes higher scores more anomalous.
        scores = -model.score_samples(X) 
    
    # Convert predictions: -1 (outlier) -> 1 (fraud), 1 (inlier) -> 0 (normal)
    binary_predictions = np.where(predictions == -1, 1, 0)
    return binary_predictions, scores


def calculate_metrics(y_true, y_pred_binary, scores_anomaly):
    """
    Calculate and return model performance metrics.
    y_true: true labels
    y_pred_binary: binary predictions (0 for normal, 1 for fraud/anomaly)
    scores_anomaly: anomaly scores where HIGHER means MORE ANOMALOUS
    """
    if y_true is None or y_pred_binary is None or scores_anomaly is None:
        return {}
        
    # Ensure scores_anomaly are correctly oriented (higher = more anomalous) for roc_curve and precision_recall_curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores_anomaly)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores_anomaly)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'confusion_matrix': confusion_matrix(y_true, y_pred_binary),
        'classification_report': classification_report(y_true, y_pred_binary, zero_division=0),
        'roc_curve_data': (fpr, tpr, roc_thresholds),
        'pr_curve_data': (precision, recall, pr_thresholds),
        'roc_auc': auc(fpr, tpr),
        'auprc': auc(recall, precision) # Area Under Precision-Recall Curve
    }

def plot_roc_curves(y_true, model_scores_dict, config):
    """
    Plot ROC curves for multiple models.
    model_scores_dict: {'Model Name': anomaly_scores_higher_is_more_anomalous, ...}
    """
    if y_true is None: return None
    plt.figure(figsize=config.FIGSIZE)
    
    auc_scores = {}
    for model_name, anomaly_scores in model_scores_dict.items():
        if anomaly_scores is None: continue
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess') # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt, auc_scores

def plot_pr_curves(y_true, model_scores_dict, config):
    """
    Plot Precision-Recall curves for multiple models.
    model_scores_dict: {'Model Name': anomaly_scores_higher_is_more_anomalous, ...}
    """
    if y_true is None: return None
    plt.figure(figsize=config.FIGSIZE)
    
    auprc_scores = {}
    for model_name, anomaly_scores in model_scores_dict.items():
        if anomaly_scores is None: continue
        precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
        pr_auc = auc(recall, precision)
        auprc_scores[model_name] = pr_auc
        plt.plot(recall, precision, label=f'{model_name} (AUPRC = {pr_auc:.4f})')
        
    # Baseline for PR curve (depends on class distribution)
    if y_true.sum() > 0: # Ensure there are positive samples
        baseline = y_true.sum() / len(y_true)
        plt.axhline(baseline, linestyle='--', color='grey', label=f'No-Skill Classifier (Prevalence = {baseline:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt, auprc_scores


def plot_anomaly_scores_distribution(model_scores_dict, y_true, config):
    """
    Plot anomaly score distributions for multiple models.
    model_scores_dict: {'Model Name': {'scores': anomaly_scores, 'lower_is_anomaly': True/False}, ...}
    y_true: The true labels for hue.
    """
    if y_true is None: return None
    
    num_models = len(model_scores_dict)
    if num_models == 0: return None

    fig, axes = plt.subplots(num_models, 1, figsize=(config.FIGSIZE[0], config.FIGSIZE[1] * num_models * 0.6 ))
    if num_models == 1: # Ensure axes is always a list
        axes = [axes]

    eval_dfs = {}

    for i, (model_name, data) in enumerate(model_scores_dict.items()):
        scores = data['scores']
        lower_is_anomaly = data.get('lower_is_anomaly', False) # Default: higher score = more anomalous

        if scores is None: continue

        current_eval_df = pd.DataFrame({'True_Class': y_true, 'Score': scores})
        eval_dfs[model_name] = current_eval_df
        
        ax = axes[i]
        sns.histplot(
            data=current_eval_df, x='Score', hue='True_Class', 
            bins=50, kde=True, ax=ax,
            palette={0: 'blue', 1: 'red'},
            element='step', stat="density"
        )
        ax.set_title(f'{model_name} Anomaly Score Distribution')
        
        score_label = 'Anomaly Score'
        if lower_is_anomaly:
            score_label += ' (lower = more anomalous)'
            quantile_val = current_eval_df['Score'].quantile(config.IF_CONTAMINATION) # Use contamination as proxy
            quantile_label = f'{config.IF_CONTAMINATION*100:.1f}% Quantile'
        else:
            score_label += ' (higher = more anomalous)'
            quantile_val = current_eval_df['Score'].quantile(1 - config.IF_CONTAMINATION) # Use 1-contamination
            quantile_label = f'{(1-config.IF_CONTAMINATION)*100:.1f}% Quantile'
            
        ax.set_xlabel(score_label)
        
        line = ax.axvline(quantile_val, color='green', linestyle='--', 
                          label=f'{quantile_label}: {quantile_val:.4f}')
        
        # Correct legend handling
        handles, labels = ax.get_legend_handles_labels()
        # Check if '1% Quantile' or similar is already in labels to avoid duplicates if re-running
        if not any(quantile_label in lab for lab in labels):
             # Find the auto-generated legend items for Normal (0) and Fraud (1)
            class_labels = ['Normal', 'Fraud'] 
            try:
                # Attempt to map hue values (0, 1) to class labels
                # This assumes seaborn creates legend entries in order of hue values
                # Or that labels are already '0' and '1' which we can map
                new_labels = []
                found_normal = False
                found_fraud = False
                for lab_idx, lab_text in enumerate(labels):
                    if '0' in lab_text and not found_normal : # Heuristic
                        new_labels.append(class_labels[0])
                        found_normal = True
                    elif '1' in lab_text and not found_fraud: # Heuristic
                        new_labels.append(class_labels[1])
                        found_fraud = True
                    else:
                        new_labels.append(lab_text) # Keep other legend items
                
                # Add the quantile line's handle and label
                ax.legend(handles=handles + [line], labels=new_labels + [line.get_label()])

            except Exception as e: # Fallback if mapping is tricky
                print(f"Legend adjustment warning for {model_name}: {e}")
                ax.legend() # Default legend
        else:
            ax.legend()


    plt.tight_layout()
    return plt, eval_dfs


def plot_score_comparison_scatter(eval_dfs, model1_name, model2_name, config):
    """
    Plot a scatter comparison of anomaly scores between two models.
    eval_dfs: Dictionary of DataFrames from plot_anomaly_scores_distribution.
    model1_name, model2_name: Keys in eval_dfs.
    """
    if model1_name not in eval_dfs or model2_name not in eval_dfs:
        print("Error: One or both model names not found in eval_dfs.")
        return None
        
    df1 = eval_dfs[model1_name]
    df2 = eval_dfs[model2_name]
    
    # Ensure 'True_Class' is consistent
    comparison_df = pd.DataFrame({
        f'{model1_name}_Score': df1['Score'],
        f'{model2_name}_Score': df2['Score'],
        'True_Class': df1['True_Class'] # Assuming 'True_Class' is identical
    })

    plt.figure(figsize=config.FIGSIZE)
    scatter = plt.scatter(
        comparison_df[f'{model1_name}_Score'], 
        comparison_df[f'{model2_name}_Score'],
        c=comparison_df['True_Class'],
        cmap='coolwarm', # Red for fraud (1), Blue for normal (0)
        alpha=0.5, # Adjust for point density
        s=30 # Adjust point size
    )
    plt.colorbar(scatter, label='Class (0=Normal, 1=Fraud)')
    plt.xlabel(f'{model1_name} Score') # Add (lower/higher is anomaly) based on model
    plt.ylabel(f'{model2_name} Score') # Add (lower/higher is anomaly) based on model
    plt.title(f'Comparison of Anomaly Scores: {model1_name} vs {model2_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt

def create_comparison_report(metrics_dict):
    """
    Create a comparison DataFrame of model performance from calculated metrics.
    metrics_dict: {'Model Name': {'roc_auc': val, 'auprc': val, 'accuracy': val}, ...}
    """
    report_data = []
    for model_name, metrics in metrics_dict.items():
        report_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', np.nan),
            'ROC-AUC': metrics.get('roc_auc', np.nan),
            'AUPRC': metrics.get('auprc', np.nan)
            # Add other metrics from classification_report if needed, e.g., F1-score for class 1
        })
    
    report_df = pd.DataFrame(report_data)
    return report_df.set_index('Model')

if __name__ == '__main__':
    config = Config()

    # Load data
    df = load_data(config)
    
    if df is not None:
        # Get data summary and potentially update contamination
        summary = get_data_summary(df)
        print("Data Summary:", summary)
        # OPTIONAL: Update contamination based on actual fraud percentage
        # if summary and 'fraud_percentage' in summary:
        #     actual_contamination = summary['fraud_percentage'] / 100
        #     if actual_contamination > 0: # Ensure it's not zero
        #         config.IF_CONTAMINATION = actual_contamination
        #         config.LOF_CONTAMINATION = actual_contamination
        #         print(f"Updated IF/LOF contamination to: {actual_contamination:.6f}")

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df, config)

        if X_train is not None:
            # Train models
            print("\nTraining Isolation Forest...")
            if_model = train_isolation_forest(X_train, config)
            
            print("\nTraining Local Outlier Factor...")
            lof_model = train_lof(X_train, config)

            # Get predictions and scores
            if_preds_binary, if_scores_anomaly = None, None
            if if_model:
                if_preds_binary, if_scores_anomaly = get_model_predictions(if_model, X_test, is_isolation_forest=True)

            lof_preds_binary, lof_scores_anomaly = None, None
            if lof_model:
                lof_preds_binary, lof_scores_anomaly = get_model_predictions(lof_model, X_test, is_isolation_forest=False)

            # Calculate metrics
            all_metrics = {}
            if if_preds_binary is not None:
                print("\nIsolation Forest Metrics:")
                if_metrics = calculate_metrics(y_test, if_preds_binary, if_scores_anomaly)
                all_metrics['Isolation Forest'] = if_metrics
                print(f"  Accuracy: {if_metrics.get('accuracy', np.nan):.4f}")
                print(f"  ROC-AUC: {if_metrics.get('roc_auc', np.nan):.4f}")
                print(f"  AUPRC: {if_metrics.get('auprc', np.nan):.4f}")
                # print("  Confusion Matrix:\n", if_metrics.get('confusion_matrix'))
                # print("  Classification Report:\n", if_metrics.get('classification_report'))


            if lof_preds_binary is not None:
                print("\nLocal Outlier Factor Metrics:")
                lof_metrics = calculate_metrics(y_test, lof_preds_binary, lof_scores_anomaly)
                all_metrics['Local Outlier Factor'] = lof_metrics
                print(f"  Accuracy: {lof_metrics.get('accuracy', np.nan):.4f}")
                print(f"  ROC-AUC: {lof_metrics.get('roc_auc', np.nan):.4f}")
                print(f"  AUPRC: {lof_metrics.get('auprc', np.nan):.4f}")
                # print("  Confusion Matrix:\n", lof_metrics.get('confusion_matrix'))
                # print("  Classification Report:\n", lof_metrics.get('classification_report'))
            
            # Create comparison report
            comparison_df = create_comparison_report(all_metrics)
            print("\nModel Comparison Report:")
            print(comparison_df)

            # Plotting
            model_scores_for_plots = {}
            if if_scores_anomaly is not None:
                model_scores_for_plots['Isolation Forest'] = if_scores_anomaly
            if lof_scores_anomaly is not None:
                model_scores_for_plots['Local Outlier Factor'] = lof_scores_anomaly

            if model_scores_for_plots:
                roc_plot, roc_aucs = plot_roc_curves(y_test, model_scores_for_plots, config)
                if roc_plot: roc_plot.show()
                
                pr_plot, pr_aucs = plot_pr_curves(y_test, model_scores_for_plots, config)
                if pr_plot: pr_plot.show()

            # For plot_anomaly_scores_distribution, it expects a slightly different dict structure
            model_scores_for_dist_plot = {}
            if if_scores_anomaly is not None:
                 # IF: score_samples() is higher for inliers. -score_samples() means higher is anomaly.
                model_scores_for_dist_plot['Isolation Forest'] = {'scores': if_scores_anomaly, 'lower_is_anomaly': False}
            if lof_scores_anomaly is not None:
                # LOF: score_samples() is higher for inliers. -score_samples() means higher is anomaly.
                model_scores_for_dist_plot['Local Outlier Factor'] = {'scores': lof_scores_anomaly, 'lower_is_anomaly': False}

            if model_scores_for_dist_plot:
                dist_plot, eval_dfs_dist = plot_anomaly_scores_distribution(model_scores_for_dist_plot, y_test, config)
                if dist_plot: dist_plot.show()

                if 'Isolation Forest' in eval_dfs_dist and 'Local Outlier Factor' in eval_dfs_dist:
                    scatter_comp_plot = plot_score_comparison_scatter(eval_dfs_dist, 'Isolation Forest', 'Local Outlier Factor', config)
                    if scatter_comp_plot: scatter_comp_plot.show()
            
            # Example: Plotting class distribution
            # class_dist_plot = plot_class_distribution(df, config)
            # if class_dist_plot: class_dist_plot.show()
            
            # Example: Plotting amount distribution
            # amount_dist_plot = plot_amount_distribution(df, config)
            # if amount_dist_plot: amount_dist_plot.show()

