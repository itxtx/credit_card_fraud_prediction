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
                           auc, roc_curve)
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
    IF_CONTAMINATION = 0.01  # Percentage of outliers in the dataset, to be determined
    IF_N_ESTIMATORS = 100
    IF_MAX_SAMPLES = 'auto'
    
    # LOF parameters
    LOF_N_NEIGHBORS = 20
    LOF_CONTAMINATION = 0.01  # Same as IF_CONTAMINATION by default
    
    # Resampling parameters
    USE_RESAMPLING = False
    RESAMPLING_STRATEGY = 'none'  # Options: 'none', 'ros', 'rus', 'smote', 'adasyn', 'nearmiss', 'tomek', 'enn', 'smote-tomek', 'smote-enn'
    SAMPLING_RATIO = 0.1  # Target ratio of minority to majority class (or 'auto')
    
    # Visualization
    FIGSIZE = (12, 8)

def load_data(config):
    """Load and return the credit card fraud dataset."""
    return pd.read_csv(config.DATA_PATH)

def get_data_summary(df):
    """Get a summary of the dataset."""
    # Check class distribution
    fraud_count = df['Class'].value_counts()[1]
    normal_count = df['Class'].value_counts()[0]
    fraud_percentage = (fraud_count / df.shape[0]) * 100
    
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
    plt.figure(figsize=config.FIGSIZE)
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    return plt

def plot_amount_distribution(df, config):
    """Plot the transaction amount distribution by class."""
    plt.figure(figsize=config.FIGSIZE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot amount distribution by class
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax1)
    ax1.set_title('Transaction Amount Distribution by Class')
    ax1.set_yscale('log')
    ax1.set_xticklabels(['Normal', 'Fraud'])
    
    # Compare normal vs fraudulent transaction amounts
    amount_fraud = df[df['Class'] == 1]['Amount']
    amount_normal = df[df['Class'] == 0]['Amount']
    
    sns.histplot(amount_fraud, color='red', label='Fraud', ax=ax2, kde=True, bins=50)
    sns.histplot(amount_normal, color='blue', label='Normal', ax=ax2, kde=True, bins=50, alpha=0.5)
    ax2.set_title('Transaction Amount Distribution: Normal vs Fraud')
    ax2.set_xscale('log')
    ax2.legend()
    
    plt.tight_layout()
    return plt

def plot_correlation_matrix(df, config):
    """Plot the correlation matrix of features."""
    plt.figure(figsize=(15, 12))
    correlation_matrix = df.corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt, correlation_matrix

def get_top_correlations(correlation_matrix, feature='Class', top_n=10):
    """Get top correlated features with the target feature."""
    correlations = correlation_matrix[feature].sort_values(ascending=False)
    return correlations[1:top_n+1]  # Exclude self-correlation

def apply_resampling(X, y, strategy, config):
    """Apply the specified resampling strategy to the data."""
    if not IMBLEARN_AVAILABLE:
        print("Warning: imblearn not installed. Skipping resampling.")
        return X, y
    
    if strategy == 'none' or strategy is None:
        return X, y
    
    print(f"Applying {strategy} resampling strategy...")
    
    # Sampling ratio
    sampling_ratio = config.SAMPLING_RATIO
    if sampling_ratio != 'auto':
        sampling_strategy = {0: int(y.value_counts()[0]), 
                            1: int(y.value_counts()[0] * sampling_ratio)}
    else:
        sampling_strategy = 'auto'
    
    # Initialize the appropriate resampler
    if strategy == 'ros':
        resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=config.RANDOM_STATE)
    elif strategy == 'rus':
        resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=config.RANDOM_STATE)
    elif strategy == 'smote':
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=config.RANDOM_STATE)
    elif strategy == 'adasyn':
        resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=config.RANDOM_STATE)
    elif strategy == 'nearmiss':
        resampler = NearMiss(sampling_strategy=sampling_strategy)
    elif strategy == 'tomek':
        resampler = TomekLinks(sampling_strategy='majority')
    elif strategy == 'enn':
        resampler = EditedNearestNeighbours(sampling_strategy='majority')
    elif strategy == 'smote-tomek':
        resampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=config.RANDOM_STATE)
    elif strategy == 'smote-enn':
        resampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=config.RANDOM_STATE)
    else:
        print(f"Unknown resampling strategy: {strategy}. Using original data.")
        return X, y
    
    # Apply resampling
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    # Print class distribution before and after resampling
    print(f"Class distribution before resampling: {dict(sorted(y.value_counts().items()))}")
    print(f"Class distribution after resampling: {dict(sorted(pd.Series(y_resampled).value_counts().items()))}")
    
    return X_resampled, y_resampled

def preprocess_data(df, config):
    """Preprocess the data and return train/test splits."""
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Features that need scaling
    # Time and Amount need scaling, the other features (V1-V28) are already PCA transformed
    features_to_scale = ['Time', 'Amount']
    
    # Create a copy of X to avoid the SettingWithCopyWarning
    X_preprocessed = X.copy()
    
    # Apply scaling to Time and Amount
    scaler = StandardScaler()
    X_preprocessed[features_to_scale] = scaler.fit_transform(X_preprocessed[features_to_scale])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y  # Ensure both sets have the same proportion of fraud cases
    )
    
    # Apply resampling if configured
    if config.USE_RESAMPLING:
        X_train, y_train = apply_resampling(X_train, y_train, config.RESAMPLING_STRATEGY, config)
    
    return X_train, X_test, y_train, y_test

def train_isolation_forest(X_train, config):
    """Train and return an Isolation Forest model."""
    model = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        max_samples=config.IF_MAX_SAMPLES,
        contamination=config.IF_CONTAMINATION,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train)
    return model

def train_lof(X_train, config):
    """Train and return a Local Outlier Factor model."""
    model = LocalOutlierFactor(
        n_neighbors=config.LOF_N_NEIGHBORS,
        contamination=config.LOF_CONTAMINATION,
        novelty=True
    )
    model.fit(X_train)
    return model

def get_model_predictions(model, X, is_isolation_forest=True):
    """Get predictions and anomaly scores from a model."""
    if is_isolation_forest:
        predictions = model.predict(X)
        scores = model.score_samples(X)
    else:
        predictions = model.predict(X)
        scores = -model.score_samples(X)
    
    # Convert predictions: -1 (outlier) -> 1 (fraud), 1 (inlier) -> 0 (normal)
    predictions = np.where(predictions == -1, 1, 0)
    return predictions, scores

def calculate_metrics(y_true, y_pred, scores):
    """Calculate and return model performance metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred),
        'roc_curve': roc_curve(y_true, scores),
        'pr_curve': precision_recall_curve(y_true, scores)
    }

def plot_roc_curves(y_true, if_scores, lof_scores):
    """Plot ROC curves for both models."""
    plt.figure(figsize=(12, 8))
    
    # ROC Curve for Isolation Forest
    fpr_if, tpr_if, _ = roc_curve(y_true, -if_scores)  # Negative scores because lower = more anomalous
    auc_if = auc(fpr_if, tpr_if)
    plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC = {auc_if:.4f})')
    
    # ROC Curve for LOF
    fpr_lof, tpr_lof, _ = roc_curve(y_true, lof_scores)  # Higher score = more anomalous
    auc_lof = auc(fpr_lof, tpr_lof)
    plt.plot(fpr_lof, tpr_lof, label=f'Local Outlier Factor (AUC = {auc_lof:.4f})')
    
    # Plot the random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt, auc_if, auc_lof

def calculate_auprc(y_true, scores):
    """Calculate Area Under the Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return auc(recall, precision)

def plot_pr_curves(y_true, if_scores, lof_scores):
    """Plot Precision-Recall curves for both models."""
    plt.figure(figsize=(12, 8))
    
    # PR Curve for Isolation Forest
    precision_if, recall_if, _ = precision_recall_curve(y_true, -if_scores)
    plt.plot(recall_if, precision_if, label='Isolation Forest')
    
    # PR Curve for LOF
    precision_lof, recall_lof, _ = precision_recall_curve(y_true, lof_scores)
    plt.plot(recall_lof, precision_lof, label='Local Outlier Factor')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt

def plot_anomaly_scores(if_scores, lof_scores, y_true):
    """Plot anomaly score distributions for both models."""
    # Create a dataframe for visualization
    eval_df = pd.DataFrame({
        'True_Class': y_true,
        'IF_Score': if_scores,
        'LOF_Score': lof_scores
    })
    
    # Set up the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Isolation Forest scores distribution
    sns.histplot(
        data=eval_df, x='IF_Score', hue='True_Class', 
        bins=50, kde=True, ax=ax1,
        palette={0: 'blue', 1: 'red'},
        element='step'
    )
    ax1.set_title('Isolation Forest Anomaly Score Distribution')
    ax1.set_xlabel('Anomaly Score (lower = more anomalous)')
    ax1.axvline(eval_df['IF_Score'].quantile(0.01), color='green', linestyle='--', 
                label=f'1% Quantile: {eval_df["IF_Score"].quantile(0.01):.4f}')
    ax1.legend(['1% Quantile', 'Normal', 'Fraud'])
    
    # LOF scores distribution
    sns.histplot(
        data=eval_df, x='LOF_Score', hue='True_Class', 
        bins=50, kde=True, ax=ax2,
        palette={0: 'blue', 1: 'red'},
        element='step'
    )
    ax2.set_title('Local Outlier Factor Score Distribution')
    ax2.set_xlabel('Anomaly Score (higher = more anomalous)')
    ax2.axvline(eval_df['LOF_Score'].quantile(0.99), color='green', linestyle='--', 
                label=f'99% Quantile: {eval_df["LOF_Score"].quantile(0.99):.4f}')
    ax2.legend(['99% Quantile', 'Normal', 'Fraud'])
    
    plt.tight_layout()
    return plt, eval_df

def plot_score_comparison(eval_df):
    """Plot a scatter comparison of anomaly scores between models."""
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        eval_df['IF_Score'], 
        eval_df['LOF_Score'],
        c=eval_df['True_Class'],
        cmap='coolwarm',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label='Class (0=Normal, 1=Fraud)')
    plt.xlabel('Isolation Forest Score (lower = more anomalous)')
    plt.ylabel('LOF Score (higher = more anomalous)')
    plt.title('Comparison of Anomaly Scores Between Models')
    plt.grid(True, linestyle='--', alpha=0.5)
    return plt

def compare_models(y_true, if_preds, lof_preds, if_scores, lof_scores, auc_if, auc_lof):
    """Create a comparison dataframe of model performance."""
    # Calculate AUPRC for both models
    auprc_if = calculate_auprc(y_true, -if_scores)  # Negative for IF because lower = more anomalous
    auprc_lof = calculate_auprc(y_true, lof_scores)
    
    return pd.DataFrame({
        'Model': ['Isolation Forest', 'Local Outlier Factor'],
        'Accuracy': [accuracy_score(y_true, if_preds), accuracy_score(y_true, lof_preds)],
        'ROC-AUC': [auc_if, auc_lof],
        'AUPRC': [auprc_if, auprc_lof]
    })