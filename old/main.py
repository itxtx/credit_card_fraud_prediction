import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, auc, roc_curve
import warnings
warnings.filterwarnings('ignore')

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
    
    # Visualization
    FIGSIZE = (12, 8)

# 1. Understanding the Problem and Data
def load_and_understand_data(config):
    print("1. Loading and Understanding the Data...")
    
    # Load data
    print("Loading data from:", config.DATA_PATH)
    df = pd.read_csv(config.DATA_PATH)
    
    # Print basic information
    print("\nData Overview:")
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Transactions: {df.shape[0]}")
    print(f"Number of Features: {df.shape[1]}")
    
    # Check data types and missing values
    print("\nData Types and Missing Values:")
    print(df.info())
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {missing_values}")
    
    # Check class distribution
    fraud_count = df['Class'].value_counts()[1]
    normal_count = df['Class'].value_counts()[0]
    fraud_percentage = (fraud_count / df.shape[0]) * 100
    
    print("\nClass Distribution:")
    print(f"Normal Transactions: {normal_count} ({100 - fraud_percentage:.2f}%)")
    print(f"Fraudulent Transactions: {fraud_count} ({fraud_percentage:.2f}%)")
    
    # Update the contamination parameter based on actual fraud percentage
    config.IF_CONTAMINATION = fraud_percentage / 100
    config.LOF_CONTAMINATION = fraud_percentage / 100
    print(f"\nSetting contamination parameter to: {config.IF_CONTAMINATION:.4f}")
    
    return df

# 2. Exploratory Data Analysis
def exploratory_data_analysis(df, config):
    print("\n2. Performing Exploratory Data Analysis...")
    
    # Plot class distribution
    plt.figure(figsize=config.FIGSIZE)
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Examine transaction amounts
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
    plt.savefig('amount_distributions.png')
    plt.close()
    
    # Correlation analysis
    plt.figure(figsize=(15, 12))
    correlation_matrix = df.corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Examine top correlated features with Class
    class_correlations = correlation_matrix['Class'].sort_values(ascending=False)
    print("\nTop 10 Features Correlated with Fraud:")
    print(class_correlations[1:11])  # Exclude self-correlation
    
    # Return top correlated features (might be useful for feature selection)
    return class_correlations

# 3. Data Preprocessing
def preprocess_data(df, config):
    print("\n3. Preprocessing the Data...")
    
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
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# 4. Isolation Forest Model
def train_isolation_forest(X_train, X_test, y_train, y_test, config):
    print("\n4. Training Isolation Forest Model...")
    
    # Initialize and train the Isolation Forest model
    if_model = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        max_samples=config.IF_MAX_SAMPLES,
        contamination=config.IF_CONTAMINATION,
        random_state=config.RANDOM_STATE
    )
    
    # Fit the model on training data
    if_model.fit(X_train)
    
    # Predict anomalies
    # Isolation Forest returns -1 for outliers and 1 for inliers, so we need to convert
    y_pred_train = if_model.predict(X_train)
    y_pred_test = if_model.predict(X_test)
    
    # Convert predictions: -1 (outlier) -> 1 (fraud), 1 (inlier) -> 0 (normal)
    y_pred_train = np.where(y_pred_train == -1, 1, 0)
    y_pred_test = np.where(y_pred_test == -1, 1, 0)
    
    # Calculate anomaly scores
    # Lower score (more negative) = more likely to be an anomaly
    anomaly_scores_train = if_model.score_samples(X_train)
    anomaly_scores_test = if_model.score_samples(X_test)
    
    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred_test))
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    
    return if_model, anomaly_scores_test, y_pred_test

# 5. Local Outlier Factor (LOF) Model
def train_lof(X_train, X_test, y_train, y_test, config):
    print("\n5. Training Local Outlier Factor Model...")
    
    # Initialize and train the LOF model
    lof_model = LocalOutlierFactor(
        n_neighbors=config.LOF_N_NEIGHBORS,
        contamination=config.LOF_CONTAMINATION
    )
    
    # LOF is transductive, not inductive, so we fit and predict in one step
    # LOF returns -1 for outliers and 1 for inliers
    y_pred_train = lof_model.fit_predict(X_train)
    
    # For test set, we need to use a separate fitted model
    lof_fitted = LocalOutlierFactor(
        n_neighbors=config.LOF_N_NEIGHBORS,
        contamination=config.LOF_CONTAMINATION,
        novelty=True  # Required for predicting on new samples
    )
    lof_fitted.fit(X_train)
    y_pred_test = lof_fitted.predict(X_test)
    
    # Convert predictions: -1 (outlier) -> 1 (fraud), 1 (inlier) -> 0 (normal)
    y_pred_train = np.where(y_pred_train == -1, 1, 0)
    y_pred_test = np.where(y_pred_test == -1, 1, 0)
    
    # Calculate negative outlier factor scores
    # More negative = more likely to be an outlier
    anomaly_scores_train = -lof_fitted.negative_outlier_factor_
    anomaly_scores_test = -lof_fitted.score_samples(X_test)
    
    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred_test))
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    
    return lof_fitted, anomaly_scores_test, y_pred_test

# 6. Evaluate and Compare Models
def evaluate_and_compare_models(y_test, if_scores, if_preds, lof_scores, lof_preds):
    print("\n6. Evaluating and Comparing Models...")
    
    # Create a dataframe for visualization
    eval_df = pd.DataFrame({
        'True_Class': y_test,
        'IF_Prediction': if_preds,
        'IF_Score': if_scores,
        'LOF_Prediction': lof_preds,
        'LOF_Score': lof_scores
    })
    
    # ROC Curve and AUC for both models
    plt.figure(figsize=(12, 8))
    
    # ROC Curve for Isolation Forest
    fpr_if, tpr_if, _ = roc_curve(y_test, -if_scores)  # Negative scores because lower = more anomalous
    auc_if = auc(fpr_if, tpr_if)
    plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC = {auc_if:.4f})')
    
    # ROC Curve for LOF
    fpr_lof, tpr_lof, _ = roc_curve(y_test, lof_scores)  # Higher score = more anomalous
    auc_lof = auc(fpr_lof, tpr_lof)
    plt.plot(fpr_lof, tpr_lof, label=f'Local Outlier Factor (AUC = {auc_lof:.4f})')
    
    # Plot the random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('roc_comparison.png')
    plt.close()
    
    # Precision-Recall Curve for both models
    plt.figure(figsize=(12, 8))
    
    # PR Curve for Isolation Forest
    precision_if, recall_if, _ = precision_recall_curve(y_test, -if_scores)
    plt.plot(recall_if, precision_if, label='Isolation Forest')
    
    # PR Curve for LOF
    precision_lof, recall_lof, _ = precision_recall_curve(y_test, lof_scores)
    plt.plot(recall_lof, precision_lof, label='Local Outlier Factor')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('pr_comparison.png')
    plt.close()
    
    # Compare detection accuracy
    comparisons = pd.DataFrame({
        'Model': ['Isolation Forest', 'Local Outlier Factor'],
        'Accuracy': [accuracy_score(y_test, if_preds), accuracy_score(y_test, lof_preds)],
        'AUC': [auc_if, auc_lof]
    })
    
    print("\nModel Comparison:")
    print(comparisons)
    
    return eval_df, comparisons

# 7. Visualize Anomaly Scores
def visualize_anomaly_scores(eval_df):
    print("\n7. Visualizing Anomaly Scores...")
    
    # Visualize anomaly scores distribution for both models
    plt.figure(figsize=(15, 10))
    
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
    plt.savefig('anomaly_score_distributions.png')
    plt.close()
    
    # Scatter plot of anomaly scores
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
    plt.savefig('anomaly_score_comparison.png')
    plt.close()

# Main function to execute the entire pipeline
def main():
    # Initialize configuration
    config = Config()
    
    # Load and understand the data
    df = load_and_understand_data(config)
    
    # Explore the data
    class_correlations = exploratory_data_analysis(df, config)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df, config)
    
    # Train Isolation Forest model
    if_model, if_scores, if_preds = train_isolation_forest(X_train, X_test, y_train, y_test, config)
    
    # Train LOF model
    lof_model, lof_scores, lof_preds = train_lof(X_train, X_test, y_train, y_test, config)
    
    # Evaluate and compare models
    eval_df, comparisons = evaluate_and_compare_models(y_test, if_scores, if_preds, lof_scores, lof_preds)
    
    # Visualize anomaly scores
    visualize_anomaly_scores(eval_df)
    
    print("\nCredit Card Fraud Detection Analysis Complete!")
    print("Check the generated visualization files to explore the results further.")

# Execute the main function
if __name__ == "__main__":
    main()