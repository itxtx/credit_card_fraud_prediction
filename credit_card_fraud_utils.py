import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # MinMaxScaler removed as it was not actively used for IF/LOF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_curve, 
                             auc, roc_curve, average_precision_score)
import warnings

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# Global constants for visualization
FIGSIZE = (12, 8)
SUBPLOT_FIGSIZE = (12,12)
HEATMAP_FIGSIZE = (15,12)


# --- PyTorch Autoencoder Definition ---
class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim1, hidden_dim2):
        super(PyTorchAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(True),
            nn.Linear(hidden_dim2, encoding_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim2),
            nn.ReLU(True),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(True),
            nn.Linear(hidden_dim1, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_pytorch_autoencoder(model, X_train_normal_tensor, 
                              epochs, batch_size, learning_rate, device):
    """Train the PyTorch Autoencoder model."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(X_train_normal_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data_batch in train_loader:
            inputs = data_batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs -1 :
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}')
    return model

def get_ae_reconstruction_errors(model, X_data_tensor, device, batch_size=256):
    """Get reconstruction errors (MSE) for the given data using the trained AE."""
    model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction='none') 
    
    dataset = TensorDataset(X_data_tensor)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    all_errors = []
    with torch.no_grad():
        for data_batch in loader:
            inputs = data_batch[0].to(device)
            reconstructions = model(inputs)
            errors = criterion(reconstructions, inputs).mean(axis=1) 
            all_errors.extend(errors.cpu().numpy())
            
    return np.array(all_errors)

# --- Utility Functions ---
def load_data(data_path):
    """Load and return the credit card fraud dataset."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        print(f"Error: The data file was not found at {data_path}")
        return None

def get_data_summary(df):
    if df is None: return None
    fraud_count = df['Class'].value_counts().get(1, 0)
    normal_count = df['Class'].value_counts().get(0, 0)
    fraud_percentage = (fraud_count / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    return {'shape': df.shape, 'missing_values': df.isnull().sum().sum(),
            'fraud_count': fraud_count, 'normal_count': normal_count,
            'fraud_percentage': fraud_percentage}

def plot_class_distribution(df):
    if df is None: return None
    plt.figure(figsize=FIGSIZE)
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.xticks([0, 1], ['Normal', 'Fraud']); plt.xlabel('Transaction Type'); plt.ylabel('Count')
    return plt

def plot_amount_distribution(df):
    if df is None: return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=SUBPLOT_FIGSIZE)
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax1)
    ax1.set_title('Transaction Amount Distribution by Class'); ax1.set_yscale('log'); ax1.set_xticklabels(['Normal', 'Fraud'])
    amount_fraud = df[df['Class'] == 1]['Amount']; amount_normal = df[df['Class'] == 0]['Amount']
    sns.histplot(amount_fraud, color='red', label='Fraud', ax=ax2, kde=True, bins=50, stat="density")
    sns.histplot(amount_normal, color='blue', label='Normal', ax=ax2, kde=True, bins=50, alpha=0.6, stat="density")
    ax2.set_title('Transaction Amount Distribution: Normal vs Fraud'); ax2.set_xscale('log'); ax2.legend()
    plt.tight_layout(); return plt

def plot_correlation_matrix(df):
    if df is None: return None
    plt.figure(figsize=HEATMAP_FIGSIZE)
    correlation_matrix = df.drop(columns=['Time'], errors='ignore').corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title('Feature Correlation Matrix (excluding Time)'); plt.tight_layout()
    return plt, correlation_matrix

def get_top_correlations(correlation_matrix, feature='Class', top_n=10):
    if correlation_matrix is None or feature not in correlation_matrix: return None
    correlations = correlation_matrix[feature].sort_values(ascending=False)
    return correlations.drop(feature, errors='ignore').head(top_n)

def preprocess_data(df, test_size_param, random_state_param):
    """Preprocesses data: splits and scales 'Time' and 'Amount'."""
    if df is None: return None, None, None, None, None, None
    
    X_orig = df.drop('Class', axis=1)
    y_orig = df['Class']
    features_to_scale = ['Time', 'Amount']
    
    # Split data into training and testing sets
    X_train_raw, X_test_raw, y_train_series, y_test_series = train_test_split(
        X_orig, y_orig, 
        test_size=test_size_param, 
        random_state=random_state_param, 
        stratify=y_orig
    )
    
    # Work with copies for scaling
    X_train_df = X_train_raw.copy()
    X_test_df = X_test_raw.copy()
    
    # Initialize and apply scaling
    scaler = StandardScaler()
    X_train_df[features_to_scale] = scaler.fit_transform(X_train_raw[features_to_scale])
    X_test_df[features_to_scale] = scaler.transform(X_test_raw[features_to_scale])
    
    # Data for Autoencoder (normal instances from the original training split, before any potential resampling)
    X_train_ae_normal_np = X_train_df[y_train_series == 0].values
    
    # Convert to NumPy arrays for model training
    X_train_np = X_train_df.values
    y_train_np = y_train_series.values
    X_test_np = X_test_df.values
    y_test_np = y_test_series.values

    ae_input_dim = X_train_np.shape[1]

    return X_train_np, X_test_np, y_train_np, y_test_np, X_train_ae_normal_np, ae_input_dim

def get_model_predictions(model, X_np, is_isolation_forest=True):
    if model is None or X_np is None: return None, None
    predictions = model.predict(X_np)
    # For IF and LOF (novelty=True), score_samples returns higher for inliers.
    # We want scores where higher = more anomalous for consistent evaluation.
    scores = -model.score_samples(X_np) 
    binary_predictions = np.where(predictions == -1, 1, 0)
    return binary_predictions, scores

def calculate_metrics(y_true, y_pred_binary, scores_anomaly):
    if y_true is None or y_pred_binary is None or scores_anomaly is None: return {}
    fpr, tpr, _ = roc_curve(y_true, scores_anomaly)
    precision, recall, _ = precision_recall_curve(y_true, scores_anomaly)
    return {'accuracy': accuracy_score(y_true, y_pred_binary),
            'confusion_matrix': confusion_matrix(y_true, y_pred_binary),
            'classification_report': classification_report(y_true, y_pred_binary, zero_division=0),
            'roc_auc': auc(fpr, tpr), 'auprc': auc(recall, precision)}

def plot_roc_curves(y_true, model_scores_dict):
    if y_true is None: return None
    plt.figure(figsize=FIGSIZE); auc_scores = {}
    for model_name, anomaly_scores in model_scores_dict.items():
        if anomaly_scores is None: continue
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        roc_auc = auc(fpr, tpr); auc_scores[model_name] = roc_auc
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve Comparison')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); return plt, auc_scores

def plot_pr_curves(y_true, model_scores_dict):
    if y_true is None: return None
    plt.figure(figsize=FIGSIZE); auprc_scores = {}
    for model_name, anomaly_scores in model_scores_dict.items():
        if anomaly_scores is None: continue
        precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
        pr_auc = auc(recall, precision); auprc_scores[model_name] = pr_auc
        plt.plot(recall, precision, label=f'{model_name} (AUPRC = {pr_auc:.4f})')
    if y_true.sum() > 0:
        baseline = y_true.sum() / len(y_true)
        plt.axhline(baseline, linestyle='--', color='grey', label=f'No-Skill (Prevalence = {baseline:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve Comparison')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); return plt, auprc_scores

def plot_anomaly_scores_distribution(model_scores_dist_dict, y_true, 
                                     default_contamination_proxy, ae_threshold_percentile_proxy):
    if y_true is None or not model_scores_dist_dict: return None, {}
    num_models = len(model_scores_dist_dict)
    fig, axes = plt.subplots(num_models, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * num_models * 0.7), squeeze=False)
    eval_dfs = {}
    for i, (model_name, data) in enumerate(model_scores_dist_dict.items()):
        scores = data['scores']; lower_is_anomaly = data.get('lower_is_anomaly', False)
        if scores is None: continue
        current_eval_df = pd.DataFrame({'True_Class': y_true, 'Score': scores}); eval_dfs[model_name] = current_eval_df
        ax = axes[i, 0]
        sns.histplot(data=current_eval_df, x='Score', hue='True_Class', bins=50, kde=True, ax=ax, palette={0: 'blue', 1: 'red'}, element='step', stat="density")
        ax.set_title(f'{model_name} Anomaly Score Distribution')
        score_label = 'Anomaly Score' + (' (lower = more anomalous)' if lower_is_anomaly else ' (higher = more anomalous)')
        
        contamination_for_viz = default_contamination_proxy
        if 'Autoencoder' in model_name: # Use specific AE threshold percentile for viz
            contamination_for_viz = 1 - (ae_threshold_percentile_proxy / 100.0)

        # Calculate quantile based on whether lower or higher scores indicate anomaly
        if lower_is_anomaly: # e.g. original IF scores, not used here as scores are flipped
            quantile_val = np.percentile(scores, contamination_for_viz * 100)
            quantile_label_text = f'{contamination_for_viz*100:.1f}%ile'
        else: # Higher score is anomaly (current setup for IF, LOF, AE errors)
            quantile_val = np.percentile(scores, (1 - contamination_for_viz) * 100)
            quantile_label_text = f'{(1-contamination_for_viz)*100:.1f}%ile'
        
        ax.set_xlabel(score_label)
        line = ax.axvline(quantile_val, color='green', linestyle='--', label=f'{quantile_label_text}: {quantile_val:.4f}')
        
        # Improved legend handling
        handles, labels = ax.get_legend_handles_labels()
        # Check if quantile_label_text is already in labels (e.g. if cell re-run)
        if not any(quantile_label_text in lab for lab in labels):
            custom_labels = []
            custom_handles = []
            # Keep original hue legend entries
            for h, l in zip(handles, labels):
                if "Score" not in l and quantile_label_text not in l : # Avoid adding score distribution itself or duplicate quantile line
                    custom_handles.append(h)
                    if l == '0': custom_labels.append('Normal')
                    elif l == '1': custom_labels.append('Fraud')
                    else: custom_labels.append(l)
            # Add the quantile line
            custom_handles.append(line)
            custom_labels.append(line.get_label())
            ax.legend(custom_handles, custom_labels)
        else: # If already customized, just call legend()
            ax.legend()

    plt.tight_layout(); return plt, eval_dfs

def create_comparison_report(metrics_dict):
    report_data = []
    for model_name, metrics in metrics_dict.items():
        report_data.append({'Model': model_name, 'Accuracy': metrics.get('accuracy', np.nan),
                            'ROC-AUC': metrics.get('roc_auc', np.nan), 'AUPRC': metrics.get('auprc', np.nan)})
    return pd.DataFrame(report_data).set_index('Model')