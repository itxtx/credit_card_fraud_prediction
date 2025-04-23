from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_curve, roc_curve, auc, confusion_matrix,
        matthews_corrcoef
    )
import numpy as np




def calculate_model_metrics(model, X_test, y_test, model_name):
    """
    Calculate comprehensive metrics for a model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to evaluate
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str
        Name of the model (for printing)
        
    Returns:
    --------
    dict
        Dictionary containing all calculated metrics
    """

    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get scores (handle different model types)
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'score_samples'):
        y_scores = -model.score_samples(X_test)  # For IsolationForest
    else:
        y_scores = y_pred
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Precision-Recall and ROC curves
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auprc = auc(recall_curve, precision_curve)
    auroc = auc(fpr, tpr)
    
    # Advanced metrics
    cm = confusion_matrix(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Custom metrics
    def recall_at_k_fpr(y_true, y_scores, k=0.005):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.argmin(np.abs(fpr - k))
        return tpr[idx]
    
    def precision_at_k_recall(y_true, y_scores, k=0.005):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        idx = np.argmin(np.abs(recall - k))
        return precision[idx]
    
    def ks_statistic(y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        return np.max(np.abs(fpr - tpr))
    
    recall_at_k = recall_at_k_fpr(y_test, y_scores, k=0.005)
    precision_at_k = precision_at_k_recall(y_test, y_scores, k=0.005)
    ks = ks_statistic(y_test, y_scores)
    
    # Print results
    print(f"\nMetrics for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Recall at 0.5% FPR: {recall_at_k:.4f}")
    print(f"Precision at 0.5% Recall: {precision_at_k:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Kolmogorov-Smirnov Statistic: {ks:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Return all metrics in a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auprc': auprc,
        'auroc': auroc,
        'recall_at_k': recall_at_k,
        'precision_at_k': precision_at_k,
        'mcc': mcc,
        'ks': ks,
        'confusion_matrix': cm,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'fpr': fpr,
        'tpr': tpr
    }