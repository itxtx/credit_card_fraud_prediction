
# Credit Card Fraud Detection Analysis Report

This report summarizes the findings from analyzing a credit card transaction dataset to detect fraudulent activities. The primary focus of the provided analysis was on supervised learning using Logistic Regression, including experiments with various data resampling techniques.

# 1. Supervised Learning

## 1.1. Logistic Regression

Logistic Regression was employed as the primary supervised learning algorithm for fraud detection.

**Baseline Performance (Unscaled Data, No Resampling):**

* Accuracy: 0.9988
* Precision: 0.6786
* Recall: 0.5816
* F1-score: 0.6264
* AUPRC: 0.4837 (Note: AUPRC calculation can vary, another calculation yielded 0.6305)
* AUROC: 0.8729
* Confusion Matrix:
    ```
    [[56837    27]
     [   41    57]]
    ```

**Handling Imbalance:**

Several techniques were tested to address the significant class imbalance (0.17% fraud):

* **Random Oversampling (Minority):** Led to high recall (0.9286) but extremely low precision (0.0370) and F1-score (0.0712), indicating it flagged too many non-fraudulent transactions.
* **Random Oversampling (Targeted Ratio 0.002):** Showed slight improvement over baseline but wasn't optimal (F1: 0.5934).
* **Random Undersampling (Majority):** Similar to oversampling, resulted in high recall (0.9286) but poor precision (0.0367) and F1 (0.0706).
* **SMOTE:** Achieved high recall (0.8878) but lower precision (0.0852) and F1 (0.1555).

The initial conclusion was that basic resampling techniques with default Logistic Regression did not yield satisfactory improvements over the baseline on unscaled data.

**Hyperparameter Tuning and Scaling (RandomizedSearchCV):**

A `RandomizedSearchCV` approach was used with a pipeline that included `StandardScaler` and tested various resampling methods (SMOTE, RandomOverSampler, RandomUnderSampler, ADASYN, TomekLinks, NearMiss, None) along with Logistic Regression hyperparameters (C, penalty, solver).

* **Best F1-Score during CV:** The search, optimizing for F1-score, identified a configuration with `StandardScaler`, **no resampling**, and `LogisticRegression(solver='liblinear', penalty='l2', C=0.1)` as the best based *on cross-validation F1-score* (CV F1 ≈ 0.7208). *However, subsequent testing revealed slightly different optimal parameters when evaluating on the hold-out test set.*

**Final Selected Model & Performance:**

Based on detailed evaluation on the test set, comparing configurations derived from the hyperparameter search, the best performing model was identified as:

* **Preprocessing:** `StandardScaler`
* **Resampling:** None
* **Model:** `LogisticRegression(solver='liblinear', penalty='l1', C=1.0, random_state=42, max_iter=1000)`

* **Test Set Performance Metrics:**
    * Accuracy: 0.9991
    * Precision: 0.8636
    * Recall: 0.5816
    * F1-score: 0.6951
    * AUPRC: 0.7573
    * AUROC: 0.9750
    * Recall at 0.5% FPR: 0.8776
    * Precision at 0.5% Recall: 1.0000
    * Matthews Correlation Coefficient (MCC): 0.7084
    * Kolmogorov-Smirnov (KS) Statistic: 0.9090
    * Confusion Matrix:
        ```
        [[56855     9]
         [   41    57]]
        ```

**Conclusion for Logistic Regression:** Scaling the data using `StandardScaler` was crucial. The best results were achieved *without* explicit resampling techniques in the final model, using L1 regularization (penalty='l1') and C=1.0. This model demonstrated a good balance between precision and recall compared to the baseline and resampling experiments, significantly improving the F1-score and AUROC. The Precision-Recall and ROC curves visually confirmed the model's ability to distinguish between classes better than the baseline.

# 2. Unsupervised Learning

While mentioned in the initial overview of the notebook, the following unsupervised anomaly detection methods were **not implemented or evaluated** in the provided analysis.

## 2.1. Isolation Forest

* No analysis performed in the provided notebook.

## 2.2. Local Outlier Factor (LOF)

* No analysis performed in the provided notebook.