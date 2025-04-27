# Credit Card Fraud Detection Analysis Report

This report summarizes the findings from analyzing a credit card transaction dataset to detect fraudulent activities. The analysis covers both supervised learning using Logistic Regression and XGBoost, as well as unsupervised anomaly detection techniques.

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

## 1.2. XGBoost

XGBoost (Extreme Gradient Boosting) was evaluated as a more complex supervised learning model.

**Preprocessing and Imbalance Handling:**

* Data was scaled using `StandardScaler`.
* A `Pipeline` was used to integrate scaling, optional resampling, and the XGBoost classifier.
* Both explicit resampling techniques (SMOTE, ADASYN, Over/Under-sampling, etc.) and XGBoost's internal `scale_pos_weight` parameter (calculated as `neg_count / pos_count ≈ 577.29`) were explored to handle class imbalance during hyperparameter tuning.

**Hyperparameter Tuning (RandomizedSearchCV):**

* A broad `RandomizedSearchCV` (75 iterations, 5-fold CV) was performed, optimizing for F1-score. It explored various XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`) combined with different resampling strategies or `scale_pos_weight`.
* **Note:** The search process was interrupted (`KeyboardInterrupt`). However, the `best_estimator_` found *before* the interruption was evaluated.
* **Best Model Found (Pre-Interruption) & Performance:**
    * The best configuration identified involved **no resampling** and `scale_pos_weight=1` (i.e., not using the calculated weight). Specific XGBoost parameters were determined by the search but not explicitly printed before interruption.
    * **Test Set Performance Metrics:**
        * Accuracy: 0.9996
        * Precision: 0.9750
        * Recall: 0.7959
        * F1-score: 0.8764
        * AUPRC: 0.8927
        * AUROC: 0.9851
        * Recall at 0.5% FPR: 0.9082
        * Precision at 0.5% Recall: 1.0000
        * Matthews Correlation Coefficient (MCC): 0.8807
        * Kolmogorov-Smirnov (KS) Statistic: 0.9273
        * Confusion Matrix:
            ```
            [[56862     2]
             [   20    78]]
            ```

**Hyperparameter Tuning (BayesSearchCV):**

* To refine the parameters, a `BayesSearchCV` (50 iterations, 5-fold CV) was conducted, focusing the search space around the promising results from the RandomizedSearch (specifically, `resampler=None`, `scale_pos_weight=1`).
* **Best Model Found & Parameters:**
    * **Preprocessing:** `StandardScaler`
    * **Resampling:** None
    * **Model:** `XGBClassifier` with parameters:
        ```python
        {
            'classifier__colsample_bytree': 0.939866032482152,
            'classifier__gamma': 0.43802918644092337,
            'classifier__learning_rate': 0.041624924398728134,
            'classifier__max_depth': 9,
            'classifier__n_estimators': 471,
            'classifier__reg_alpha': 0.08614807334958828,
            'classifier__reg_lambda': 0.35263559617083484,
            'classifier__scale_pos_weight': 1,
            'classifier__subsample': 0.8059251536583913,
            'resampler': None # Explicitly shown for clarity
        }
        ```
* **Test Set Performance Metrics:**
    * Accuracy: 0.9996
    * Precision: 0.9875
    * Recall: 0.8061
    * F1-score: 0.8876
    * AUPRC: 0.8890
    * AUROC: 0.9860
    * Recall at 0.5% FPR: 0.8980
    * Precision at 0.5% Recall: 1.0000
    * Matthews Correlation Coefficient (MCC): 0.8921
    * Kolmogorov-Smirnov (KS) Statistic: 0.9199
    * Confusion Matrix:
        ```
        [[56863     1]
         [   19    79]]
        ```
* Precision-Recall and ROC curves were generated for this best model.

**Conclusion for XGBoost:** XGBoost significantly outperformed the optimized Logistic Regression model. The best configuration, found via Bayesian optimization after an initial randomized search, achieved a much higher F1-score (0.8876 vs 0.6951) and Recall (0.8061 vs 0.5816) while maintaining very high precision (0.9875). Interestingly, the best performance was achieved *without* explicit resampling or using the `scale_pos_weight` parameter, relying instead on the model's inherent capabilities and regularization found during tuning.

## 1.3. Random Forest

Random Forest was also evaluated as a supervised learning model.

**Preprocessing and Imbalance Handling:**

* Data was scaled using `StandardScaler`.
* A `Pipeline` was used to integrate scaling, optional resampling, and the RandomForestClassifier.
* Both explicit resampling techniques (SMOTE, ADASYN, Over/Under-sampling, etc.) and Random Forest's internal `class_weight` parameter (`balanced`, `balanced_subsample`) were explored during hyperparameter tuning.

**Hyperparameter Tuning (RandomizedSearchCV):**

* A `RandomizedSearchCV` (75 iterations, 5-fold CV) optimizing for F1-score explored various Random Forest hyperparameters (`n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`) combined with different resampling strategies or `class_weight` settings.
* **Best Model Found & Parameters:**
    * The best configuration involved **RandomOverSampler** with `sampling_strategy=0.7` and no internal class weighting (`class_weight=None`).
    * **Best Parameters:**
        ```python
        {
            'resampler__sampling_strategy': 0.7,
            'resampler': RandomOverSampler(random_state=42),
            'classifier__n_estimators': 200,
            'classifier__min_samples_split': 5,
            'classifier__min_samples_leaf': 3,
            'classifier__max_features': 'sqrt',
            'classifier__max_depth': 30,
            'classifier__class_weight': None,
            'classifier__bootstrap': True
        }
        ```
    * **Best CV F1 Score:** 0.8556
* **Test Set Performance Metrics (RandomizedSearch Best):**
    * Accuracy: 0.9996
    * Precision: 0.9405
    * Recall: 0.8061
    * F1-score: 0.8681
    * AUPRC: 0.8823
    * AUROC: 0.9719
    * Recall at 0.5% FPR: 0.8878
    * Precision at 0.5% Recall: 1.0000
    * Matthews Correlation Coefficient (MCC): 0.8705
    * Kolmogorov-Smirnov (KS) Statistic: 0.9091
    * Confusion Matrix:
        ```
        [[56859     5]
         [   19    79]]
        ```

**Hyperparameter Tuning (BayesSearchCV):**

* A focused `BayesSearchCV` (10 iterations, 5-fold CV) was conducted, refining parameters around the best RandomizedSearch results (fixing `resampler=RandomOverSampler(sampling_strategy=0.7)`, `class_weight=None`).
* **Best Model Found & Parameters:**
    * **Preprocessing:** `StandardScaler`
    * **Resampling:** `RandomOverSampler(sampling_strategy=0.7, random_state=42)`
    * **Model:** `RandomForestClassifier` with parameters:
        ```python
        {
            'classifier__bootstrap': True,
            'classifier__class_weight': None,
            'classifier__max_depth': 39,
            'classifier__max_features': 'sqrt',
            'classifier__min_samples_leaf': 3,
            'classifier__min_samples_split': 5,
            'classifier__n_estimators': 228,
            'resampler': RandomOverSampler(random_state=42, sampling_strategy=0.7) # Explicitly shown
        }
        ```
    * **Best CV F1 Score:** 0.8556 (Note: Same as RandomizedSearch best, suggesting convergence or limited search space benefit in 10 iterations)
* **Test Set Performance Metrics (BayesSearch Best):**
    * Accuracy: 0.9996
    * Precision: 0.9518
    * Recall: 0.8061
    * F1-score: 0.8729
    * AUPRC: 0.8834
    * AUROC: 0.9720
    * Recall at 0.5% FPR: 0.8878
    * Precision at 0.5% Recall: 1.0000
    * Matthews Correlation Coefficient (MCC): 0.8757
    * Kolmogorov-Smirnov (KS) Statistic: 0.9229
    * Confusion Matrix:
        ```
        [[56860     4]
         [   19    79]]
        ```
* Precision-Recall, ROC, and convergence plots were generated.

**Conclusion for Random Forest:** The optimized Random Forest model performed significantly better than Logistic Regression and achieved results comparable to XGBoost in terms of F1-score (0.8729 vs 0.8876) and Recall (0.8061 vs 0.8061), although with slightly lower precision (0.9518 vs 0.9875). Unlike XGBoost, the best Random Forest model benefited from explicit resampling (RandomOverSampler). However, Random Forest training, especially during hyperparameter search, was noted to be computationally more expensive than XGBoost.

# 2. Unsupervised Learning

Unsupervised methods were explored for anomaly detection without relying on predefined labels. Data was scaled using `StandardScaler` before applying these models.

## 2.1. Isolation Forest

Isolation Forest attempts to isolate anomalies by randomly partitioning the data.

**Initial Model Performance:**

A model was trained with default parameters and `contamination` set to the known fraud rate (0.001727).

* **Preprocessing:** `StandardScaler`
* **Model:** `IsolationForest(contamination=0.001727, random_state=42, n_estimators=100, max_samples='auto')`
* **Test Set Performance Metrics:**
    * Accuracy: 0.9977
    * Precision: 0.3300
    * Recall: 0.3367
    * F1-score: 0.3333
    * AUPRC: 0.3339 (Calculated from binary predictions)
    * Recall at 0.5% FPR: 0.5510
    * Precision at 0.5% Recall: 1.0000
    * Matthews Correlation Coefficient (MCC): 0.3322
    * Kolmogorov-Smirnov (KS) Statistic: 0.8172 (Calculated from anomaly scores)
    * Confusion Matrix:
        ```
        [[56797    67]
         [   65    33]]
        ```
* Precision-Recall and ROC curves were generated based on the anomaly scores.

**Hyperparameter Tuning Attempt:**

A `GridSearchCV` was attempted to optimize `n_estimators` and `max_samples`. However, the search failed due to an issue with the custom scoring function used, which did not receive the required `y_true` argument. Therefore, the results from this tuning process are unreliable. The evaluation performed *after* the GridSearch (using `best_if_model`) showed worse performance than the initial run (F1: 0.2647, MCC: 0.2636), likely due to the failed optimization.

**Conclusion for Isolation Forest:** The initial Isolation Forest model showed some capability in identifying anomalies but with significantly lower precision and recall compared to the optimized supervised models. The hyperparameter tuning attempt was unsuccessful due to technical issues.

## 2.2. Local Outlier Factor (LOF)

Local Outlier Factor measures the local density deviation of a data point with respect to its neighbors.

**Analysis Status:**

The analysis for LOF, including a `GridSearchCV` for hyperparameter tuning, was **interrupted** during execution (`KeyboardInterrupt`). As a result, the model could not be fully trained or evaluated, and no definitive results can be reported for LOF based on the provided execution logs. Subsequent code cells attempting to use the results of the LOF GridSearch or utility functions (`ccf`) did not have their outputs included in the provided information.

**Conclusion for LOF:** The LOF analysis was incomplete.
