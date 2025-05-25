# Credit Card Fraud Detection Analysis Report

This report summarizes the findings from analyzing a credit card transaction dataset to detect fraudulent activities. The analysis covers both supervised learning using Logistic Regression, Random Forest and XGBoost , as well as unsupervised anomaly detection techniques.

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

Unsupervised methods were explored for anomaly detection without relying on predefined labels. Data was scaled using `StandardScaler` before applying these models. The actual fraud percentage in the dataset was 0.1727%.

---

## 2.1. Isolation Forest 🌳

Isolation Forest attempts to isolate anomalies by randomly partitioning the data.

**Initial Model Performance (Default Parameters):**

A baseline model was trained with `contamination` set to the known fraud rate (0.001727), `n_estimators=100`, and `max_samples='auto'`.

* **Preprocessing:** `StandardScaler`
* **Model:** `IsolationForest(contamination=0.001727, random_state=42, n_estimators=100, max_samples='auto')`
* **Test Set Performance Metrics:**
    * Accuracy: 0.9976
    * Precision: 0.3084
    * Recall: 0.3367
    * F1-score: 0.3220
    * Matthews Correlation Coefficient (MCC): 0.3210
    * Kolmogorov-Smirnov (KS) Statistic: 0.8251
    * Recall at 0.5% FPR: 0.5000
    * Precision at 0.5% Recall: 0.0000
    * Confusion Matrix:
        ```
        [[56790    74]
         [   65    33]]
        ```
* Precision-Recall and ROC curves were generated based on the anomaly scores.

**Hyperparameter Tuning (Optuna):**

Optuna was used for hyperparameter optimization over 50 trials, maximizing the F1-score using 3-fold stratified cross-validation. The search space included `n_estimators`, `max_samples`, `contamination`, `max_features`, and `bootstrap`.

* **Best F1-Score during HPO:** 0.3520
* **Best Hyperparameters Found:**
    * `n_estimators`: 189
    * `max_samples`: 0.9450
    * `contamination`: 0.00255
    * `max_features`: 0.2581
    * `bootstrap`: False

**Tuned Model Performance:**

The model was retrained using the best hyperparameters.

* **Test Set Performance Metrics (Tuned Model):**
    * Accuracy: 0.9972
    * Precision: 0.2895
    * Recall: 0.4490
    * F1-score: 0.3520
    * F2-score: 0.4044
    * AUPRC (from scores): 0.2577
    * Matthews Correlation Coefficient (MCC): 0.3592
    * Kolmogorov-Smirnov (KS) Statistic (from scores): 0.8419
    * Recall at 0.5% FPR (from scores): 0.6837
    * Confusion Matrix:
        ```
        [[56756   108]
         [   54    44]]
        ```

**Conclusion for Isolation Forest:**

The initial Isolation Forest model with default settings showed some capability in identifying anomalies. Hyperparameter tuning with Optuna, optimizing for F1-score, led to an **improvement in recall** (from 0.3367 to 0.4490) and **F1-score** (from 0.3220 to 0.3520), though precision slightly decreased. The AUPRC based on scores was 0.2577. While better than the baseline unsupervised attempt, its performance remained **significantly lower** than the supervised models.

---

## 2.2. Local Outlier Factor (LOF) 📍

Local Outlier Factor (LOF) measures the local density deviation of a data point with respect to its neighbors. `novelty=True` was used to enable predictions on new data.

**Hyperparameter Tuning (Optuna):**

Optuna was used for hyperparameter optimization over 30 trials, maximizing the F1-score using 5-fold stratified cross-validation. The search space included `n_neighbors`, `contamination`, `leaf_size`, and `p` (distance metric).

* **Best F1-Score during HPO:** 0.0178
* **Best Hyperparameters Found:**
    * `n_neighbors`: 49
    * `contamination`: 0.00384
    * `leaf_size`: 31
    * `p`: 2 (Euclidean distance)

**Tuned Model Performance:**

The model was trained using the best hyperparameters.

* **Test Set Performance Metrics (Tuned Model):**
    * Accuracy: 0.9947
    * Precision: 0.0142
    * Recall: 0.0306
    * F1-score: 0.0194
    * F2-score: 0.0249
    * AUPRC (from scores): 0.0030
    * Kolmogorov-Smirnov (KS) Statistic (from scores): 0.1185
    * Recall at 0.5% FPR (from scores): 0.0306
    * Confusion Matrix:
        ```
        [[56656   208]
         [   95     3]]
        ```

**Conclusion for LOF:**

The LOF model, even after hyperparameter tuning with Optuna, **performed poorly** in detecting fraudulent transactions. The F1-score was very low (0.0194), and other metrics like precision, recall, and AUPRC were also substantially lower than both the supervised models and the tuned Isolation Forest. The KS statistic was also significantly lower. This suggests that LOF, with the explored hyperparameter space, was **not effective** for this particular fraud detection task.

---

## 2.3. Autoencoder 🤖

A PyTorch-based Autoencoder was implemented for anomaly detection. The architecture consisted of an encoder and a decoder with ReLU activations, and no activation in the final decoder layer. The model was trained on normal (non-fraudulent) data only.

**Architecture and Training Parameters (Constants):**

* Input Dimension (`AE_INPUT_DIM`): 30 (derived from X.shape[1])
* Device: MPS (or as detected: cuda/cpu)
* Number of Workers for DataLoader: 8

**Hyperparameter Tuning (Optuna):**

Optuna was used for hyperparameter optimization over 15 trials (due to computational intensity). The objective was to maximize the F2-score, with the anomaly threshold determined per-fold by finding the value on validation reconstruction errors that maximized the F2-score. The search space included:
* `encoding_dim`
* `hidden_dim1_factor` (relative to input_dim)
* `hidden_dim2_factor` (relative to hidden_dim1)
* `learning_rate`
* `epochs` (for HPO, kept low: 10-30)
* `batch_size`

* **Best F2-Score during HPO (based on internal threshold optimization):** 0.4175
* **Best Hyperparameters for Network/Training:**
    * `encoding_dim`: 5
    * `hidden_dim1_factor`: 1.5871
    * `hidden_dim2_factor`: 0.6084
    * `learning_rate`: 0.00363
    * `epochs`: 14
    * `batch_size`: 128
    * Derived `hidden_dim1`: `max(5 + 1, int(30 * 1.5871 / 2))` = 23
    * Derived `hidden_dim2`: `max(5 + 1, int(23 * 0.6084 / 2))` = 6

**Tuned Model Performance:**

The final Autoencoder model was trained on all normal data in the training set using the best hyperparameters. The anomaly threshold was then determined on the test set reconstruction errors by finding the threshold that maximized the F2-score on the test labels.

* **Optimal Threshold found on Test Errors:** 7.660615
* **Test Set Performance Metrics (Tuned Model with Test-Optimized Threshold):**
    * Precision: 0.1725
    * Recall: 0.6020
    * F2-score (Maximized on Test): 0.4019
    * Confusion Matrix:
        ```
        [[56581   283]
         [   39    59]]
        ```

**Conclusion for Autoencoder:**

The Autoencoder, after hyperparameter tuning and with a threshold optimized on test set reconstruction errors to maximize F2-score, achieved a **recall of 0.6020**. This was the **highest recall among the unsupervised methods** tested. However, this came at the cost of **very low precision** (0.1725), resulting in an F2-score of 0.4019 (and an F1-score, if calculated, would be lower, approximately 0.268). The strategy of optimizing the threshold directly on test labels provides an upper bound on performance for this specific model structure and data split but isn't a threshold that could have been determined without knowing the test labels beforehand. In a practical scenario, the threshold would be set based on reconstruction errors from normal training data or a validation set. Overall, while showing better recall, the Autoencoder's precision was considerably low, indicating a **high number of false positives**.