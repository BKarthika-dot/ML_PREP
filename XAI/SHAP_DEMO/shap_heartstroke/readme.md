---

# 📌 Model Comparison: Random Forest vs XGBoost

**Task:** Stroke Risk Classification (Imbalanced Dataset)

## 🎯 Objective

To compare ensemble learning approaches (Bagging vs Boosting) for stroke prediction and analyze model behavior using SHAP explainability.

---

## 🧠 Models Used

* **Random Forest (Bagging-based ensemble)**
* **XGBoost (Gradient Boosted Trees)**

Both models were trained on the same dataset and evaluated on an identical test split.

---

## 📊 Performance Comparison

| Metric             | Random Forest | XGBoost   |
| ------------------ | ------------- | --------- |
| Accuracy           | 0.948         | 0.942     |
| ROC AUC            | 0.777         | **0.780** |
| Precision (Stroke) | **0.333**     | 0.263     |
| Recall (Stroke)    | 0.06          | **0.10**  |
| F1 Score (Stroke)  | 0.102         | **0.145** |
| PR AUC             | 0.147         | **0.152** |

---

## 🔎 Key Observations

### 1️⃣ Imbalanced Data Behavior

The dataset is highly imbalanced (stroke cases are rare).

* High accuracy (~94%) is largely driven by correct prediction of non-stroke cases.
* Recall for stroke cases remains low in both models at the default 0.5 threshold.

This highlights the importance of evaluating:

* Precision-Recall AUC
* Class-specific recall
  instead of relying solely on accuracy.

---

### 2️⃣ ROC vs Precision-Recall

Although both models achieved similar ROC AUC (~0.78),
the PR AUC (~0.15) reveals limited sensitivity to the minority class.

This demonstrates how ROC can appear optimistic in imbalanced settings.

---

## 🔬 SHAP Explainability Analysis

SHAP was used to analyze both:

* **Global feature importance**
* **Local (instance-level) explanations**

### Findings:

* Local explanations differed slightly between models for the same patient.
* XGBoost captured more nuanced feature interactions compared to Random Forest.

This demonstrates that:

> Model architecture influences how feature contributions are learned and interpreted.

---

## 🏁 Conclusion

* XGBoost showed slightly stronger discriminative ability (higher ROC AUC and PR AUC).
* Both models struggled with minority class recall at default thresholds.
* SHAP analysis revealed architectural differences in feature attribution.

This project demonstrates:

* Practical implementation of ensemble models
* Evaluation under class imbalance
* Model comparison using multiple metrics
* Interpretability using SHAP (force plots, waterfall plots, dependence plots)

---

