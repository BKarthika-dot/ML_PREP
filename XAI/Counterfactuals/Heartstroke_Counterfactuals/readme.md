
---

# 🧠 Stroke Prediction with Counterfactual Explanations

This project builds a machine learning model to predict stroke risk and uses **counterfactual explanations (DiCE)** to interpret model decisions.

## ⚙️ What I Did

* Built an end-to-end **scikit-learn pipeline**

  * OneHotEncoding (categorical features)
  * SMOTE (class imbalance handling)
  * Random Forest classifier
* Evaluated using **Macro F1 score** due to imbalanced data
* Generated **counterfactual explanations** using DiCE

---

## 🔍 What Are Counterfactuals?

Counterfactual explanations answer:

> *“What minimal changes to the input would flip the model’s prediction?”*

Important distinction:

* Counterfactuals flip the **model prediction**
* Not necessarily the true label

---

## 🎯 Actionable Explanations

I restricted certain features (e.g., age cannot change) to generate **realistic and actionable counterfactuals**.

In some cases, no counterfactuals were found under constraints, showing:

* The model relies heavily on non-actionable risk factors
* Not all predictions are easily reversible

---

## 🧠 Key Learning

This project demonstrates:

* Understanding of model interpretability (XAI)
* Practical use of counterfactual reasoning
* Limitations of tree-based models for explanation
* Difference between prediction probability and class labels

---

## 🚀 Tech Stack

* Python
* scikit-learn
* imbalanced-learn (SMOTE)
* DiCE (Counterfactual Explanations)

---


