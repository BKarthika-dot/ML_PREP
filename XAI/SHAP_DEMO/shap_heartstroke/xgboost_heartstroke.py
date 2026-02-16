import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,precision_recall_curve,precision_score,recall_score,average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data=pd.read_csv("healthcare-dataset-stroke-data.csv")

data["bmi"]=data["bmi"].fillna(data["bmi"].median())

X=data[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]]
y=data["stroke"]

X=pd.get_dummies(X,drop_first=True,dtype=int)

X=X.astype(float)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


#oversampling-Increasing the number of samples in the minority class so the model doesn’t ignore it.
smote=SMOTE(random_state=42)
X_train,y_train=smote.fit_resample(X_train,y_train)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train,y_train)

#evaluating test data
y_pred=model.predict(X_test)
print(f"F1 Score: {f1_score(y_test,y_pred)}")
print(f"Accuracy:{accuracy_score(y_test,y_pred)}")

#creating shap explainer
explainer=shap.TreeExplainer(model)

#calculating shapley values for test data

i=0 #choose instance
shap_values=explainer.shap_values(X_test)


#visualising local predictors
shap.initjs()
#force plot
prediction=model.predict(X_test.iloc[[i]])[0]
print(f"The XGBOOST model predicted: {prediction}")

#plotting

xgb_base = explainer.expected_value
xgb_values = shap_values[i]



shap.force_plot(
    xgb_base,
    xgb_values,
    X_test.iloc[i],
    feature_names=X_test.columns,
    matplotlib=True
)

#beeswarm plot for global explainability
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X_test.columns,
)



shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[i],                 # SHAP values for instance i
        base_values=explainer.expected_value,  # baseline value (scalar)
        data=X_test.iloc[i],
        feature_names=X_test.columns
    )
)


# Predict probabilities
xgb_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot
plt.figure()
plt.plot(fpr_xgb, tpr_xgb)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"XGBoost ROC Curve (AUC = {auc_xgb:.4f})")
plt.show()

print("XGBoost AUC:", auc_xgb)


xgb_preds = model.predict(X_test)

xgb_precision = precision_score(y_test, xgb_preds)
xgb_recall = recall_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_pr_auc = average_precision_score(y_test, xgb_probs)

print("\nXGBoost Metrics")
print("Precision:", xgb_precision)
print("Recall:", xgb_recall)
print("F1:", xgb_f1)
print("PR AUC:", xgb_pr_auc)