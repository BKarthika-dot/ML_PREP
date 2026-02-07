#Using SHAP to obtain both local and global explainability for the predictions made by a RandomForestClassifier(blackbox model)
#This allows us to understand the reason behind each prediction for each instance AND identify the most important contributors in the dataset as a whole 
#Tools used:

#pandas for loading dataset
#scikit learn
#RandomForestClassifier as the blackbox model
#Shap for global and local explainability

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap

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

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

print(f"F1 Score: {f1_score(y_test,y_pred,average='macro')}")
print(f"Accuracy:{accuracy_score(y_test,y_pred)}")

#create shap explainer
explainer=shap.TreeExplainer(rf)

#calculating shapley values for test data

i=0 #choose instance
shap_values=explainer.shap_values(X_test)


#visualising local predictors
shap.initjs()
#force plot
prediction=rf.predict(X_test.iloc[[i]])[0]
print(f"The RF predicted: {prediction}")


shap.force_plot(explainer.expected_value[1], #base value for class 1
                shap_values[i, :, 1],        #shap values for instance i class 1  ( # instance i, class = stroke)
                X_test.iloc[i],              #feature values for instance i
                feature_names=X_test.columns,
                matplotlib=True)


#beeswarm plot for global explainability
shap.summary_plot(
    shap_values[:, :, 1],  # all samples, all features, class 1
    X_test,
    feature_names=X_test.columns,
)


shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[i, :, 1],                 # SHAP values for instance i, class 1
        base_values=explainer.expected_value[1],     # baseline risk of stroke
        data=X_test.iloc[i],                          # feature values
        feature_names=X_test.columns
    )
)