#Applying LIME to a healthcare dataset to determine the contribution of each feature in the given instance
#This allows us to understand the reason behind each prediction made by the blackbox model

#Tools Used:
#pandas for loading the data
#RandomForestClassifier as the blacbox ML model
#LimeTabular for the local use
#Matplotlib for visualisation

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from interpret.blackbox import LimeTabular
from interpret import show

import matplotlib.pyplot as plt

data=pd.read_csv("healthcare-dataset-stroke-data.csv")

data["bmi"]=data["bmi"].fillna(data["bmi"].median())

X=data[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]]
y=data["stroke"]

X=pd.get_dummies(X,drop_first=True,dtype=int)

assert X.isnull().sum().sum() == 0
X = X.astype(float)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#for oversampling
smote=SMOTE(random_state=42)
X_train,y_train=smote.fit_resample(X_train,y_train)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

print(f"F1 Score: {f1_score(y_test,y_pred,average='macro')}")
print(f"Accuracy{accuracy_score(y_test,y_pred)}")

#initialize lime
lime=LimeTabular(model=rf,data=X_train,random_state=1)   

#get local explanation for a specific test entity(20th from the end)
lime_local=lime.explain_local(X_test[-20:-19],y_test[-20:-19],name='LIME')

explanation = lime_local.data(0)

for feature, weight in zip(explanation["names"], explanation["scores"]):
    print(f"{feature}: {weight:.4f}")

exp=lime_local.data(0)
features=exp["names"]
weights=exp["scores"]

plt.figure(figsize=(8, 5))
plt.barh(features, weights)
plt.axvline(0)
plt.title("LIME Feature Contributions")
plt.xlabel("Contribution to Prediction")
plt.ylabel("Features")
plt.show()