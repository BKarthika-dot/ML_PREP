import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from imblearn.over_sampling import SMOTE
import dice_ml

#loading dataset
data=pd.read_csv("healthcare-dataset-stroke-data.csv")
data = data.drop("id", axis=1)

data["bmi"]=data["bmi"].fillna(data["bmi"].median())

X=data[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]]
y=data["stroke"]

#defining categorical and continuous features
categorical_features = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

continuous_features = ['age', 'avg_glucose_level', 'bmi']


#Building pipeline with th encoding inside (Build Preprocessing + Model Pipeline)

# ✔ Model trains normally
# ✔ Encoding happens automatically
# ✔ No manual dummy columns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', continuous_features)
    ]
)

#Train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipeline = ImbPipeline(steps=[
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Accuracy:", accuracy_score(y_test, y_pred))

#Setup DiCE

train_data = X_train.copy()
train_data["stroke"] = y_train

data_dice=dice_ml.Data(dataframe=train_data,continuous_features=continuous_features,categorical_features=categorical_features,outcome_name="stroke")

rf_dice=dice_ml.Model(model=pipeline,backend="sklearn")

explainer=dice_ml.Dice(data_dice,rf_dice,method="random")

#Generate Counterfactual
input_datapoint=X_test.iloc[[0]]

try:
    cf=explainer.generate_counterfactuals(input_datapoint,total_CFs=3,desired_class="opposite")

    original = input_datapoint.copy()
    cf_df = cf.cf_examples_list[0].final_cfs_df

    print("Original Instance:")
    print("Model prediction:", pipeline.predict(input_datapoint)[0])
    print("Model probability:", pipeline.predict_proba(input_datapoint)[0])
    print(original)

    print("\nCounterfactuals:")
    print(cf_df)
except Exception as e:
    print("No feasible counterfactuals.")

#specifying features to vary to get realistic , actionable outputs
features_to_vary=['smoking_status', 'avg_glucose_level', 'bmi',"age","hypertension"]

permitted_range={"avg_glucose_level":[50,250],"bmi":[18,35]}

#generating explanations with new feature weights
try:
    cf_new = explainer.generate_counterfactuals(
        input_datapoint,
        total_CFs=3,
        desired_class="opposite",
        permitted_range=permitted_range,
        features_to_vary=features_to_vary,
    )

    cf_new_df = cf_new.cf_examples_list[0].final_cfs_df
    print("\nCounterfactuals with predefined features:")
    print(cf_new_df)

except Exception as e:
    print("No feasible counterfactuals under given constraints.")
