"""
Random Forest Classifier on 500hits Dataset

This script demonstrates how to train, evaluate, and improve a Random Forest model 
using a dataset of 500 hits. 

Key steps:
1. Data preprocessing: drop irrelevant columns, separate features (X) and target (y)
2. Train/test split: 80/20 split to evaluate generalization
3. Model training: 
   - First with default hyperparameters
   - Then with tuned hyperparameters (n_estimators, max_depth, min_samples_split, criterion)
4. Evaluation: 
   - Accuracy score
   - Precision, recall, F1-score for each class
   - Feature importance
5. Observations: Hyperparameter tuning improves accuracy and model performance.

This notebook illustrates the impact of hyperparameter tuning on Random Forest performance 
and prepares the workflow for explainability or further feature analysis.
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df=pd.read_csv("500hits.csv",encoding="latin-1")


df=df.drop(columns=['PLAYER','CS'])  #dropping columns that won't impact the decision

#features
X=df.iloc[:,0:13]    #taking all rows and 0-12 columns

#target
y=df.iloc[:,13]   #all rows and the 13th column


#X= input variables/features ----->what the model is allowed to look at; what we use to predict something
#y=target/label -------->what we want the model to predict


#training
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=17)
rf=RandomForestClassifier()
rf.fit(X_train,y_train)

#prediction
y_pred=rf.predict(X_test)
print(rf.score(X_test,y_test))    #score() returns the model’s default performance metric on the given data.----->it returns the accuracy; accuracy = (number of correct predictions) / (total predictions)



print(classification_report(y_test,y_pred))

features=pd.DataFrame(rf.feature_importances_,index=X.columns)
print(features)

#hyper parameters
#seeing if adding hyperparameters increases accuracy
#n_estimators->no of trees in forest ; criterion='entropy'->how each tree decides where to split
#min_sample_split->minm no of samples required to split a node ; max_depth->maxm depth of tree
rf2=RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    min_samples_split=10,
    max_depth=14,
    random_state=42
)

rf2.fit(X_train,y_train)
y_pred2=rf2.predict(X_test)
print(rf2.score(X_test,y_test))

print(classification_report(y_test,y_pred2))   #tuning hyperparameters has given us a much better result

