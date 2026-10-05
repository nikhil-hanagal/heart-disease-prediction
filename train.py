import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression 
import joblib
import os
from sklearn.ensemble import RandomForestClassifier 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,f1_score,precision_score,classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/heart.csv")

# print(df.head)
# print(df.info)
# print(df.shape)
# print(df.describe)

# print(df.isnull().sum())
# print(df["restecg"].value_counts())
# # print(df["dataset"].value_counts())
# # print(df.dtypes)
# print(df.head())

x=df.drop(columns=["id","dataset","num"])
y=(df["num"]>0).astype(int)
print(y.head())


#  train and testing the model on th basis of  data set 
x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=0.2 ,random_state=42,)
print("training data ",x_train.shape)
print("testing data ", x_test.shape)



# finding the numerical values 
numerical_features=["age","trestbps","chol","thalch","oldpeak","ca"]

# finding the categorical values
categorical_features=["sex","cp","fbs","restecg","exang","slope","thal"]

# numerical_preprocessor [ scaling the numerical_features]
numerical_transformer=Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scalar", StandardScaler())

])

# categorical_preprocessor [converting  categorical  data in to a numerical data ]
categorical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("cat",OneHotEncoder(handle_unknown="ignore"))
])

# combining the  2 transformer in to one transformer 
preprocessor=ColumnTransformer(
    transformers=[
        ("num", numerical_transformer,numerical_features),
        ("cat",categorical_transformer,categorical_features)
    ]
)

# printing the result 
print("the preprocessor data is completed ")

# training and evaluating the model[logistic regression ] 
logistic_model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
])

# random forest 
random_model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(n_estimators=1000,random_state=42))
])

#  xgboost 
xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ))
])

# training with the logistic regression  

logistic_model.fit(x_train,y_train)
y_pred=logistic_model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("  logistic accuracy are:",accuracy)

recall=recall_score(y_test,y_pred)
print("\n logistic  recall score",recall)

f1score=f1_score(y_test,y_pred)
print("\n logistic f2_Score are:",f1score)

precision=precision_score(y_test,y_pred)
print("\n logistic precision score are:", precision)

confusion=confusion_matrix(y_test,y_pred)
print("\n logistic confusion matrix are:",y_test,y_pred)

# training the random forest classifier 
random_model.fit(x_train,y_train)
y_pred_rm=random_model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred_rm)
print("  random accuracy are:",accuracy)

recall=recall_score(y_test,y_pred_rm)
print("\n random  recall score",recall)

f1score=f1_score(y_test,y_pred_rm)
print("\n random f2_Score are:",f1score)

precision=precision_score(y_test,y_pred_rm)
print("\n random precision score are:", precision)

confusion=confusion_matrix(y_test,y_pred_rm)
print("\n random confusion matrix are:",y_test,y_pred)

# train the xgboost classifier 
xgb_model.fit(x_train,y_train)
y_pred_xg=xgb_model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred_xg)
print(" xgb ",accuracy)

recall=recall_score(y_test,y_pred_xg)
print("\n xgb recall score",recall)

f1score=f1_score(y_test,y_pred_xg)
print("\n xgb  f2_Score are:",f1score)

precision=precision_score(y_test,y_pred_xg)
print("\n xgb  precision score are:", precision)

confusion=confusion_matrix(y_test,y_pred_xg)
print("\n xgb   confusion matrix are:",y_test,y_pred)


#  to perform the hyperparameter with grid search cv 

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("\nBest F1 score:", grid_search.best_score_)
print("\nBest estimator:", grid_search.best_estimator_)


# print the best predicted model fo y_pred_best 

best_xgB_model=grid_search.best_estimator_

y_pred_best=best_xgB_model.predict(x_test)

# evaluation are
print("\n_______the best result are:_______\n")

print("accuracy are:",accuracy_score(y_test,y_pred_best))
print("recall score are:",recall_score(y_test,y_pred_best))
print("the f1 score are :",f1_score(y_test,y_pred_best))
print("the precision  score are:",precision_score(y_test,y_pred_best))

print("\nthe confusion matrix are:")
print(confusion_matrix(y_test,y_pred_best))

print("\n the classification report are: ")
print(classification_report(y_test,y_pred_best))

# saving the model is a folder or file using the joblib 
os.makedirs("models",exist_ok=True)
joblib.dump(best_xgB_model,"models/model.pkl")
print("\n the xgboost save successfully")

