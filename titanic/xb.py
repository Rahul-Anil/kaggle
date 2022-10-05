import pandas as pd
import numpy as np
import titanic_preprocessing as tp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


train_df = pd.read_csv(
    "/Users/rahulanil/garchomp/projects/kaggle/titanic/data/train.csv"
)
test_df = pd.read_csv("/Users/rahulanil/garchomp/projects/kaggle/titanic/data/test.csv")

# generic_perprocessing
test_df_passengeId = test_df["PassengerId"]
train_df, test_df = tp.generic_perprocessing(train_df, test_df)

X = train_df.loc[:, train_df.columns != "Survived"]
y = train_df["Survived"]

ct = ColumnTransformer(
    [("onehot", OneHotEncoder(sparse=False), ["Pclass", "Sex", "Embarked", "Initials"])]
)

ct.fit(X)
X_ct = ct.transform(X)
print(f"X_ct shape: {X_ct.shape}")

test = ct.transform(test_df)
print(f"X_test shape: {test.shape}")

param_grid = {
    "eta": [i for i in np.arange(0, 1, 0.1)],
    "max_depth": [i for i in range(2, 10)],
    "n_estimators": [i for i in range(10, 200, 10)],
    "gamma": [i for i in range(0, 10, 1)],
    "subsample": [i for i in np.arange(0, 1, 0.1)],
    "sampling_method": ["uniform", "subsampe", "gradient_based"],
    "lambda": [i for i in range(0, 10)],
    "alpha": [i for i in range(0, 10)],
}

grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,
    return_train_score=True,
    n_jobs=-1,
    verbose=10,
)

X_train, X_test, y_train, y_test = train_test_split(X_ct, y, random_state=0)

grid_search.fit(X_train, y_train)
print(f"test scores: {grid_search.score(X_test, y_test)}")
print(f"Best parametesrs: {grid_search.best_params_}")
print(f"best cross validation score: {grid_search.best_score_}")
