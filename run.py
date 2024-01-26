# filename: my_script.py
#pip install scikit-learn==1.2.2
import pandas as pd
import sys
# pip install catboost xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import pickle
# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python code.py test_data.csv")
    sys.exit(1)

# Retrieve arguments
arg1 = sys.argv[1]

# Your code here using arg1 and arg2
print(f"File Name: {arg1}")


df_train = pd.read_csv(arg1)
df_train = df_train.drop('datasetId', axis=1)
uuidcol = df_train["uuid"]
df_train = df_train.drop('uuid', axis=1)
X = df_train.iloc[:, :]

EX = pd.get_dummies(X, columns=['condition'], prefix='condition')
xgb_model = joblib.load('models/xgb_best_model.pkl')
catboost_model = joblib.load('models/catboost_best_model.pkl')
et_regressor = joblib.load('models/extratrees_best_model.pkl')


# Make predictions on the test set using each model
catboost_pred = catboost_model.predict(EX)
xgb_pred = xgb_model.predict(EX)
et_pred = et_regressor.predict(EX)

# Ensemble predictions by averaging
ensemble_pred = (0.562*catboost_pred + 0.315*xgb_pred + 0.939*et_pred) / 1.816

result_df = pd.DataFrame({'uuid': uuidcol, 'HR': ensemble_pred})
print(result_df)
result_df.to_csv('results.csv', index=False)
