import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
from sklearn.model_selection import train_test_split
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import skew
from collections import OrderedDict
from sklearn.model_selection import cross_val_score

# Load the dataset.
train = pd.read_csv('training_FE_data.csv', index_col=0)
submission_set = pd.read_csv('test_FE_data.csv', index_col=0)

X = train.loc[:, train.columns != 'SalePrice']
y = train.loc[:, 'SalePrice']


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)


scores = cross_val_score(best_xgb_model, X, y, cv=4)
print(scores)

best_xgb_model.fit(X, y)
result = best_xgb_model.predict(submission_set)
submission_set['SalePrice'] = result
submission = submission_set[['SalePrice']]
submission.to_csv('FirstXGBoostSubmission.csv')