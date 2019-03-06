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
from sklearn.linear_model import Lasso

# Load the dataset.
train = pd.read_csv('training_FE_data.csv', index_col=0)
submission_set = pd.read_csv('test_FE_data.csv', index_col=0)

X = train.loc[:, train.columns != 'SalePrice']
y = train.loc[:, 'SalePrice']


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
lassoreg = Lasso(normalize=True, max_iter=1e5)


scores = cross_val_score(lassoreg, X, y, cv=4)
print(scores)

lassoreg.fit(X, y)
result = lassoreg.predict(submission_set)
submission_set['SalePrice'] = result
submission = submission_set[['SalePrice']]
submission.to_csv('FirstLassoRegression.csv')