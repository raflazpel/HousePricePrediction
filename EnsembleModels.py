import pandas as pd
import numpy as np

ridge = pd.read_csv('FirstRidgeRegression.csv', index_col=0)
xgboost = pd.read_csv('FirstXGBoostSubmission.csv', index_col=0)

result = (ridge['SalePrice']+xgboost['SalePrice'])/2
result.to_csv('FirstEnsembleRegression.csv')