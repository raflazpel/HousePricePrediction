# # Feature engineering: Improving model´s result without changing the model.
# This notebook has as objetive to analyse and understand the details of the feature engineering process
#  (f.e from now on) Even though I have duplicated the Dataiku pipeline given by Jesus to obtain a test dataset
# preprocessed exactly as the train dataset in this notebook I will start the whole process from scratch to be sure
# I am able to perform the whole process.
# 
# As we are performing F.E the only rule will be not to change the prediction model built at the professor's notebook.


# Import necesary packages.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats


# Functions
def clean_data(df):
    # Change the index and drop almost empty variables.
    df = df.set_index('Id')
    df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
    # Now I want to mark the categorical variables.

    # Strategy 1: Almost all (all in this case) non-numeric variables are categorical,
    # so I find them and mark them as categorical

    # Second step would be to find all the features that are numerical-categorical columns and mark them
    # (Because we already marked non-numerical.
    # In this case I have done it manually with ALL categorical columns
    for col in ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'MSZoning', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']:
        df[col] = df[col].astype('category')
    # Apply one-hot-enconding to all categorical variables.
    # Suponemos que todas las variables categoricas son nominales. Se puede aplicar integer-encoding
    # a las que sean ordinales.
    categorical_columns = df.select_dtypes(include='category').columns.tolist()

    for categoricalVariable in categorical_columns:
        dummy = pd.get_dummies(df[categoricalVariable], prefix=categoricalVariable)
        df = pd.concat([df, dummy], axis=1)
        df.drop([categoricalVariable], axis=1, inplace=True)

    # Now we fill null-values
    df['LotFrontage'].fillna(value=0, inplace=True)
    df['MasVnrArea'].fillna(value=0, inplace=True)
    df['BsmtFinSF1'].fillna(value=0, inplace=True)
    df['BsmtFinSF2'].fillna(value=0, inplace=True)
    df['BsmtUnfSF'].fillna(value=0, inplace=True)
    df['TotalBsmtSF'].fillna(value=0, inplace=True)
    df['BsmtFullBath'].fillna(value=0, inplace=True)
    df['GarageArea'].fillna(value=0, inplace=True)
    df['GarageCars'].fillna(value=0, inplace=True)
    df['GarageArea'].fillna(value=0, inplace=True)
    df['BsmtHalfBath'].fillna(value=0, inplace=True)
    df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)

    return df


# Load the dataset.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Allow info in bigger dataframes
pd.options.display.max_info_columns = 350

# Clean datasets
clean_train = clean_data(train)
clean_train['MSSubClass_150'] = 0
clean_test = clean_data(test)

# Because of discrepancies in values of some categorical features we have to adjust the features for both datasets.
clean_train_set = set(list(clean_train.columns.values))
clean_test_set = set(list(clean_test.columns.values))
differences = list(clean_test_set ^ clean_train_set)
# Do not take into consideration target feature
differences.remove('SalePrice')

for dif in differences:
    clean_test[dif] = 0

# Let's train the model
X = clean_train.loc[:, clean_train.columns != 'SalePrice']
y = clean_train.loc[:, 'SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print(stats.describe(y_pred))
# The metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred)))
r2 = r2_score(np.log(y_test), np.log(y_pred))
print(rmse)

test_prediction = regr.predict(clean_test)
clean_test['SalePrice'] = test_prediction

submission = clean_test[['SalePrice']]
submission.to_csv('Submission3.csv')

print(submission)
