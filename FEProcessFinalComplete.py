#!/usr/bin/env python
# coding: utf-8

# # Feature engineering: Improving model´s result without changing the model.
# 
# This notebook has as objetive to analyse and understand the details of the feature engineering process
#  (f.e from now on) Even though I have duplicated the Dataiku pipeline given by Jesus to obtain a test dataset
#  preprocessed exactly as the train dataset in this notebook I will start the whole process from scratch to be
# sure I am able to perform the whole process.
# 
# As we are performing F.E the only rule will be not to change the prediction model built at the professor's notebook.

# Import necesary packages.
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

############################
# DATA PREPARATION FUNCTIONS
############################


def fill_na_values(df):
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


def one_hot_encode(df):
    for col in ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'MSZoning', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Apply one-hot-enconding to all categorical variables.
    # Suponemos que todas las variables categoricas son nominales. Se puede aplicar integer-encoding
    # a las que sean ordinales.
    categorical_columns = df.select_dtypes(include='category').columns.tolist()

    for categoricalVariable in categorical_columns:
        dummy = pd.get_dummies(df[categoricalVariable], prefix=categoricalVariable).astype('category')
        df = pd.concat([df, dummy], axis=1)
        df.drop([categoricalVariable], axis=1, inplace=True)

    return df


def merge_one_hot_encoded_columns(train_df, test_df):
    """
    After one-hot encoding, some columns might exist in the train dataset and not in the test one, or viceversa.
    If a column exists in one of the two dataframes and not in the other, we create it and fill it with zeros.
    """
    clean_test_set = set(list(test_df.columns.values))
    clean_train_set = set(list(train_df.columns.values))

    differences = list(clean_test_set ^ clean_train_set)
    differences.remove('SalePrice')

    for dif in differences:
        if dif not in train_df:
            train_df[dif] = 0
            train_df[dif] = train_df[dif].astype('category')
        if dif not in test_df:
            test_df[dif] = 0
            test_df[dif] = test_df[dif].astype('category')
    return train_df, test_df


###############################
# FEATURE ENGINEERING FUNCTIONS
###############################

def sum_SF(df):
    columns_to_add = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2']
    if pd.Series(columns_to_add).isin(df.columns).all():
        df['House_SF'] = df[columns_to_add].sum(axis=1)
        df.drop(columns_to_add, axis=1, inplace=True)
    return df


def sum_Baths(df):
    bath_features = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if pd.Series(bath_features).isin(df.columns).all():
        df['Total_Baths'] = (df['FullBath'] +
                             df['BsmtFullBath'] +
                             (0.8*df['HalfBath']) +
                             (0.8*df['BsmtHalfBath']))
        df.drop(bath_features, axis=1,inplace = True)
    return df


def sum_Porch(df):
    columns_to_add = ['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF']
    if pd.Series(columns_to_add).isin(df.columns).all():
        df['Porch_sf'] = df[columns_to_add].sum(axis=1)
        df.drop(columns_to_add, axis=1,inplace=True)
    return df


def feature_skewness(df):
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_features = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes: 
            numeric_features.append(i)

    feature_skew = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':feature_skew})
    return feature_skew, numeric_features


def fix_skewness(df):
    feature_skew, numeric_features = feature_skewness(df)
    high_skew = feature_skew[feature_skew > 0.9]
    skew_index = high_skew.index
    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

    #skew_features = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    #skews = pd.DataFrame({'skew':skew_features})
    return df


def categorical_to_ordinal(df):
    """
    Some textual features(e.g.basement quality) should be handled as numerical (i.e.ordinal) values
    """

    ordinal_features = ['ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'PoolQC',
                        'ExterCond', 'BsmtCond', 'GarageCond']
    for ordinalFeature in ordinal_features:
        if ordinalFeature in df:
            df[ordinalFeature].fillna(value=0, inplace=True)
            df[ordinalFeature] = df[ordinalFeature].replace({
                                            'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1,
                                            'NA': 0
                                            }).astype('int32')
    if 'Foundation' in df:
        df['Foundation'].fillna(value=0, inplace=True)
        df['Foundation'] = df['Foundation'].replace({
                                            'PConc': 3,
                                            'CBlock': 2,
                                            'BrkTil': 1,
                                            'Slab': 0,
                                            'Stone': 0,
                                            'Wood': 0,
                                            'NA': 0
                                            }).astype('int32')
    return df


def drop_empty_features(df):
    """
    Drop features 'Alley', 'PoolQC', 'Fence' and 'MiscFeature', which are almost empty
    """
    df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True, errors='ignore')
    return df


def drop_categories(df):
    categorical_columns = df.select_dtypes(include='category').columns.tolist()

    for categoricalVariable in categorical_columns:
        if categoricalVariable not in ['ExterQual_Ex', 'ExterQual_Gd', 'ExterQual_TA', 'ExterQual_Fa', 'ExterQual_Po',
                                       'KitchenQual_Ex', 'KitchenQual_Gd', 'KitchenQual_TA', 'KitchenQual_Fa',
                                       'KitchenQual_Po']:
            df.drop([categoricalVariable], axis=1, inplace=True, errors='ignore')

    return df


# Load the dataset.
train = pd.read_csv('train.csv').set_index('Id')
test = pd.read_csv('test.csv').set_index('Id')

#print(train['ExterCond'].isnull().values.any())
#print(train[train['PoolQC'].isnull()]['PoolQC'])

#exit(0)

# Allow info in bigger dataframes
pd.options.display.max_info_columns = 350


# Data preparation
clean_train = one_hot_encode(fill_na_values(train))
clean_test = one_hot_encode(fill_na_values(test))
clean_train, clean_test = merge_one_hot_encoded_columns(clean_train, clean_test)

# Feature engineering
all_fe_functions = ['drop_categories', 'categorical_to_ordinal', 'sum_SF', 'sum_Porch', 'sum_Baths', 'drop_empty_features', 'fix_skewness']
fe_functions_only_for_training_set = ['fix_skewness']

for fe_function in all_fe_functions:
    clean_train = globals()[fe_function](clean_train)
    if fe_function not in fe_functions_only_for_training_set:
        clean_test = globals()[fe_function](clean_test)

# Save the feature engineered data
clean_train.to_csv('training_FE_data.csv')
clean_test.to_csv('test_FE_data.csv')

# From here you can delete it and put it in a linear regression python file
# TODO: Move rest of the code to other python file.

X = clean_train.loc[:, clean_train.columns != 'SalePrice']
y = clean_train.loc[:, 'SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
# Create linear regression object
regr = linear_model.LinearRegression(n_jobs=1)

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
    
# The metrics

#print(stats.describe(regr.coef_))

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(np.log(y_test),np.log(y_pred)))
r2 = r2_score(np.log(y_test), np.log(y_pred))
print(" sklearn score: {}".format(regr.score(X_test, y_test)))
print("r2 {}".format(r2))
print("rmse {}".format(rmse))

# Reentrenar con datos de validacion y cargar en csv
regr2 = linear_model.LinearRegression()
regr2.fit(X, y)
'''
test_prediction = regr2.predict(clean_test)
clean_test['SalePrice'] = test_prediction


submission = clean_test[['SalePrice']]
submission.to_csv('Submission9-RETRAINED.csv')
print(submission)

'''
test_prediction = regr.predict(clean_test)
clean_test['SalePrice'] =  test_prediction


submission = clean_test[['SalePrice']]
submission.to_csv('FirstGroupSubmission.csv')

#clean_train.info()
