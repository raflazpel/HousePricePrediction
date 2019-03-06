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

# Import necessary packages.
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import skew

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


def one_hot_encode(df, skip_features=[]):
    for col in [
                'Alley', 'PoolQC', 'Fence', 'MiscFeature',
                'MSSubClass', 'MSZoning', 'Street', 'LotShape', 'MSZoning', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']:
        if col in df.columns and col not in skip_features:
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


def transform_sales_to_log_of_sales(df):
    """
    Our target values distribution get closer to a normal distribution using the log-transformation
    """
    if 'SalePrice' in df:
        df['SalePrice'] = df['SalePrice'].apply(np.log1p)
    return df


def add_expensive_neighborhood_feature(df):
    """
    Instead of using all the neighborhoods, we use a binary classification: are they located in one of the 5 most
    expensive neighborhoods?
    """
    expensive_neighborhoods = ['Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_StoneBr',
                               'Neighborhood_Somerst', 'Neighborhood_Crawfor']

    for neighborhood in expensive_neighborhoods:
        df.loc[df[neighborhood] == 1, "Expensive_Neighborhood"] = 1
    df["Expensive_Neighborhood"].fillna(0, inplace=True)
    df.drop([col for col in df if col.startswith('Neighborhood')], axis=1, inplace=True, errors="ignore")
    return df


def add_home_quality(df):
    df['HomeQuality'] = df['OverallQual'] + df['OverallCond']
    return df


def add_years_since_last_remodel(df):
    df['YearsSinceLastRemodel'] = df['YrSold'].astype(int) - df['YearRemodAdd'].astype(int)
    return df


def remove_too_cheap_outliers(df):
    new_df = df[df["SalePrice"] > 50000]
    if new_df.shape[0] > 500:
        return new_df
    else:
        return df


def remove_garage_cars_feature(df):
    """
    'GarageCars' feature is related to GarageArea feature, it might be interesting to remove it
    """
    df.drop(['GarageCars'], axis=1, inplace=True)
    return df


def remove_lotfrontage_feature(df):
    df.drop(['LotFrontage'], axis=1, inplace=True)
    return df


def drop_empty_features(df):
    """
    Drop features 'Alley', 'PoolQC', 'Fence' and 'MiscFeature', which are almost empty
    """
    df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True, errors='ignore')
    return df


def remove_under_represented_features(df):
    """
    Eliminate those columns with most of the information belonging to the same class
    """
    under_rep = []
    for i in df.columns:
        if i != 'SalePrice':
            counts = df[i].value_counts()
            zeros = counts.iloc[0]
            if ((zeros / len(df)) * 100) > 99.0:
                under_rep.append(i)
    #not_dropped_features = set(df.columns) - set(under_rep)
    df.drop(under_rep, axis=1, inplace=True)
    return df


def feature_selection_lasso(df):
    """
    Use Lasso to select the most meaningful features
    """
    clf = linear_model.Lasso(alpha=0.01)
    X = df.drop(['SalePrice'], axis=1)
    y = df.SalePrice.reset_index(drop=True)
    clf.fit(X, y)
    zero_indexes = np.where(clf.coef_ == 0)[0]
    #not_dropped_features = set(df.columns) - set(zero_indexes)
    if len(df.columns) - len(zero_indexes) > 5:
        df.drop(X.columns[zero_indexes], axis=1, inplace=True)
    return df


def f_regression_feature_filtering(df):
    """
    Select the 18 best features to the target using f-test regression
    (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)
    """
    X = df.drop(['SalePrice'], axis=1)
    y = df.SalePrice.reset_index(drop=True)
    best_features_indexes = SelectKBest(k=18, score_func=f_regression).fit(X, y).get_support(indices=True)
    filtered_features = df.filter(items=X.columns[best_features_indexes], axis=1)
    return filtered_features.join(df.SalePrice)



def drop_categories(df):
    categorical_columns = df.select_dtypes(include='category').columns.tolist()

    for categoricalVariable in categorical_columns:
        if categoricalVariable not in ['ExterQual_Ex', 'ExterQual_Gd', 'ExterQual_TA', 'ExterQual_Fa', 'ExterQual_Po',
                                       'KitchenQual_Ex', 'KitchenQual_Gd', 'KitchenQual_TA', 'KitchenQual_Fa',
                                       'KitchenQual_Po']:
            df.drop([categoricalVariable], axis=1, inplace=True, errors='ignore')

    return df


########################
# MAIN
########################


all_fe_functions = ['add_expensive_neighborhood_feature', 'add_home_quality', 'add_years_since_last_remodel',
                    'sum_SF', 'sum_Porch', 'sum_Baths',
                    'drop_empty_features', 'remove_garage_cars_feature', 'remove_lotfrontage_feature', 'drop_categories',
                    'remove_too_cheap_outliers',
                    'categorical_to_ordinal', 'transform_sales_to_log_of_sales',
                    'f_regression_feature_filtering', 'feature_selection_lasso', 'remove_under_represented_features']
fe_functions_only_for_training_set = ['remove_too_cheap_outliers', 'transform_sales_to_log_of_sales']
dynamic_feature_selection_functions = ['remove_under_represented_features', 'feature_selection_lasso',
                                       'f_regression_feature_filtering']

functions_to_use = ['add_expensive_neighborhood_feature', 'add_home_quality', 'add_years_since_last_remodel', 'sum_SF', 'sum_Baths', 'drop_empty_features', 'remove_lotfrontage_feature', 'drop_categories', 'remove_too_cheap_outliers', 'categorical_to_ordinal', 'transform_sales_to_log_of_sales']

# Load data set
train = pd.read_csv('train.csv').set_index('Id')
test = pd.read_csv('test.csv').set_index('Id')

# Allow info in bigger dataframes
pd.options.display.max_info_columns = 350

# in some cases, there are specific features that we do not want to one-hot encode
skip_one_hot_encode_features = []
if 'categorical_to_ordinal' in functions_to_use:
    skip_one_hot_encode_features = ['PoolQC']

# Data preparation
clean_train = one_hot_encode(fill_na_values(train))
clean_test = one_hot_encode(fill_na_values(test))
clean_train, clean_test = merge_one_hot_encoded_columns(clean_train, clean_test)

# Feature engineering
for fe_function in functions_to_use:
    clean_train = globals()[fe_function](clean_train)
    if fe_function in fe_functions_only_for_training_set:
        continue
    elif fe_function in dynamic_feature_selection_functions:
        # some functions remove features dynamically, we need to apply the same changes to the test data set
        clean_test = clean_test[clean_train.drop('SalePrice', axis=1).columns]
    else:
        clean_test = globals()[fe_function](clean_test)

# Save the feature engineered data
clean_train.to_csv('training_FE_data.csv')
clean_test.to_csv('test_FE_data.csv')


# From here you can delete it and put it in a linear regression python file
# TODO: Move rest of the code to other python file.

X = clean_train.loc[:, clean_train.columns != 'SalePrice']
y = clean_train.loc[:, 'SalePrice']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# regr.fit(X_train, y_train)
# y_pred = regr.predict(X_test)

# The metrics
# score = r2_score(y_test, y_pred)
scores = cross_val_score(regr, X, y, cv=5, n_jobs=-1)
score = scores.mean()
# print(stats.describe(regr.coef_))
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred)))
# r2 = r2_score(np.log(y_test), np.log(y_pred))
# print(" FUNCTIONS : {}".format(functions))
# print(" sklearn score: {}".format(regr.score(X_test, y_test)))
# print("r2 {}".format(r2))
# print("rmse {}".format(rmse))

# keep track of results so far
string_result = "functions : {0}, r^2 score; {1}".format(functions_to_use, score)
print("\n")
print(string_result)

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
test_prediction = regr2.predict(clean_test)

# if SalePrice has been log-transformed, we must carry out the inverse operation (exp)
if 'transform_sales_to_log_of_sales' in functions_to_use:
    test_prediction = np.expm1(test_prediction)

clean_test['SalePrice'] = test_prediction
submission = clean_test[['SalePrice']]
submission.to_csv('FirstGroupSubmission.csv')