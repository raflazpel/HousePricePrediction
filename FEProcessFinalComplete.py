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

#F unctions


def sum_SF(df):
    columns_to_add = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2']
    df['House_SF'] = df[columns_to_add].sum(axis=1)
    df.drop(columns_to_add, axis=1, inplace=True)
    return df


def sum_Baths(df):
    df['Total_Baths'] = (df['FullBath'] + 
                         df['BsmtFullBath'] + 
                         (0.8*df['HalfBath']) + 
                         (0.8*df['BsmtHalfBath']))
    df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1,inplace = True)
    return df

def sum_Porch(df):
    columns_to_add = ['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF']
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
    print(feature_skew)
    print(numeric_features)
    high_skew = feature_skew[feature_skew > 0.9]
    skew_index = high_skew.index
    print(high_skew)
    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

    skew_features = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':skew_features})
    print(skews)
    return df


def clean_data(df):
    # Change the index and drop almost empty variables.
    df = df.set_index('Id')
    df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
         
    # Now I want to mark the categorical variables.

    # Strategy 1: Almost all non-numeric variables are categorical,
    # so I find them and mark them as categorical

    # Second step would be to find all the features that are numerical-categorical columns and mark them
    # (Because we already marked non-numerical.
    # In this case I have done it manually with ALL categorical columns

    # Modificacion: Submission 7: hemos pasado Overall Qal y cond a no categor
    for col in ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'MSZoning', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
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
        dummy = pd.get_dummies(df[categoricalVariable], prefix=categoricalVariable).astype('category')
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


# Este bloque deja los datasets limpios y listos para f.e

clean_train = clean_data(train)
clean_train['MSSubClass_150'] = 0
clean_train['MSSubClass_150'] = clean_train['MSSubClass_150'].astype('category')
clean_test = clean_data(test)


clean_test_set = set(list(clean_test.columns.values))

clean_train_set = set(list(clean_train.columns.values))

differences = list(clean_test_set ^ clean_train_set)
differences.remove('SalePrice')
print(differences)

for dif in differences:
    clean_test[dif] = 0
    clean_test[dif] = clean_test[dif].astype('category')


# In[5]:


# Modificación 1: Solo variables no-categoricas. (Submission 4)
'''print(clean_test.select_dtypes(include='category').columns)

def drop_categories(df):
    
    categorical_columns = df.select_dtypes(include='category').columns.tolist()

    for categoricalVariable in categorical_columns:
        df.drop([categoricalVariable], axis=1, inplace=True)
    
    return df

clean_train = drop_categories(clean_train)
clean_test = drop_categories(clean_test)

'''

# Modificación 2: Sobre la modificación 1 metemos algunas variables categoricas (Submission 5)


def drop_categories(df):
    categorical_columns = df.select_dtypes(include='category').columns.tolist()

    for categoricalVariable in categorical_columns:
        if categoricalVariable not in ['ExterQual_Ex', 'ExterQual_Gd', 'ExterQual_TA', 'ExterQual_Fa', 'ExterQual_Po', 'KitchenQual_Ex','KitchenQual_Gd','KitchenQual_TA','KitchenQual_Fa','KitchenQual_Po']:
            df.drop([categoricalVariable], axis=1, inplace=True)
    
    return df


print(clean_test.select_dtypes(include='category').columns)
clean_train = drop_categories(clean_train)
clean_test = drop_categories(clean_test)
print(clean_test.select_dtypes(include='category').columns)


clean_test = sum_SF(clean_test)
clean_test = sum_Porch(clean_test)
clean_test = sum_Baths(clean_test)
clean_train = sum_SF(clean_train)
clean_train = sum_Porch(clean_train)
clean_train = sum_Baths(clean_train)


# In[10]:


X = clean_train.loc[:, clean_train.columns != 'SalePrice']
y = clean_train.loc[:, 'SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
# Create linear regression object
regr = linear_model.LinearRegression(n_jobs = 1)

regr.fit(X_train, y_train)
print(regr)
y_pred = regr.predict(X_test)

# The coefficients
# print('Coefficients: \n', regr.coef_)
    
# The metrics

print(stats.describe(regr.coef_))
#print('Coefficients: \n', regr.coef_)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(np.log(y_test),np.log(y_pred)))
r2 = r2_score(np.log(y_test), np.log(y_pred))
print(r2)
print(rmse)

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
print(submission)
 
clean_train.info()
