from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset.
train = pd.read_csv('training_FE_data.csv')
test = pd.read_csv('test.csv')

X = train.loc[:, train.columns != 'SalePrice']
y = train.loc[:, 'SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X.info)
tree_model = tree.DecisionTreeRegressor()
tree_model = tree_model.fit(X_train, y_train)
print(X_test.iloc[0])
result = tree_model.predict(X_test.iloc[0])
print(result)
