from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset.
train = pd.read_csv('training_FE_data.csv', index_col=0)
submission_set = pd.read_csv('test_FE_data.csv', index_col=0)

X = train.loc[:, train.columns != 'SalePrice']
y = train.loc[:, 'SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tree_model = RandomForestRegressor(random_state=42)
tree_model = tree_model.fit(X_train, y_train)
print(tree_model.score(X_test, y_test))


# We make the prediction
result = tree_model.predict(submission_set)
submission_set['SalePrice'] = result
submission = submission_set[['SalePrice']]
submission.to_csv('FirstForestSubmission.csv')