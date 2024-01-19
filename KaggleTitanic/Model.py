import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Model Configuration
UNITS = 2 ** 11 # 2048
ACTIVATION = 'relu'
DROPOUT = 0.1
BATCH_SIZE_PER_REPLICA = 2**11


numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Training Configuration

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(train_data.columns)
features = list(train_data.columns).remove(['PassengerId', 'Survived', 'Name'])
object_cols = train_data[train_data.dtypes=='object']
print("Categorical variables:")
print(object_cols)





