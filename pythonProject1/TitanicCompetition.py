import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from keras import regularizers
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Training Configuration

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(train_data.columns)

#features
features = list(train_data.columns)
features.remove('PassengerId')
features.remove('Name')
features.remove('Survived')

X = train_data[features]
y = train_data.Survived

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

#Columns differ
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(exclude=['object']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#Model define
X_train1 = preprocessor.fit_transform(X_train)
X_valid1 = preprocessor.transform(X_valid)

input_shape = [X_train1.shape[1]]
X_train1 = X_train1.toarray()
X_valid1 = X_valid1.toarray()


model = RandomForestRegressor(n_estimators=100, random_state=0)
model1 = keras.Sequential([layers.BatchNormalization(),
                          layers.Dense(1024,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
                          layers.BatchNormalization(),
                          layers.Dropout(0.7),
                          layers.Dense(1024,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
                          layers.BatchNormalization(),
                          layers.Dropout(0.7),
                          layers.Dense(1024,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
                          layers.BatchNormalization(),
                          layers.Dropout(0.7),
                          layers.Dense(1024,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
                          layers.BatchNormalization(),
                          layers.Dropout(0.7),
                          layers.Dense(1,activation='sigmoid')
                         ])

model1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

early_stopping = keras.callbacks.EarlyStopping(
    patience=60,
    min_delta=0.01,
    restore_best_weights=True,
)

history = model1.fit(
    X_train1, y_train,
    validation_data=(X_valid1, y_valid),
    batch_size=512,
    epochs=750,
    callbacks=[early_stopping],
    verbose=0
)

"""history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
plt.show()
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
plt.show()"""

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)
print(preds)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

preds1 = model1.predict(X_valid1)
score1 = mean_absolute_error(y_valid, preds1)
print('MAE1:', score1)