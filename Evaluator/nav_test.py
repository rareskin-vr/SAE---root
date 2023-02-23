import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load the data
df = pd.read_csv('finaldataset.csv')

# Split the data into features (X) and target (y)
X = df[['keyword', 'grammar', 'qst']]
y = df['class']

# Create a preprocessing pipeline that one-hot encodes the categorical features
# and leaves the numerical feature unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False), ['keyword', 'grammar']),
        ('num', 'passthrough', ['qst'])
    ])

# Create a pipeline that applies the preprocessing and the classifier
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit the pipeline on the training data
pipe.fit(X, y)

# Define a function to predict the class of new data
def predict(k, g, q):
    # Convert the input to a numpy array and reshape it
    input_data = np.array([[k, g, q]])
    input_data = input_data.reshape(1, -1)

    # Convert the numpy array back to a pandas DataFrame
    input_df = pd.DataFrame(input_data, columns=X.columns)

    # Use the pipeline to predict the class of the input data
    predicted = pipe.predict(input_df)
    # result = predicted * 5 / 10
    # print(result[0])
    return predicted

