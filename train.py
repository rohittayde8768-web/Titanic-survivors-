import pandas as pd 
import numpy as np 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("Data/Titanic-Dataset.csv")
df.head()
df.info()

## Feature Selection
features= ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
 
X= df[features]
y= df['Survived']

# Seperate numerical and categorical features
numerical_features= ['Age', 'SibSp', 'Parch', 'Fare']
# Categorical features:-
categorical_features= ['Pclass', 'Sex', 'Embarked']

# Preprocessing for numerical data
numerical_transformer= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

processor= ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a model:-
model=RandomForestClassifier(n_estimators=100, random_state=42)

# Create and evaluate the pipeline
clf= Pipeline(steps=[('preprocessor', processor),
                      ('classifier', model)])

# Split the dataset into train and test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

#Accureacy Score 
accuracy= pipeline_score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
