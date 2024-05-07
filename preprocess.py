import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def preprocess(df):
    missing_values = df.isnull().sum()
    print("Missing Values in data:\n", missing_values)
    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    df['pixel_density'] = df['px_height'] * df['px_width']
    return df

def select_features(df):
    X = df.drop(['price_range'], axis=1)  # Features
    y = df['price_range']  # Target variable

    # Select the top k=10 features
    selector = SelectKBest(score_func=chi2, k=10)
    X_new = selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_features_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_features = X.columns[selected_features_indices]
    selected_features = selected_features.tolist() + ['price_range']  # Append 'price_range' column name
    # Print the selected features
    print("Selected Features:")
    print(selected_features)
    return selected_features
