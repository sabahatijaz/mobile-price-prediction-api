import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import logging

def preprocess(df):
    """
    Preprocess the input dataframe by handling missing values.

    Parameters:
    df (DataFrame): Input dataframe.

    Returns:
    DataFrame: Preprocessed dataframe with missing values removed.
    """
    missing_values = df.isnull().sum()
    logging.info("Missing Values in data:\n%s", missing_values)
    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    """
    Perform feature engineering by creating a new feature 'pixel_density'.

    Parameters:
    df (DataFrame): Input dataframe.

    Returns:
    DataFrame: Dataframe with the 'pixel_density' feature added.
    """
    df['pixel_density'] = df['px_height'] * df['px_width']
    return df

def select_features(df, k=10):
    """
    Select the top k features using chi-squared test.

    Parameters:
    df (DataFrame): Input dataframe.
    k (int): Number of top features to select.

    Returns:
    list: List of selected feature names.
    """
    X = df.drop(['price_range'], axis=1)  # Features
    y = df['price_range']  # Target variable

    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)

    selected_features_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_features_indices].tolist() + ['price_range']

    logging.info("Selected Features: %s", selected_features)
    return selected_features
