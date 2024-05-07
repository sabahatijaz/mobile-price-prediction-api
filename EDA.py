import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def EDA(df):
    print(df.describe())
    feature_data_types = df.dtypes

    # Print the data types
    print("Feature Data Types:")
    print(feature_data_types)
    # Set the style of seaborn plots
    sns.set(style="ticks")

    # Plot histograms for numeric features
    numeric_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 
                        'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
    df[numeric_features].hist(bins=20, figsize=(15, 10))
    plt.suptitle("Histograms of Numeric Features", y=1.02)
    plt.tight_layout()
    plt.show()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Plot scatter plots for numeric features vs. price_range individually
    numeric_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 
                        'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

    for feature in numeric_features:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=feature, y='price_range', data=df)
        plt.title(f"Scatter Plot of {feature.capitalize()} vs. Price Range")
        plt.xlabel(feature.capitalize())
        plt.ylabel("Price Range")
        plt.show()


    # Plot bar plots for categorical features vs. price_range
    categorical_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
    for feature in categorical_features:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature, y='price_range', data=df)
        plt.title(f"{feature.capitalize()} vs. Price Range")
        plt.xlabel(feature.capitalize())
        plt.ylabel("Price Range")
        plt.show()