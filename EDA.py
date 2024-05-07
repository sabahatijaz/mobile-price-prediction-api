import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, numeric_features):
    df[numeric_features].hist(bins=20, figsize=(15, 10))
    plt.suptitle("Histograms of Numeric Features", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_scatter_plots(df, numeric_features):
    for feature in numeric_features:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=feature, y='price_range', data=df)
        plt.title(f"Scatter Plot of {feature.capitalize()} vs. Price Range")
        plt.xlabel(feature.capitalize())
        plt.ylabel("Price Range")
        plt.show()

def plot_bar_plots(df, categorical_features):
    for feature in categorical_features:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature, y='price_range', data=df)
        plt.title(f"{feature.capitalize()} vs. Price Range")
        plt.xlabel(feature.capitalize())
        plt.ylabel("Price Range")
        plt.show()

def EDA(df):
    print(df.describe())
    print("Feature Data Types:")
    print(df.dtypes)
    sns.set(style="ticks")
    
    numeric_features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 
                        'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
    categorical_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

    plot_histograms(df, numeric_features)
    plot_correlation_heatmap(df)
    plot_scatter_plots(df, numeric_features)
    plot_bar_plots(df, categorical_features)
