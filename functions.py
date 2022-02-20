import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_intervals(df: pd.DataFrame, column: str, n_intervals: int):
    # Create the bins
    array_percentile = np.linspace(0, 1, n_intervals + 1)
    list_bins = list(df[[column]].quantile(array_percentile)[column])
    list_bins[0], list_bins[-1] = -np.inf, +np.inf

    # Create some value names
    new_column_name = column + "_cat"
    no_value_name = 'No ' + column

    # Adjust people with the correct interval
    df[new_column_name] = no_value_name
    df.loc[df[column].notnull(), new_column_name] = pd.cut(
        df[column].loc[df[column].notnull()], bins=list_bins)

def create_df_freq(serie: pd.Series):
    list_words = []
    for string in serie:
        list_words += string.split()

    serie_freq = pd.value_counts(np.array(list_words))
    return pd.DataFrame(serie_freq)

def plot_survived_cat_analysis(df: pd.DataFrame, column_name: str):
    df_grouped = df[[column_name, "Survived"]].groupby([column_name])
    
    plot = df_grouped.agg(np.mean).plot(kind='bar')
    for container in plot.containers:
        plot.bar_label(container)
    plt.title("Survived percentage")
    plt.show()

    plot = df_grouped.agg(len).plot(kind='bar', label='Quantity')
    for container in plot.containers:
        plot.bar_label(container)
    plt.title("Quantity")
    plt.show()

def plot_survived_tab_analysis(df: pd.DataFrame, column_name: str):
    df_grouped = df[[column_name, "Survived"]].groupby([column_name])
    print(df_grouped.agg([np.mean, len]))