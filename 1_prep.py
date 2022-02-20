# Import libraries
from typing import List
import params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 2000)

# Load the data and backup the dataframe
df_train = pd.read_csv(params.path_train_data)

# Feature engineering

## Name
df_train["Name_title"] = df_train.Name\
    .str.split(",", expand=True)[1]\
    .str.split(" ", expand=True, n=2)[1]
df_train.head()

title_list = ["Mr.", "Miss.", "Mrs.", "Master.", "Rev.", "Dr."]
df_train["Name_title"].loc[~df_train["Name_title"].isin(title_list)] = "Other"

df_train["Name_quote"] = df_train.Name\
    .str.contains('\"').astype(int)
df_name["Name_parenthesis"] = df_name.Name\
    .str.contains('\)').astype(int)

df_name['Name'] = df_name['Name']\
    .str.replace('\"', '')\
    .str.replace('\(', '')\
    .str.replace('\)', '')

df_name["Name_number_words"] = df_name.Name\
    .str.count(' ') + 1

df_name["Name_name_count"] = df_name.Name\
    .str.split(",", expand=True)[1]\
    .str.replace(' ', '')\
    .str.replace(',', '')\
    .str.replace('.', '')\
    .str.len()

df_name["Name_lastname_count"] = df_name.Name\
    .str.split(",", expand=True)[0]\
    .str.replace(' ', '')\
    .str.replace(',', '')\
    .str.replace('.', '')\
    .str.len()

df_name["Name_lastname_composed"] = df_name.Name\
    .str.split(",", expand=True)[0]\
    .str.contains('-')\
    .astype('int')

list_common_names = ['William', ' John', ' Henry', ' Charles', ' George', 'Sage']
for name in list_common_names:
    str_column_name = "Name_" + name
    df_name[str_column_name] = df_name.Name\
        .str.contains(name).astype(int)

## Age
df_age["hasAge"] = df_age.Age.notnull().astype('int')

bins = [-np.inf, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]

df_age["Age_cat"] = 'No Age'
df_age.loc[df_age.Age.notnull(), "Age_cat"] = pd.cut(
    df_age["Age"].loc[df_age.Age.notnull()], bins)

