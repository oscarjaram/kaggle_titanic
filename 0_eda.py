# %%
# Jupyter commands
# %load_ext autoreload
# %autoreaload 2

# Import libraries
import params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 2000)

# %%
# Load the data and backup the dataframe
df_train = pd.read_csv(params.path_train_data)
df_train_backup = df_train.copy(deep=True)
df_train.head()

# %%
# See the data info
print(df_train.info(verbose=True))
df_train.describe()

# %%
# See the correlation between variables
sns.pairplot(df_train)
plt.show()

# Mean probability of survive is 38.39%

# %%
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
# %%
# Pclass Analysis
plot_survived_cat_analysis(
    df=df_train, column_name='Pclass')

# Pclass 1: 62.96
# Pclass 2: 47.28
# Pclass 3: 24.23
# Conclusion: Pclass is relevant to predictability

# %%
# Name Analysis
serie_name = df_train.Name.copy()
df_name = df_train[["Name", "Survived"]].copy(deep=True)

# Print the list of names
big_list_words = []
for name in serie_name:
    name_list_words = name.split()
    big_list_words += name_list_words

# Print most frequent words
array_words = np.array(big_list_words)
words_freq = pd.value_counts(array_words).to_dict()
for key, value in words_freq.items():
    print(f"{key}: {value}")

# Interesting words:
# Mr.(517), Miss(182), Mrs.(125), Master(40), Jr(10), Dr(7), Don(), Sage(7), Rev(7)
# Common Names: William(62), John(44), Henry(33), James(24), Charles(23), George(22), Thomas(21)
# Short Names: H(8), J(8), E(6)

# %%%
# Create attribute for title (Mrs, Mr, Miss)
df_name["Name_title"] = df_name.Name\
    .str.split(",", expand=True)[1]\
    .str.split(" ", expand=True, n=2)[1]
df_name.head()

title_list = ["Mr.", "Miss.", "Mrs.", "Master.", "Rev.", "Dr."]
df_name["Name_title"].loc[~df_name["Name_title"].isin(title_list)] = "Other"

plot_survived_cat_analysis(
    df=df_name, column_name='Name_title')

# %%
# Create value for quote
df_name["Name_quote"] = df_name.Name.str.contains('\"').astype(int)
df_name.loc[df_name["Name_quote"] > 0].head()

plot_survived_cat_analysis(
    df=df_name, column_name='Name_quote')

# %%
# Create value for parenthesis
df_name["Name_parenthesis"] = df_name.Name.str.contains('\)').astype(int)
df_name.loc[df_name["Name_parenthesis"] > 0].head()

plot_survived_cat_analysis(
    df=df_name, column_name='Name_parenthesis')

# %%
# Clean the characters in the name
df_name['Name'] = df_name['Name']\
    .str.replace('\"', '')\
    .str.replace('\(', '')\
    .str.replace('\)', '')
df_name.head()

# %%
# Create value for number of words
df_name["Name_number_words"] = df_name.Name.str.count(' ') + 1

plot_survived_cat_analysis(
    df=df_name, column_name='Name_number_words')

# %%
# Large of the names and lastnames
df_name["Name_name"] = df_name.Name.str.split(",", expand=True)[1]

df_name["Name_name_count"] = df_name["Name_name"]\
    .str.replace(' ', '')\
    .str.replace(',', '')\
    .str.replace('.', '')\
    .str.len()

plot_survived_cat_analysis(
    df=df_name, column_name='Name_name_count')

# %%
df_name["Name_lastname"] = df_name.Name.str.split(",", expand=True)[0]

df_name["Name_lastname_count"] = df_name["Name_lastname"]\
    .str.replace(' ', '')\
    .str.replace(',', '')\
    .str.replace('.', '')\
    .str.len()

plot_survived_cat_analysis(
    df=df_name, column_name='Name_lastname_count')

df_name.head()

# %%
# Composed lastname
df_name["Name_lastname_composed"] = df_name["Name_lastname"]\
    .str.contains('-')\
    .astype('int')

plot_survived_cat_analysis(
    df=df_name, column_name='Name_lastname_composed')

df_name.head()


# %%
# Common names: William(62), John(44), Henry(33), James(24), Charles(23), George(22), Thomas(21)
list_common_names = ['William', ' John', ' Henry', ' Charles', ' George', 'Sage']
for name in list_common_names:
    str_column_name = "Name_" + name
    df_name[str_column_name] = df_name.Name.str.contains(name).astype(int)
    print(name)
    plot_survived_cat_analysis(df_name, str_column_name)
    print("-"*50)

# %%
# Sex analysis
plot_survived_cat_analysis(
    df=df_train, column_name='Sex')

# %%
# Age analysis
plot_survived_cat_analysis(
    df=df_train, column_name='Age')

# %%
df_age = df_train[["Age", "Survived"]].copy(deep=True)
df_age.head()

# %%
df_train.Age.isna().sum()
# 177 values without age

# %%
# Have the NA values the same survive probability?
df_age = df_train[["Age", "Survived", "Sex", "Pclass"]].copy(deep=True)
df_age["hasAge"] = 1
df_age["hasAge"].loc[df_age["Age"].isna()] = 0

# The people with age has 40.6% probability vs 29.3% the people without age.
df_age[["hasAge", "Survived", "Sex", "Pclass"]].groupby(["hasAge", "Sex", "Pclass"]).agg([np.mean, len])

# %%
bins = [-np.inf, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]

df_age["Age_cat"] = 'No Age'
df_age.loc[df_age.Age.notnull(), "Age_cat"] = pd.cut(
    df_age["Age"].loc[df_age.Age.notnull()], bins)

plot_survived_cat_analysis(
    df=df_age, column_name='Age_cat')

# %%
