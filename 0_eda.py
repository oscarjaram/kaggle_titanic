# %%
# Jupyter commands
# %load_ext autoreload
# %autoreaload 2

# Import libraries
from re import A
from typing import List
import params
import functions
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
# Pclass Analysis
functions.plot_survived_cat_analysis(df_train, 'Pclass')
# Conclusion: Pclass is relevant to predictability

# %%
# Name Analysis
serie_name = df_train.Name.copy()
df_name = df_train[["Name", "Survived"]].copy(deep=True)

# Print the list of names
df_name_freq = functions.create_df_freq(serie_name)
df_name_freq.head(20)

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
functions.plot_survived_cat_analysis(df_name, 'Name_title')

# %%
# Create value for quote
df_name["Name_quote"] = df_name.Name.str.contains('\"').astype(int)
functions.plot_survived_cat_analysis(df_name, 'Name_quote')

# %%
# Create value for parenthesis
df_name["Name_parenthesis"] = df_name.Name.str.contains('\)').astype(int)
functions.plot_survived_cat_analysis(df_name, 'Name_parenthesis')

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
functions.plot_survived_cat_analysis(df_name, 'Name_number_words')

# %%
# Large of the names and lastnames
df_name["Name_name"] = df_name.Name.str.split(",", expand=True)[1]

df_name["Name_name_count"] = df_name["Name_name"]\
    .str.replace(' ', '')\
    .str.replace(',', '')\
    .str.replace('.', '')\
    .str.len()

functions.plot_survived_cat_analysis(df_name, 'Name_name_count')

# %%
df_name["Name_lastname"] = df_name.Name.str.split(",", expand=True)[0]

df_name["Name_lastname_count"] = df_name["Name_lastname"]\
    .str.replace(' ', '')\
    .str.replace(',', '')\
    .str.replace('.', '')\
    .str.len()

functions.plot_survived_cat_analysis(df_name, 'Name_lastname_count')
df_name.head()

# %%
# Composed lastname
df_name["Name_lastname_composed"] = df_name["Name_lastname"]\
    .str.contains('-')\
    .astype('int')

functions.plot_survived_cat_analysis(df_name, 'Name_lastname_composed')
df_name.head()

# %%
# Common names: William(62), John(44), Henry(33), James(24), Charles(23), George(22), Thomas(21)
list_common_names = ['William', ' John', ' Henry', ' Charles', ' George', 'Sage']
for name in list_common_names:
    str_column_name = "Name_" + name
    df_name[str_column_name] = df_name.Name.str.contains(name).astype(int)
    print(name)
    functions.plot_survived_cat_analysis(df_name, str_column_name)
    print("-"*50)

# %%
# Sex analysis
functions.plot_survived_cat_analysis(
    df=df_train, column_name='Sex')

# %%
# Age analysis
functions.plot_survived_cat_analysis(
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
df_age["hasAge"] = df_age.Age.notnull().astype('int')

# The people with age has 40.6% probability vs 29.3% the people without age.
df_age[["hasAge", "Survived", "Sex", "Pclass"]].groupby(["hasAge", "Sex", "Pclass"]).agg([np.mean, len])

# %%
# Create age intervals and evaluate
functions.create_intervals(df_age, 'Age', 25)
functions.plot_survived_cat_analysis(df_age, 'Age_cat')

# %%
# Create age logarithm scale and evaluate
df_age["Age_log"] = np.trunc(np.log2(df_age.Age))
functions.plot_survived_cat_analysis(df_age, 'Age_log')

# %%
# SibSp analysis
functions.plot_survived_cat_analysis(df_train, 'SibSp')

# It has decrease exponential distribution
# There are a difference between "Alone person" than the people with Company
# There are a difference between "Big family" (more than 5) than the people with Company
df_sibsp = df_train[["SibSp", "Survived"]].copy(deep=True)
df_sibsp["SibSp_alone"] = (df_sibsp.SibSp == 0).astype('int')
df_sibsp["SibSp_bigfamily"] = (df_sibsp.SibSp > 4).astype('int')

df_sibsp.head(100)
# %%
# Parch analysis
functions.plot_survived_cat_analysis(df_train, 'Parch')

df_parch = df_train[["Parch", "Survived"]].copy(deep=True)
df_parch["Parch_alone"] = (df_parch.Parch == 0).astype('int')
df_parch["Parch_bigfamily"] = (df_parch.Parch > 3).astype('int')

df_parch.head(100)

# %%
# Ticket analysis
df_ticket = df_train[['Ticket', 'Survived']].copy(deep=True)
serie_ticket = df_train['Ticket'].copy(deep=True)

df_ticket["Ticket_number"] =  serie_ticket\
    .str.replace(".", "")\
    .str.rsplit(" ", n=1)\
    .apply(lambda x: x[-1])

df_ticket["Ticket_number"] = pd.to_numeric(
    df_ticket["Ticket_number"], 
    errors='coerce'
)

def _get_ticket_str(list_str: List[str]):
    if len(list_str) < 2:
        return np.nan
    return " ".join(list_str[:-1])\
        .upper().replace(" ", "")

df_ticket["Ticket_code"] = serie_ticket\
    .str.replace(".", "")\
    .str.replace("/", " ")\
    .str.rsplit(" ", n=1)\
    .apply(_get_ticket_str)

df_ticket.head()

# %%
# Get the correct bins to adjust intervals with equal volumne 
functions.create_intervals(df_ticket, 'Ticket_number', 45)
functions.plot_survived_cat_analysis(df_ticket, 'Ticket_number_cat')
# We could use more intervals and then classify using the survive probability

# %%
# Print the list of names
big_list_words = []
for ticket in df_ticket["Ticket_code"]:
    big_list_words += str(ticket).upper().replace(" ", "").split()

# Print most frequent words
serie_ticket_words = pd.value_counts(np.array(big_list_words))
df_ticket_words = pd.DataFrame(serie_ticket_words)
df_ticket_words.head(20)

# %%
# Has ticket code
df_ticket["hasTicket_code"] = df_ticket.Ticket_code.notnull().astype(int)
functions.plot_survived_cat_analysis(df_ticket, 'hasTicket_code')
# It seems no relevant information

# %%
# Has ticket code with /
df_ticket["Ticket_slash"] = df_ticket.Ticket.str.contains("/").astype('int')
functions.plot_survived_cat_analysis(df_ticket, 'Ticket_slash')

# %%
# Lenght of ticket code
df_ticket["Ticket_number_log"] = np.trunc(np.log2(df_ticket.Ticket_number))
functions.plot_survived_cat_analysis(df_ticket, 'Ticket_number_log')

# %%
# This ideas has descarted to affect many little rows
# Frequent words: PC(60), CA(41), A5(21), STONO2(18), SOTONOQ(15), 
# SCPARIS(11), WC(10), A4(7), OQ(15), O(12), 2(12), W(11), PP(10), O2(8), PARIS(7)
# Has ticket code common prefix

# %%
# Fare analysis
df_fare = df_train[["Fare", "Survived"]]
serie_fare = df_fare["Fare"]

functions.create_intervals(df_fare, 'Fare', 20)
functions.plot_survived_cat_analysis(df_fare, 'Fare_cat')

# %%
# Add some exponential scaler to the value
df_fare["Fare_log"] = np.trunc(np.log2(df_fare.Fare))
functions.plot_survived_cat_analysis(df_fare, 'Fare_log')

# %% Cabin Analysis
df_cabin = df_train[["Cabin", "Survived"]]
serie_cabin = df_cabin["Cabin"]

# %%
df_cabin["hasCabin"] = df_cabin.Cabin.notnull().astype(int)
functions.plot_survived_cat_analysis(df_cabin, 'hasCabin')

# %%
df_cabin["Cabin_letter"] = 'No Cabin'
df_cabin.loc[df_cabin.Cabin.notnull(), "Cabin_letter"] = df_cabin.Cabin.str[0]
functions.plot_survived_cat_analysis(df_cabin, 'Cabin_letter')
# Some numbers (We don't going to test it because its groups are too litle)

# %%
# Embarked analysis
functions.plot_survived_cat_analysis(df_train, 'Embarked')
