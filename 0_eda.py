# %%
# Import libraries
import params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Load the data
df_train = pd.read_csv(params.path_train_data)
df_train.head()

# %%
df_train.info()

# %%
df_train.describe()

# %%
