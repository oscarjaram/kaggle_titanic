# %%
# Import libraries
from pathlib import Path

# %%
# Create directory paths
cwd = Path.cwd()
folder_data = cwd.joinpath("data")
folder_raw_data = folder_data.joinpath("raw_data")
folder_model_data = folder_data.joinpath("model_data")
folder_prep_data = folder_data.joinpath("prep_data")
results = folder_data.joinpath("results")

# Create file paths
path_train_data = folder_raw_data.joinpath("train.csv")
path_test_data = folder_raw_data.joinpath("test.csv")
