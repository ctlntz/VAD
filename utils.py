import numpy as np
import pandas as pd


### Data manipulation ###
def read_csv(file_path):
  # Citirea fișierului CSV
    data = pd.read_csv(file_path)

    # Initialize the sets
    train_set, val_set = [], []

    # Păstrarea doar a ID, Clasa și Features
    data = data.loc[:, ~data.columns.isin(['ID'])]
    train_set = data[data['Split'] == 'Train']
    val_set = data[data['Split'] == 'Test']

    X_train = train_set.iloc[:, 1:16].to_numpy(dtype=np.float32)  # Features
    Y_train = train_set.iloc[:, -2].to_numpy(dtype=np.uint8)     # Clase

    X_val = val_set.iloc[:, 1:16].to_numpy(dtype=np.float32)      # Features
    Y_val = val_set.iloc[:, -2].to_numpy(dtype=np.uint8)
    return X_train, Y_train, X_val, Y_val