import numpy as np
import pandas as pd

### Data manipulation ###
def read_csv(file_paths):
    # Initialize the sets as empty DataFrames
    train_set = pd.DataFrame()
    val_set = pd.DataFrame()

    # Loop through the provided file paths
    for i, file_path in enumerate(file_paths):
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Print the DataFrame to debug
            print(f"Data from {file_path}:")
            print(data.head())  # Print the first few rows of the DataFrame
            print(f"Columns: {data.columns.tolist()}")  # Print the column names



            # Keep only the relevant columns (excluding 'ID' and 'speech_label')
            data = data.loc[:, ~data.columns.isin(['ID'])]

            # Check if the DataFrame is empty after filtering
            if data.empty:
                print(f"No relevant data in {file_path} after filtering.")
                continue

            # Split the data into train and validation sets
            if i < 3:  # First 3 files for training
                train_set = pd.concat([train_set, data], ignore_index=True)
            else:  # Last 2 files for validation
                val_set = pd.concat([val_set, data], ignore_index=True)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Check if train_set and val_set are not empty
    if train_set.empty or val_set.empty:
        raise ValueError("Training or validation set is empty. Please check your CSV files.")

    # Extract features and labels
    print("Training set shape:", train_set.shape)
    print("Validation set shape:", val_set.shape)

    X_train = train_set.iloc[:, 1:].to_numpy(dtype=np.float32)  # Features
    Y_train = train_set.iloc[:, -1:].to_numpy(dtype=np.uint8)     # Classes

    X_val = val_set.iloc[:, 1:].to_numpy(dtype=np.float32)      # Features
    Y_val = val_set.iloc[:, -1:].to_numpy(dtype=np.uint8)

    return X_train, Y_train, X_val, Y_val