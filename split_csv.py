import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import os

# Function to split a CSV file into 5 folds using Stratified Group K-Fold
def split_csv_stratified_group_kfold(input_file, output_dir="split_data", id_col="ID", label_col="speech_label"):
    """
    Split a CSV file into 5 folds using Stratified Group K-Fold cross-validation.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_dir : str
        Directory to save the output CSV files (default: "split_data")
    id_col : str
        Column name for the group identifier (default: "ID")
    label_col : str
        Column name for the class labels (default: "speech_label")
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Extract the required columns
    X = df.drop(columns=[label_col]) if label_col in df.columns else df
    y = df[label_col] if label_col in df.columns else np.zeros(len(df))
    groups = df[id_col]
    
    # Initialize the StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Split the data and save to separate CSV files
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_df = df.iloc[test_idx]
        output_file = os.path.join(output_dir, f"fold_{fold}.csv")
        fold_df.to_csv(output_file, index=False)
        print(f"Fold {fold} saved to {output_file} with {len(fold_df)} samples")
        
        # Print class distribution in each fold
        if label_col in df.columns:
            class_dist = fold_df[label_col].value_counts(normalize=True)
            print(f"Class distribution in fold {fold}:")
            print(class_dist)
            print()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Split CSV into 5 folds using Stratified Group K-Fold")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output-dir", default="split_data", help="Directory to save the output CSV files")
    parser.add_argument("--id-col", default="ID", help="Column name for the group identifier")
    parser.add_argument("--label-col", default="speech_label", help="Column name for the class labels")
    
    args = parser.parse_args()
    
    split_csv_stratified_group_kfold(
        args.input_file, 
        output_dir=args.output_dir,
        id_col=args.id_col,
        label_col=args.label_col
    )