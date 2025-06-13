import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import os

# Function to split a CSV file into 5 folds using Stratified Group K-Fold
def split_csv_stratified_group_kfold(input_file="16000_25_0.6_hamming_none_MFCCS_F0_labeled.csv", 
                                   output_dir="split_data", 
                                   id_col="ID", 
                                   label_col="speech_label",
                                   train_ratio=0.8):
    """
    Split a CSV file into 5 folds using Stratified Group K-Fold cross-validation.
    Each fold will have an additional 'Split' column with 80% train and 20% test.
         
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
    train_ratio : float
        Ratio of training data (default: 0.8 for 80% train, 20% test)
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
        fold_df = df.iloc[test_idx].copy()
        
        # Add train/test split column
        np.random.seed(42 + fold)  # Different seed for each fold for variety
        n_samples = len(fold_df)
        n_train = int(n_samples * train_ratio)
        
        # Create split labels
        split_labels = ['Train'] * n_train + ['Test'] * (n_samples - n_train)
        np.random.shuffle(split_labels)
        
        # Add the Split column
        fold_df['Split'] = split_labels
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"fold_{fold}_with_splits.csv")
        fold_df.to_csv(output_file, index=False)
        
        # Print statistics
        train_count = sum(1 for x in split_labels if x == 'Train')
        test_count = sum(1 for x in split_labels if x == 'Test')
        
        print(f"Fold {fold} saved to {output_file}")
        print(f"  Total samples: {len(fold_df)}")
        print(f"  Train samples: {train_count} ({train_count/len(fold_df)*100:.1f}%)")
        print(f"  Test samples: {test_count} ({test_count/len(fold_df)*100:.1f}%)")
                
        # Print class distribution in each fold
        if label_col in df.columns:
            print(f"  Overall class distribution in fold {fold}:")
            class_dist = fold_df[label_col].value_counts(normalize=True)
            for class_name, proportion in class_dist.items():
                print(f"    {class_name}: {proportion:.3f}")
            
            # Print class distribution by split
            print(f"  Class distribution by split:")
            for split_type in ['train', 'test']:
                split_data = fold_df[fold_df['Split'] == split_type]
                if len(split_data) > 0:
                    split_class_dist = split_data[label_col].value_counts(normalize=True)
                    print(f"    {split_type.capitalize()}:")
                    for class_name, proportion in split_class_dist.items():
                        print(f"      {class_name}: {proportion:.3f}")
            print()

if __name__ == "__main__":
    # Example usage
    import argparse
        
    parser = argparse.ArgumentParser(description="Split CSV into 5 folds using Stratified Group K-Fold with train/test splits")
    parser.add_argument("--input_file", default='16000_25_0.6_hamming_none_MFCCS_F0_labeled.csv', 
                       help="Path to the input CSV file")
    parser.add_argument("--output-dir", default="split_data", 
                       help="Directory to save the output CSV files")
    parser.add_argument("--id-col", default="ID", 
                       help="Column name for the group identifier")
    parser.add_argument("--label-col", default="speech_label", 
                       help="Column name for the class labels")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of training data (default: 0.8 for 80%% train)")
        
    args = parser.parse_args()
        
    split_csv_stratified_group_kfold(
        args.input_file, 
        output_dir=args.output_dir,
        id_col=args.id_col,
        label_col=args.label_col,
        train_ratio=args.train_ratio
    )