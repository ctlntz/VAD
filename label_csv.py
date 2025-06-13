import pandas as pd
import os
import sys

def add_speech_label(input_file, output_file=None):
    """
    Add a speech_label column to CSV based on ID column patterns.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    
    Returns:
        pandas.DataFrame: Modified DataFrame with speech_label column
    """
    
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if 'ID' column exists
        if 'ID' not in df.columns:
            print("Error: 'ID' column not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Create speech_label column based on ID patterns
        def classify_speech(ID_value):
            if pd.isna(ID_value):
                return None
            
            ID_str = str(ID_value).lower()
            if ID_str.endswith('_non_speech'):
                return 0
            else:
                return 1
        
        # Apply the classification
        df['speech_label'] = df['ID'].apply(classify_speech)
        
        # Print summary statistics
        print("\nSummary:")
        print(f"Total rows: {len(df)}")
        print(f"Speech samples (label=1): {sum(df['speech_label'] == 1)}")
        print(f"Non-speech samples (label=0): {sum(df['speech_label'] == 0)}")
        print(f"Unclassified samples: {sum(df['speech_label'].isna())}")
        
        # Show sample of the data
        print("\nSample of processed data:")
        print(df[['ID', 'speech_label']].head(10))
        
        # Save to output file
        if output_file is None:
            # Create output filename by adding '_labeled' before file extension
            base_name = os.path.splitext(input_file)[0]
            extension = os.path.splitext(input_file)[1]
            output_file = f"{base_name}_labeled{extension}"
        
        df.to_csv(output_file, index=False)
        print(f"\nProcessed CSV saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def main():
    """
    Main function to handle command line arguments or interactive input.
    """
    
    # Check if file path is provIDed as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Interactive mode
        input_file = "16000_25_0.6_hamming_none_MFCCS_F0.csv"
        output_file = "16000_25_0.6_hamming_none_MFCCS_F0_labeled.csv"
        if not output_file:
            output_file = None
    
    # Process the file
    if os.path.exists(input_file):
        result_df = add_speech_label(input_file, output_file)
        
        if result_df is not None:
            print("\nProcessing completed successfully!")
        else:
            print("Processing failed!")
    else:
        print(f"Error: File '{input_file}' does not exist.")

# Example usage for batch processing multiple files
def batch_process(input_directory, output_directory=None):
    """
    Process multiple CSV files in a directory.
    
    Args:
        input_directory (str): Directory containing CSV files
        output_directory (str): Directory to save processed files (optional)
    """
    
    if output_directory is None:
        output_directory = input_directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Process each file
    for csv_file in csv_files:
        input_path = os.path.join(input_directory, csv_file)
        output_path = os.path.join(output_directory, f"{os.path.splitext(csv_file)[0]}_labeled.csv")
        
        print(f"\nProcessing: {csv_file}")
        add_speech_label(input_path, output_path)

if __name__ == "__main__":
    main()