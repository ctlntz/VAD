import os
import shutil

def rename_files_in_directory(directory, prefix):
    """
    Renames all files in the given directory with the format {i}_non_speech,
    where i is a counter starting from 1.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix to use for the counter (e.g., 'train' or 'test')
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    print(f"Found {len(files)} files in {directory}")
    
    # Rename each file
    for i, filename in enumerate(files, 1):
        old_path = os.path.join(directory, filename)
        
        # Get file extension
        _, file_extension = os.path.splitext(filename)
        
        # Create new filename
        
        new_filename = f"{i}{file_extension}"
        # new_filename = f"n{i}{file_extension}"
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        shutil.move(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

def main():
    # Get directories from user
    train_dir = 'database/timit'
    # test_dir = "database/test"
    
    # Rename files in train directory
    print("\nProcessing train directory...")
    rename_files_in_directory(train_dir, "train")
    
    # # Rename files in test directory
    # print("\nProcessing test directory...")
    # rename_files_in_directory(test_dir, "test")
    
    print("\nRenaming complete!")

if __name__ == "__main__":
    main()