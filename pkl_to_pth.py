import pickle
import torch
import os
import argparse
from pathlib import Path

def convert_pkl_to_pth(pkl_path, pth_path=None):
    """
    Convert a PKL file to PTH format
    
    Args:
        pkl_path (str): Path to the input PKL file
        pth_path (str, optional): Path for the output PTH file. If None, uses same name with .pth extension
    """
    pkl_path = Path(pkl_path)
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")
    
    # Set output path if not provided
    if pth_path is None:
        pth_path = pkl_path.with_suffix('.pth')
    else:
        pth_path = Path(pth_path)
    
    print(f"Loading PKL file: {pkl_path}")
    
    try:
        # Load the pickle file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded PKL file")
        print(f"Data type: {type(data)}")
        
        # Handle different data types
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys: {list(data.keys())}")
            # Convert numpy arrays to tensors if present, keep tensors as-is
            converted_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    # Already a PyTorch tensor, keep as-is
                    converted_data[key] = value
                elif hasattr(value, 'shape') and hasattr(value, 'dtype'):  # numpy array
                    converted_data[key] = torch.from_numpy(value)
                else:
                    converted_data[key] = value
            data = converted_data
        
        elif hasattr(data, 'state_dict'):  # PyTorch model
            print("Detected PyTorch model, extracting state_dict")
            data = data.state_dict()
        
        elif hasattr(data, 'shape'):  # numpy array or tensor
            print(f"Single array/tensor with shape: {data.shape}")
            if isinstance(data, torch.Tensor):
                # Already a tensor, keep as-is
                pass
            else:
                # Convert numpy array to tensor
                data = torch.from_numpy(data)
        
        # Save as PTH file
        print(f"Saving to PTH file: {pth_path}")
        torch.save(data, pth_path)
        
        print(f"✅ Successfully converted {pkl_path} to {pth_path}")
        
        # Print file sizes for comparison
        pkl_size = pkl_path.stat().st_size / (1024*1024)  # MB
        pth_size = pth_path.stat().st_size / (1024*1024)  # MB
        print(f"PKL file size: {pkl_size:.2f} MB")
        print(f"PTH file size: {pth_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert PKL files to PTH format")
    parser.add_argument("input", help="Input PKL file path")
    parser.add_argument("-o", "--output", help="Output PTH file path (optional)")
    parser.add_argument("--inspect", action="store_true", help="Inspect PKL contents without conversion")
    
    args = parser.parse_args()
    
    if args.inspect:
        # Just inspect the PKL file
        try:
            with open(args.input, 'rb') as f:
                data = pickle.load(f)
            
            print(f"PKL file contents:")
            print(f"Type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
                for key, value in data.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
            elif hasattr(data, 'shape'):
                print(f"Shape: {data.shape}")
                print(f"Dtype: {data.dtype}")
            
        except Exception as e:
            print(f"Error inspecting PKL file: {e}")
    else:
        # Convert the file
        convert_pkl_to_pth(args.input, args.output)

if __name__ == "__main__":
    main()