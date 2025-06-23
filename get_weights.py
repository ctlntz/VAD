import torch
import numpy as np
import argparse
from pathlib import Path
import json

def get_weights_from_pth(pth_path, output_format='dict', save_path=None):
    """
    Extract weights from a PTH file
    
    Args:
        pth_path (str): Path to the PTH file
        output_format (str): Format for output ('dict', 'list', 'numpy', 'json')
        save_path (str, optional): Path to save extracted weights
    
    Returns:
        dict or list: Extracted weights
    """
    pth_path = Path(pth_path)
    
    if not pth_path.exists():
        raise FileNotFoundError(f"PTH file not found: {pth_path}")
    
    print(f"Loading PTH file: {pth_path}")
    
    try:
        # Load the PTH file
        data = torch.load(pth_path, map_location='cpu')
        
        print(f"Successfully loaded PTH file")
        print(f"Data type: {type(data)}")
        
        weights = {}
        
        if isinstance(data, dict):
            print(f"Found {len(data)} parameters/layers:")
            
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                    
                    # Convert to numpy for easier handling
                    if output_format == 'numpy':
                        weights[key] = value.detach().numpy()
                    elif output_format == 'list':
                        weights[key] = value.detach().numpy().tolist()
                    else:
                        weights[key] = value
                else:
                    print(f"  {key}: {type(value)} (non-tensor)")
                    weights[key] = value
        
        elif isinstance(data, torch.Tensor):
            print(f"Single tensor with shape: {data.shape}")
            if output_format == 'numpy':
                weights = data.detach().numpy()
            elif output_format == 'list':
                weights = data.detach().numpy().tolist()
            else:
                weights = data
        
        else:
            print(f"Unsupported data type: {type(data)}")
            weights = data
        
        # Save weights if requested
        if save_path:
            save_path = Path(save_path)
            
            if output_format == 'json':
                # Convert tensors to lists for JSON serialization
                json_weights = {}
                for key, value in weights.items():
                    if isinstance(value, torch.Tensor):
                        json_weights[key] = {
                            'data': value.detach().numpy().tolist(),
                            'shape': list(value.shape),
                            'dtype': str(value.dtype)
                        }
                    elif isinstance(value, np.ndarray):
                        json_weights[key] = {
                            'data': value.tolist(),
                            'shape': list(value.shape),
                            'dtype': str(value.dtype)
                        }
                    else:
                        json_weights[key] = value
                
                with open(save_path.with_suffix('.json'), 'w') as f:
                    json.dump(json_weights, f, indent=2)
                print(f"✅ Saved weights as JSON to: {save_path.with_suffix('.json')}")
            
            elif output_format == 'numpy':
                np.savez(save_path.with_suffix('.npz'), **weights)
                print(f"✅ Saved weights as NPZ to: {save_path.with_suffix('.npz')}")
            
            else:
                torch.save(weights, save_path.with_suffix('.pth'))
                print(f"✅ Saved weights as PTH to: {save_path.with_suffix('.pth')}")
        
        return weights
    
    except Exception as e:
        print(f"❌ Error extracting weights: {str(e)}")
        raise

def analyze_weights(weights):
    """Analyze the weights and provide statistics"""
    print("\n📊 Weight Analysis:")
    print("=" * 50)
    
    if isinstance(weights, dict):
        total_params = 0
        
        for key, value in weights.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                num_params = np.prod(value.shape)
                total_params += num_params
                
                print(f"\n{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Parameters: {num_params:,}")
                print(f"  Mean: {np.mean(value.detach().numpy() if isinstance(value, torch.Tensor) else value):.6f}")
                print(f"  Std: {np.std(value.detach().numpy() if isinstance(value, torch.Tensor) else value):.6f}")
                print(f"  Min: {np.min(value.detach().numpy() if isinstance(value, torch.Tensor) else value):.6f}")
                print(f"  Max: {np.max(value.detach().numpy() if isinstance(value, torch.Tensor) else value):.6f}")
        
        print(f"\n🔢 Total Parameters: {total_params:,}")
        print(f"💾 Estimated Memory: {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    elif isinstance(weights, (torch.Tensor, np.ndarray)):
        num_params = np.prod(weights.shape)
        print(f"Single tensor/array:")
        print(f"  Shape: {weights.shape}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Mean: {np.mean(weights.detach().numpy() if isinstance(weights, torch.Tensor) else weights):.6f}")
        print(f"  Std: {np.std(weights.detach().numpy() if isinstance(weights, torch.Tensor) else weights):.6f}")

def main():
    parser = argparse.ArgumentParser(description="Extract weights from PTH files")
    parser.add_argument("input", help="Input PTH file path")
    parser.add_argument("-f", "--format", choices=['dict', 'list', 'numpy', 'json'], 
                       default='dict', help="Output format")
    parser.add_argument("-o", "--output", help="Output file path to save weights")
    parser.add_argument("--analyze", action="store_true", help="Analyze weights statistics")
    parser.add_argument("--layer", help="Extract specific layer/parameter by name")
    
    args = parser.parse_args()
    
    try:
        weights = get_weights_from_pth(args.input, args.format, args.output)
        
        # Extract specific layer if requested
        if args.layer:
            if isinstance(weights, dict) and args.layer in weights:
                weights = {args.layer: weights[args.layer]}
                print(f"\n🎯 Extracted layer: {args.layer}")
            else:
                print(f"❌ Layer '{args.layer}' not found")
                if isinstance(weights, dict):
                    print(f"Available layers: {list(weights.keys())}")
                return
        
        # Analyze weights if requested
        if args.analyze:
            analyze_weights(weights)
        
        print(f"\n✅ Successfully extracted weights from {args.input}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()