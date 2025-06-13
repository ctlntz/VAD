#!/usr/bin/env python3
"""
CNN Model Weight and Bias Extractor for Vitis AI with HLS
Supports PyTorch and TensorFlow/Keras pickled models
"""

import os
import pickle
import numpy as np
import argparse
from pathlib import Path

# Optional imports - will handle if not available
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - PyTorch models won't be supported")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - TensorFlow models won't be supported")

class CNNWeightExtractor:
    def __init__(self, model_path, output_dir="extracted_weights"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model = None
        self.framework = None
        
    def load_model(self):
        """Load the pickled model and detect framework"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            # Detect framework
            if PYTORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
                self.framework = 'pytorch'
                self.model.eval()
                print(f"Detected PyTorch model: {type(self.model).__name__}")
                
            elif TENSORFLOW_AVAILABLE and hasattr(self.model, 'layers'):
                self.framework = 'tensorflow'
                print(f"Detected TensorFlow/Keras model: {type(self.model).__name__}")
                
            else:
                print("Unknown model type. Attempting generic extraction...")
                self.framework = 'generic'
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
        return True
    
    def quantize_weights(self, weights, scale=127.0, dtype=np.int8):
        """Quantize weights to specified integer type"""
        if dtype == np.int8:
            max_val = 127
            min_val = -128
        elif dtype == np.int16:
            max_val = 32767
            min_val = -32768
        else:
            return weights
            
        quantized = np.round(weights * scale).astype(dtype)
        quantized = np.clip(quantized, min_val, max_val)
        return quantized
    
    def weights_to_c_header(self, weights, name, dtype="int8_t"):
        """Convert numpy array to C header format"""
        flat_weights = weights.flatten()
        header = f"// {name} - Shape: {weights.shape}\n"
        header += f"const {dtype} {name}[{len(flat_weights)}] = {{\n"
        
        for i, w in enumerate(flat_weights):
            if i % 16 == 0:
                header += "\n    "
            header += f"{w}, "
        
        header = header.rstrip(", ") + "\n};\n\n"
        return header
    
    def extract_pytorch_weights(self):
        """Extract weights from PyTorch model"""
        weights_dict = {}
        c_header_content = "#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n"
        
        print("Extracting PyTorch model weights...")
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_data = param.detach().cpu().numpy()
                weights_dict[name] = weight_data
                
                # Save as numpy file
                np.save(self.output_dir / f"{name.replace('.', '_')}.npy", weight_data)
                
                # Quantize for Vitis AI
                quantized = self.quantize_weights(weight_data)
                np.save(self.output_dir / f"{name.replace('.', '_')}_quantized.npy", quantized)
                
                # Generate C header
                c_name = name.replace('.', '_').replace('-', '_')
                c_header_content += self.weights_to_c_header(quantized, c_name)
                
                print(f"  {name}: {weight_data.shape} -> quantized to int8")
        
        # Extract layer-wise weights and biases
        layer_info = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                if hasattr(module, 'weight'):
                    weight = module.weight.detach().cpu().numpy()
                    layer_info[f"{name}_weight"] = weight
                    
                if hasattr(module, 'bias') and module.bias is not None:
                    bias = module.bias.detach().cpu().numpy()
                    layer_info[f"{name}_bias"] = bias
        
        # Save layer info
        with open(self.output_dir / "layer_info.txt", "w") as f:
            for layer_name, data in layer_info.items():
                f.write(f"{layer_name}: {data.shape}\n")
        
        c_header_content += "#endif // MODEL_WEIGHTS_H\n"
        with open(self.output_dir / "model_weights.h", "w") as f:
            f.write(c_header_content)
            
        return weights_dict
    
    def extract_tensorflow_weights(self):
        """Extract weights from TensorFlow/Keras model"""
        weights_dict = {}
        c_header_content = "#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n"
        
        print("Extracting TensorFlow/Keras model weights...")
        
        layer_info = {}
        for i, layer in enumerate(self.model.layers):
            layer_weights = layer.get_weights()
            if len(layer_weights) > 0:
                layer_name = layer.name or f"layer_{i}"
                
                # Extract weights
                weights = layer_weights[0]
                weights_dict[f"{layer_name}_weights"] = weights
                layer_info[f"{layer_name}_weights"] = weights.shape
                
                # Save weights
                np.save(self.output_dir / f"{layer_name}_weights.npy", weights)
                
                # Quantize and save
                quantized_weights = self.quantize_weights(weights)
                np.save(self.output_dir / f"{layer_name}_weights_quantized.npy", quantized_weights)
                
                # Generate C header for weights
                c_name = f"{layer_name}_weights".replace('-', '_').replace('.', '_')
                c_header_content += self.weights_to_c_header(quantized_weights, c_name)
                
                print(f"  {layer_name} weights: {weights.shape}")
                
                # Extract biases if present
                if len(layer_weights) > 1:
                    biases = layer_weights[1]
                    weights_dict[f"{layer_name}_biases"] = biases
                    layer_info[f"{layer_name}_biases"] = biases.shape
                    
                    # Save biases
                    np.save(self.output_dir / f"{layer_name}_biases.npy", biases)
                    
                    # Quantize and save
                    quantized_biases = self.quantize_weights(biases)
                    np.save(self.output_dir / f"{layer_name}_biases_quantized.npy", quantized_biases)
                    
                    # Generate C header for biases
                    c_name_bias = f"{layer_name}_biases".replace('-', '_').replace('.', '_')
                    c_header_content += self.weights_to_c_header(quantized_biases, c_name_bias)
                    
                    print(f"  {layer_name} biases: {biases.shape}")
        
        # Save layer info
        with open(self.output_dir / "layer_info.txt", "w") as f:
            for layer_name, shape in layer_info.items():
                f.write(f"{layer_name}: {shape}\n")
        
        c_header_content += "#endif // MODEL_WEIGHTS_H\n"
        with open(self.output_dir / "model_weights.h", "w") as f:
            f.write(c_header_content)
            
        return weights_dict
    
    def extract_intermediate_activations(self, input_data=None):
        """Extract intermediate node values for verification"""
        if self.framework == 'pytorch' and input_data is not None:
            activations = {}
            
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach().cpu().numpy()
                return hook
            
            # Register hooks
            hooks = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear)):
                    hook = module.register_forward_hook(get_activation(name))
                    hooks.append(hook)
            
            # Run inference
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data).float()
                else:
                    input_tensor = input_data
                    
                output = self.model(input_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Save activations
            for name, activation in activations.items():
                np.save(self.output_dir / f"activation_{name.replace('.', '_')}.npy", activation)
                print(f"  Saved activation {name}: {activation.shape}")
            
            return activations
        
        return None
    
    def generate_model_summary(self, weights_dict):
        """Generate a summary of the extracted model"""
        summary_path = self.output_dir / "model_summary.txt"
        
        with open(summary_path, "w") as f:
            f.write(f"Model Extraction Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Framework: {self.framework}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("Extracted Parameters:\n")
            f.write("-" * 30 + "\n")
            
            total_params = 0
            for name, weights in weights_dict.items():
                param_count = weights.size
                total_params += param_count
                f.write(f"{name:30s}: {str(weights.shape):15s} ({param_count:,} params)\n")
            
            f.write(f"\nTotal Parameters: {total_params:,}\n")
            
            # File listing
            f.write(f"\nGenerated Files:\n")
            f.write("-" * 20 + "\n")
            for file_path in sorted(self.output_dir.glob("*")):
                if file_path.is_file():
                    f.write(f"  {file_path.name}\n")
        
        print(f"Model summary saved to: {summary_path}")
    
    def run_extraction(self, extract_activations=False, sample_input=None):
        """Main extraction process"""
        print(f"Starting extraction for: {self.model_path}")
        
        if not self.load_model():
            return False
        
        # Extract weights based on framework
        if self.framework == 'pytorch':
            weights_dict = self.extract_pytorch_weights()
        elif self.framework == 'tensorflow':
            weights_dict = self.extract_tensorflow_weights()
        else:
            print("Unsupported model type for extraction")
            return False
        
        # Extract activations if requested
        if extract_activations and sample_input is not None:
            print("Extracting intermediate activations...")
            self.extract_intermediate_activations(sample_input)
        
        # Generate summary
        self.generate_model_summary(weights_dict)
        
        print(f"\nExtraction complete! Files saved to: {self.output_dir}")
        print("Generated files:")
        print("  - *.npy: Raw weights/biases")
        print("  - *_quantized.npy: Quantized weights (INT8)")
        print("  - model_weights.h: C header file for HLS")
        print("  - layer_info.txt: Layer information")
        print("  - model_summary.txt: Complete summary")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Extract CNN weights for Vitis AI HLS")
    parser.add_argument("model_path", help="Path to the pickled model file")
    parser.add_argument("-o", "--output", default="extracted_weights", 
                       help="Output directory (default: extracted_weights)")
    parser.add_argument("--extract-activations", action="store_true",
                       help="Extract intermediate activations")
    parser.add_argument("--input-shape", nargs="+", type=int,
                       help="Input shape for activation extraction (e.g., 1 3 224 224)")
    
    args = parser.parse_args()
    
    # Create sample input if shape provided
    sample_input = None
    if args.input_shape:
        sample_input = np.random.randn(*args.input_shape).astype(np.float32)
        print(f"Created sample input with shape: {sample_input.shape}")
    
    # Run extraction
    extractor = CNNWeightExtractor(args.model_path, args.output)
    success = extractor.run_extraction(
        extract_activations=args.extract_activations,
        sample_input=sample_input
    )
    
    if not success:
        print("Extraction failed!")
        exit(1)

if __name__ == "__main__":
    main()