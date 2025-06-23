import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnx2tf
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import json

class ModelFromWeights(nn.Module):
    """Reconstruct PyTorch model from JSON weights"""
    
    def __init__(self, weights_dict):
        super(ModelFromWeights, self).__init__()
        
        self.layers = nn.ModuleList()
        layer_configs = self._parse_weights(weights_dict)
        
        # Build the network architecture
        for i, config in enumerate(layer_configs):
            if config['type'] == 'linear':
                layer = nn.Linear(config['input_size'], config['output_size'])
                # Load the weights
                layer.weight.data = torch.tensor(config['weight_data'], dtype=torch.float32)
                layer.bias.data = torch.tensor(config['bias_data'], dtype=torch.float32)
                self.layers.append(layer)
        
        print(f"Reconstructed model with {len(self.layers)} layers")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i}: {layer}")
    
    def _parse_weights(self, weights_dict):
        """Parse weights dictionary to extract layer configurations"""
        layers = {}
        
        # Group weights and biases by layer
        for name, data in weights_dict.items():
            if isinstance(data, dict) and 'data' in data:
                # Extract layer info from name (e.g., "layers.0.weight")
                parts = name.split('.')
                if len(parts) >= 3:
                    layer_name = f"{parts[0]}.{parts[1]}"  # e.g., "layers.0"
                    param_type = parts[2]  # "weight" or "bias"
                    
                    if layer_name not in layers:
                        layers[layer_name] = {}
                    
                    layers[layer_name][param_type] = {
                        'data': data['data'],
                        'shape': data['shape']
                    }
        
        # Convert to layer configurations
        layer_configs = []
        for layer_name in sorted(layers.keys()):
            layer_data = layers[layer_name]
            
            if 'weight' in layer_data and 'bias' in layer_data:
                weight_shape = layer_data['weight']['shape']
                bias_shape = layer_data['bias']['shape']
                
                config = {
                    'name': layer_name,
                    'type': 'linear',
                    'output_size': weight_shape[0],
                    'input_size': weight_shape[1],
                    'weight_data': layer_data['weight']['data'],
                    'bias_data': layer_data['bias']['data']
                }
                layer_configs.append(config)
        
        return layer_configs
    
    def forward(self, x):
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU activation to all layers except the last one
            if i < len(self.layers) - 1:
                x = torch.relu(x)
            else:
                # Apply sigmoid to output layer (for VAD binary classification)
                x = torch.sigmoid(x)
        return x

def convert_pytorch_to_onnx(weights_dict, output_path="model.onnx", input_size=15):
    """Convert PyTorch model to ONNX format"""
    
    # Reconstruct the model from weights
    model = ModelFromWeights(weights_dict)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Test the model
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output: {test_output.numpy()}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Successfully exported to ONNX: {output_path}")
    return output_path

def convert_onnx_to_tensorflow(onnx_path, output_dir="tf_model"):
    """Convert ONNX model to TensorFlow SavedModel format"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Convert ONNX to TensorFlow
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=str(output_dir),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        
        print(f"✅ Successfully converted to TensorFlow: {output_dir}")
        
        # Verify TensorFlow model
        saved_model_path = output_dir / "saved_model"
        if saved_model_path.exists():
            model = tf.saved_model.load(str(saved_model_path))
            print("TensorFlow model loaded successfully")
            
            # Test inference
            dummy_input = tf.constant(np.random.randn(1, 15).astype(np.float32))
            if hasattr(model, 'signatures'):
                signature = list(model.signatures.keys())[0]
                output = model.signatures[signature](dummy_input)
                print(f"TensorFlow test output: {output}")
        
        return str(saved_model_path)
    
    except Exception as e:
        print(f"❌ Error converting to TensorFlow: {e}")
        print("Trying alternative conversion method...")
        
        # Alternative: Load ONNX and manually convert
        import onnxruntime as ort
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"ONNX model input: {input_info.name}, shape: {input_info.shape}")
        print(f"ONNX model output: {output_info.name}, shape: {output_info.shape}")
        
        # Create TensorFlow model wrapper
        class ONNXWrapper:
            def __init__(self, onnx_path):
                self.session = ort.InferenceSession(onnx_path)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            
            def __call__(self, x):
                if isinstance(x, tf.Tensor):
                    x = x.numpy()
                result = self.session.run([self.output_name], {self.input_name: x})
                return tf.constant(result[0])
        
        # Save as TensorFlow model
        wrapper = ONNXWrapper(onnx_path)
        
        # Create concrete function
        @tf.function
        def inference_func(x):
            return tf.py_function(wrapper, [x], tf.float32)
        
        # Save the model
        tf.saved_model.save(
            inference_func,
            str(output_dir / "saved_model"),
            signatures=inference_func.get_concrete_function(
                tf.TensorSpec(shape=[None, 15], dtype=tf.float32)
            )
        )
        
        return str(output_dir / "saved_model")

def create_frozen_graph(saved_model_path, output_path="model.pb"):
    """Convert SavedModel to frozen graph (.pb file)"""
    
    try:
        # Load the SavedModel
        model = tf.saved_model.load(saved_model_path)
        
        # Get the concrete function
        if hasattr(model, 'signatures'):
            concrete_func = list(model.signatures.values())[0]
        else:
            # For models without signatures, try to get the function directly
            concrete_func = model.inference_func if hasattr(model, 'inference_func') else None
        
        if concrete_func is None:
            raise ValueError("Could not find inference function in SavedModel")
        
        # Convert to frozen graph
        frozen_func = tf.function(concrete_func).get_concrete_function()
        frozen_graph = tf.graph_util.convert_variables_to_constants_v2(frozen_func)
        
        # Write the frozen graph
        tf.io.write_graph(
            graph_or_graph_def=frozen_graph.graph,
            logdir=str(Path(output_path).parent),
            name=Path(output_path).name,
            as_text=False
        )
        
        print(f"✅ Created frozen graph: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ Error creating frozen graph: {e}")