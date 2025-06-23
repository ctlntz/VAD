import tensorflow as tf
import numpy as np
import h5py
import json
import argparse
from pathlib import Path
import os
import subprocess
import sys

def create_calibration_dataset(num_samples=1000, input_size=15, output_path="calibration_data.h5"):
    """
    Create calibration dataset for quantization
    
    Args:
        num_samples (int): Number of calibration samples
        input_size (int): Input feature size
        output_path (str): Path to save calibration dataset
    """
    
    print(f"Creating calibration dataset with {num_samples} samples...")
    
    # Generate realistic calibration data
    # For VAD (Voice Activity Detection), create varied audio feature patterns
    calibration_data = []
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Generating sample {i}/{num_samples}")
        
        # Create diverse patterns for VAD features
        sample_type = i % 4
        
        if sample_type == 0:
            # Silence pattern (low energy)
            sample = np.random.normal(0, 0.1, input_size)
        elif sample_type == 1:
            # Speech pattern (higher energy, more variation)
            sample = np.random.normal(0.5, 0.3, input_size)
        elif sample_type == 2:
            # Noise pattern (random high frequency)
            sample = np.random.normal(0, 0.5, input_size)
        else:
            # Mixed pattern
            sample = np.random.normal(0.2, 0.4, input_size)
        
        # Normalize to typical range
        sample = np.clip(sample, -3.0, 3.0)
        calibration_data.append(sample)
    
    calibration_data = np.array(calibration_data, dtype=np.float32)
    
    # Save as HDF5 file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('input', data=calibration_data)
        f.create_dataset('labels', data=np.random.randint(0, 2, num_samples))  # Binary labels for VAD
    
    print(f"✅ Calibration dataset saved: {output_path}")
    print(f"   Shape: {calibration_data.shape}")
    print(f"   Data range: [{calibration_data.min():.3f}, {calibration_data.max():.3f}]")
    
    return output_path

def create_quantization_script(model_path, calibration_path, output_dir="quantized_model"):
    """
    Create quantization script for Vitis AI
    
    Args:
        model_path (str): Path to the frozen graph (.pb file)
        calibration_path (str): Path to calibration dataset
        output_dir (str): Output directory for quantized model
    """
    
    script_path = Path("quantize_model.py")
    
    script_content = f'''#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import h5py
import os
import sys
from pathlib import Path

# Add Vitis AI quantizer to path
sys.path.append('/opt/vitis_ai/compiler/vai_q_tensorflow2.x')

try:
    from tensorflow_model_optimization.quantization.keras import vitis_quantize
    print("Using Vitis AI quantizer")
except ImportError:
    print("Vitis AI quantizer not found, using TensorFlow Lite quantization")
    import tensorflow as tf

def load_calibration_data(calib_path):
    """Load calibration dataset"""
    print(f"Loading calibration data from: {{calib_path}}")
    
    with h5py.File(calib_path, 'r') as f:
        data = f['input'][:]
        print(f"Loaded calibration data shape: {{data.shape}}")
        return data

def quantize_model_vitis(model_path, calib_data, output_dir):
    """Quantize model using Vitis AI quantizer"""
    
    try:
        # Load the frozen graph
        print(f"Loading model from: {{model_path}}")
        
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        # Create quantization config
        quantize_config = {{
            'input_nodes': ['input'],
            'output_nodes': ['output'],
            'input_shapes': [1, {input_size}],
            'calib_iter': 100,
            'method': 1,  # 1 for non-overflow method
            'output_dir': output_dir
        }}
        
        # Run quantization
        print("Starting quantization...")
        
        # Import vai_q_tensorflow2 (Vitis AI quantizer)
        import vai_q_tensorflow2 as vai_q
        
        # Run quantization
        vai_q.quantize_frozen_graph(
            input_frozen_graph=model_path,
            input_nodes=['input'],
            output_nodes=['output'],
            input_shapes=[1, {input_size}],
            calib_dataset=calib_data,
            output_dir=output_dir
        )
        
        print(f"✅ Quantization completed! Output: {{output_dir}}")
        return True
        
    except ImportError as e:
        print(f"Vitis AI quantizer not available: {{e}}")
        return False
    except Exception as e:
        print(f"Error during Vitis AI quantization: {{e}}")
        return False

def quantize_model_tflite(model_path, calib_data, output_dir):
    """Fallback quantization using TensorFlow Lite"""
    
    print("Using TensorFlow Lite quantization as fallback...")
    
    try:
        # Load the model
        if model_path.endswith('.pb'):
            # Load frozen graph
            with tf.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file=model_path,
                input_arrays=['input'],
                output_arrays=['output'],
                input_shapes={{'input': [1, {input_size}]}}
            )
        else:
            # Load SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        
        # Set quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: (
            [calib_data[i:i+1].astype(np.float32) for i in range(0, len(calib_data), 10)]
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Convert
        tflite_model = converter.convert()
        
        # Save quantized model
        os.makedirs(output_dir, exist_ok=True)
        tflite_path = os.path.join(output_dir, 'quantized_model.tflite')
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite quantization completed: {{tflite_path}}")
        return True
        
    except Exception as e:
        print(f"❌ Error during TensorFlow Lite quantization: {{e}}")
        return False

def main():
    model_path = "{model_path}"
    calib_path = "{calibration_path}"
    output_dir = "{output_dir}"
    
    # Load calibration data
    calib_data = load_calibration_data(calib_path)
    
    # Try Vitis AI quantization first
    success = quantize_model_vitis(model_path, calib_data, output_dir)
    
    if not success:
        print("Falling back to TensorFlow Lite quantization...")
        success = quantize_model_tflite(model_path, calib_data, output_dir)
    
    if success:
        print("\\n🎉 Model quantization completed successfully!")
        print(f"📁 Quantized model saved in: {{output_dir}}")
    else:
        print("❌ Quantization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    
    print(f"✅ Created quantization script: {script_path}")
    return script_path

def create_vai_q_command(model_path, calibration_path, output_dir="quantized_model", input_size=15):
    """
    Create vai_q_tensorflow2 command for direct quantization
    """
    
    command_script = Path("run_quantization.sh")
    
    command_content = f'''#!/bin/bash

# Vitis AI Quantization Script
# Make sure you're in the Vitis AI conda environment:
# conda activate vitis-ai-tensorflow2

echo "Starting Vitis AI quantization..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=""

# Run quantization using vai_q_tensorflow2
python -m vai_q_tensorflow2.vai_q_tensorflow2 \\
    --model_type=frozen_pb \\
    --model_path="{model_path}" \\
    --input_nodes="input" \\
    --output_nodes="output" \\
    --input_shapes="1,{input_size}" \\
    --calib_dataset="{calibration_path}" \\
    --calib_batch_size=10 \\
    --calib_iter=100 \\
    --output_dir="{output_dir}" \\
    --method=1 \\
    --gpu=0

echo "Quantization completed!"
echo "Output directory: {output_dir}"
echo "Quantized model: {output_dir}/quantize_result.pb"
'''
    
    with open(command_script, 'w') as f:
        f.write(command_content)
    
    command_script.chmod(0o755)
    
    print(f"✅ Created quantization command script: {command_script}")
    return command_script

def verify_quantized_model(quantized_model_path, test_data_path):
    """
    Verify the quantized model works correctly
    """
    
    print(f"Verifying quantized model: {quantized_model_path}")
    
    try:
        # Load test data
        with h5py.File(test_data_path, 'r') as f:
            test_data = f['input'][:10]  # Use first 10 samples
        
        if quantized_model_path.endswith('.tflite'):
            # TensorFlow Lite model
            interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print("Testing TensorFlow Lite quantized model...")
            for i, sample in enumerate(test_data):
                interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                print(f"Sample {i}: Output = {output[0]}")
        
        elif quantized_model_path.endswith('.pb'):
            # Frozen graph
            with tf.gfile.GFile(quantized_model_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            
            with tf.Session() as sess:
                tf.import_graph_def(graph_def, name="")
                
                input_tensor = sess.graph.get_tensor_by_name("input:0")
                output_tensor = sess.graph.get_tensor_by_name("output:0")
                
                print("Testing frozen graph quantized model...")
                for i, sample in enumerate(test_data):
                    output = sess.run(output_tensor, {input_tensor: sample.reshape(1, -1)})
                    print(f"Sample {i}: Output = {output[0]}")
        
        print("✅ Quantized model verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying quantized model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quantize model for Vitis AI deployment")
    parser.add_argument("model_path", help="Path to frozen graph (.pb file)")
    parser.add_argument("-c", "--calibration", help="Path to calibration dataset (HDF5)")
    parser.add_argument("-o", "--output", default="quantized_model", help="Output directory")
    parser.add_argument("--input-size", type=int, default=15, help="Model input size")
    parser.add_argument("--samples", type=int, default=1000, help="Number of calibration samples")
    parser.add_argument("--create-calib", action="store_true", help="Create calibration dataset")
    parser.add_argument("--verify", action="store_true", help="Verify quantized model")
    
    args = parser.parse_args()
    
    try:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Create calibration dataset if requested
        if args.create_calib or not args.calibration:
            calib_path = "calibration_data.h5"
            create_calibration_dataset(args.samples, args.input_size, calib_path)
            args.calibration = calib_path
        
        # Create quantization scripts
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Model: {model_path}")
        print(f"Calibration: {args.calibration}")
        print(f"Output: {output_dir}")
        
        # Create Python quantization script
        create_quantization_script(str(model_path), args.calibration, str(output_dir))
        
        # Create bash command script
        create_vai_q_command(str(model_path), args.calibration, str(output_dir), args.input_size)
        
        print(f"\\n📋 Next steps:")
        print(f"1. Make sure you're in Vitis AI environment:")
        print(f"   conda activate vitis-ai-tensorflow2")
        print(f"2. Run quantization:")
        print(f"   python quantize_model.py")
        print(f"   OR")
        print(f"   ./run_quantization.sh")
        
        # Verify if requested
        if args.verify:
            quantized_pb = output_dir / "quantize_result.pb"
            if quantized_pb.exists():
                verify_quantized_model(str(quantized_pb), args.calibration)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()