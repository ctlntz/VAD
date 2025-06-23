import json
import numpy as np
import argparse
from pathlib import Path
import re

def sanitize_name(name):
    """Convert layer names to valid C identifiers"""
    # Replace dots and other special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter or underscore
    if name[0].isdigit():
        name = '_' + name
    return name

def generate_c_header(weights_dict, output_dir=".", model_name="model"):
    """
    Generate C header files from JSON weights
    
    Args:
        weights_dict (dict): Dictionary containing weights from JSON
        output_dir (str): Output directory for generated files
        model_name (str): Name prefix for generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    header_file = output_dir / f"{model_name}_weights.h"
    source_file = output_dir / f"{model_name}_weights.c"
    
    # Start generating header file
    header_content = f"""#ifndef {model_name.upper()}_WEIGHTS_H
#define {model_name.upper()}_WEIGHTS_H

#ifdef __cplusplus
extern "C" {{
#endif

// Model weights and biases
"""
    
    # Start generating source file
    source_content = f"""#include "{model_name}_weights.h"

// Model weights and biases implementation
"""
    
    layer_info = []
    
    for layer_name, layer_data in weights_dict.items():
        if isinstance(layer_data, dict) and 'data' in layer_data:
            data = layer_data['data']
            shape = layer_data.get('shape', [])
            dtype = layer_data.get('dtype', 'float32')
            
            # Sanitize layer name for C identifier
            c_name = sanitize_name(layer_name)
            
            # Flatten the data if it's multi-dimensional
            if isinstance(data[0], list):
                flat_data = []
                def flatten(lst):
                    for item in lst:
                        if isinstance(item, list):
                            flatten(item)
                        else:
                            flat_data.append(item)
                flatten(data)
                data = flat_data
            
            # Calculate total size
            total_size = len(data)
            
            # Generate C array declaration in header
            header_content += f"""
// {layer_name} - Shape: {shape}
#define {c_name.upper()}_SIZE {total_size}
extern const float {c_name}[{total_size}];
"""
            
            # Generate C array definition in source
            source_content += f"""
// {layer_name} - Shape: {shape}
const float {c_name}[{total_size}] = {{
"""
            
            # Add data values (16 per line for readability)
            for i, value in enumerate(data):
                if i % 16 == 0:
                    source_content += "    "
                
                # Format floating point numbers
                if isinstance(value, (int, float)):
                    source_content += f"{float(value):.8f}f"
                else:
                    source_content += f"{float(value):.8f}f"
                
                if i < len(data) - 1:
                    source_content += ", "
                
                if (i + 1) % 16 == 0 or i == len(data) - 1:
                    source_content += "\n"
            
            source_content += "};\n"
            
            # Store layer info for network structure
            layer_info.append({
                'name': layer_name,
                'c_name': c_name,
                'shape': shape,
                'size': total_size,
                'type': 'weight' if 'weight' in layer_name else 'bias'
            })
    
    # Add network structure information
    header_content += f"""
// Network structure information
#define NUM_LAYERS {len([l for l in layer_info if l['type'] == 'weight'])}

typedef struct {{
    const float* weights;
    const float* bias;
    int input_size;
    int output_size;
}} layer_t;

// Layer configuration
extern const layer_t network_layers[];
extern const int network_num_layers;

"""
    
    # Generate layer configuration in source file
    source_content += f"""
// Network layer configuration
const layer_t network_layers[] = {{
"""
    
    # Group weights and biases by layer
    layers = {}
    for info in layer_info:
        layer_base = info['name'].split('.')[0] + '.' + info['name'].split('.')[1]  # e.g., "layers.0"
        if layer_base not in layers:
            layers[layer_base] = {}
        
        if info['type'] == 'weight':
            layers[layer_base]['weight'] = info
        else:
            layers[layer_base]['bias'] = info
    
    for layer_base, layer_data in layers.items():
        weight_info = layer_data.get('weight')
        bias_info = layer_data.get('bias')
        
        if weight_info and bias_info:
            # Assuming weight shape is [output_size, input_size]
            if len(weight_info['shape']) >= 2:
                output_size = weight_info['shape'][0]
                input_size = weight_info['shape'][1]
            else:
                output_size = weight_info['shape'][0] if weight_info['shape'] else weight_info['size']
                input_size = 1
            
            source_content += f"    {{ {weight_info['c_name']}, {bias_info['c_name']}, {input_size}, {output_size} }},\n"
    
    source_content += f"""
}};

const int network_num_layers = sizeof(network_layers) / sizeof(layer_t);
"""
    
    # Close header file
    header_content += f"""
#ifdef __cplusplus
}}
#endif

#endif // {model_name.upper()}_WEIGHTS_H
"""
    
    # Write files
    with open(header_file, 'w') as f:
        f.write(header_content)
    
    with open(source_file, 'w') as f:
        f.write(source_content)
    
    print(f"✅ Generated C header file: {header_file}")
    print(f"✅ Generated C source file: {source_file}")
    
    return header_file, source_file, layer_info

def generate_inference_code(layer_info, output_dir=".", model_name="model"):
    """Generate basic inference code template"""
    
    inference_file = Path(output_dir) / f"{model_name}_inference.c"
    
    inference_content = f"""#include "{model_name}_weights.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Simple activation functions
float relu(float x) {{
    return (x > 0) ? x : 0;
}}

float sigmoid(float x) {{
    return 1.0f / (1.0f + expf(-x));
}}

// Forward pass for a single layer
void forward_layer(const float* input, float* output, const layer_t* layer) {{
    // Matrix multiplication: output = weights * input + bias
    for (int i = 0; i < layer->output_size; i++) {{
        float sum = 0.0f;
        for (int j = 0; j < layer->input_size; j++) {{
            sum += layer->weights[i * layer->input_size + j] * input[j];
        }}
        output[i] = sum + layer->bias[i];
    }}
}}

// Full network inference
void model_inference(const float* input, float* output, int input_size) {{
    static float layer_outputs[2][256]; // Adjust size based on your network
    int current_buffer = 0;
    
    // Copy input to first buffer
    memcpy(layer_outputs[current_buffer], input, input_size * sizeof(float));
    
    // Process each layer
    for (int layer_idx = 0; layer_idx < network_num_layers; layer_idx++) {{
        const layer_t* layer = &network_layers[layer_idx];
        int next_buffer = 1 - current_buffer;
        
        // Forward pass
        forward_layer(layer_outputs[current_buffer], layer_outputs[next_buffer], layer);
        
        // Apply activation (ReLU for hidden layers, sigmoid for output)
        if (layer_idx < network_num_layers - 1) {{
            // Hidden layer - apply ReLU
            for (int i = 0; i < layer->output_size; i++) {{
                layer_outputs[next_buffer][i] = relu(layer_outputs[next_buffer][i]);
            }}
        }} else {{
            // Output layer - apply sigmoid (or softmax for multi-class)
            for (int i = 0; i < layer->output_size; i++) {{
                layer_outputs[next_buffer][i] = sigmoid(layer_outputs[next_buffer][i]);
            }}
        }}
        
        current_buffer = next_buffer;
    }}
    
    // Copy final output
    const layer_t* final_layer = &network_layers[network_num_layers - 1];
    memcpy(output, layer_outputs[current_buffer], final_layer->output_size * sizeof(float));
}}

// Example usage
int main() {{
    // Example input (adjust size based on your model)
    float input[] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f}};
    float output[16]; // Adjust size based on your output layer
    
    // Run inference
    model_inference(input, output, 15); // Adjust input size
    
    // Print results
    printf("Model output:\\n");
    for (int i = 0; i < 1; i++) {{ // Adjust based on output size
        printf("Output[%d]: %.6f\\n", i, output[i]);
    }}
    
    return 0;
}}
"""
    
    with open(inference_file, 'w') as f:
        f.write(inference_content)
    
    print(f"✅ Generated inference code: {inference_file}")
    return inference_file

def main():
    parser = argparse.ArgumentParser(description="Convert JSON weights to C/H files for FPGA deployment")
    parser.add_argument("input", help="Input JSON file with weights")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("-n", "--name", default="model", help="Model name prefix")
    parser.add_argument("--inference", action="store_true", help="Generate inference code template")
    
    args = parser.parse_args()
    
    try:
        # Load JSON weights
        with open(args.input, 'r') as f:
            weights_dict = json.load(f)
        
        print(f"Loaded weights from: {args.input}")
        print(f"Found {len(weights_dict)} layers")
        
        # Generate C/H files
        header_file, source_file, layer_info = generate_c_header(weights_dict, args.output, args.name)
        
        # Generate inference code if requested
        if args.inference:
            inference_file = generate_inference_code(layer_info, args.output, args.name)
        
        # Print summary
        print(f"\n📊 Layer Summary:")
        for info in layer_info:
            print(f"  {info['name']}: {info['shape']} ({info['size']} parameters)")
        
        print(f"\n🚀 Ready for FPGA deployment!")
        print(f"   Include: #include \"{args.name}_weights.h\"")
        print(f"   Compile: gcc {source_file} {args.name}_inference.c -lm")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()