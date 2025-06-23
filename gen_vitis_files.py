import json
import argparse
from pathlib import Path

def generate_vitis_ai_files(weights_dict, output_dir=".", model_name="vad_model"):
    """
    Generate Vitis AI specific files for FPGA deployment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate DPU-optimized inference code
    dpu_inference_file = output_dir / f"{model_name}_dpu.cpp"
    
    dpu_content = f"""#include <iostream>
#include <vector>
#include <memory>
#include <vart/runner.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/graph/graph.hpp>

class {model_name.upper()}Runner {{
private:
    std::unique_ptr<vart::Runner> runner_;
    std::vector<const xir::Tensor*> input_tensors_;
    std::vector<const xir::Tensor*> output_tensors_;
    
public:
    {model_name.upper()}Runner(const std::string& model_path) {{
        // Load the compiled DPU model
        auto graph = xir::Graph::deserialize(model_path);
        auto subgraph = get_dpu_subgraph(graph.get());
        runner_ = vart::Runner::create_runner(subgraph[0], "run");
        
        // Get input and output tensor specifications
        input_tensors_ = runner_->get_input_tensors();
        output_tensors_ = runner_->get_output_tensors();
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Input tensors: " << input_tensors_.size() << std::endl;
        std::cout << "Output tensors: " << output_tensors_.size() << std::endl;
    }}
    
    std::vector<float> predict(const std::vector<float>& input) {{
        // Create input tensor buffers
        auto input_tensor_buffers = std::vector<std::unique_ptr<vart::TensorBuffer>>{{}};
        auto output_tensor_buffers = std::vector<std::unique_ptr<vart::TensorBuffer>>{{}};
        
        for (auto& tensor : input_tensors_) {{
            input_tensor_buffers.push_back(
                std::make_unique<vart::TensorBuffer>(tensor, nullptr));
        }}
        
        for (auto& tensor : output_tensors_) {{
            output_tensor_buffers.push_back(
                std::make_unique<vart::TensorBuffer>(tensor, nullptr));
        }}
        
        // Copy input data
        auto input_data = reinterpret_cast<float*>(
            input_tensor_buffers[0]->data({{0, 0}}).first);
        std::copy(input.begin(), input.end(), input_data);
        
        // Run inference
        auto v_input_tensor_buffers = std::vector<vart::TensorBuffer*>{{}};
        auto v_output_tensor_buffers = std::vector<vart::TensorBuffer*>{{}};
        
        for (auto& tb : input_tensor_buffers) {{
            v_input_tensor_buffers.push_back(tb.get());
        }}
        for (auto& tb : output_tensor_buffers) {{
            v_output_tensor_buffers.push_back(tb.get());
        }}
        
        auto job_id = runner_->execute_async(v_input_tensor_buffers, v_output_tensor_buffers);
        runner_->wait(job_id.first, -1);
        
        // Get output data
        auto output_data = reinterpret_cast<float*>(
            output_tensor_buffers[0]->data({{0, 0}}).first);
        auto output_size = output_tensors_[0]->get_element_num();
        
        return std::vector<float>(output_data, output_data + output_size);
    }}
    
private:
    std::vector<const xir::Subgraph*> get_dpu_subgraph(const xir::Graph* graph) {{
        auto root = graph->get_root_subgraph();
        auto children = root->children_topological_sort();
        std::vector<const xir::Subgraph*> dpu_subgraphs;
        for (auto& child : children) {{
            if (child->get_attr<std::string>("device") == "DPU") {{
                dpu_subgraphs.push_back(child);
            }}
        }}
        return dpu_subgraphs;
    }}
}};

// Example usage
int main(int argc, char** argv) {{
    if (argc != 2) {{
        std::cerr << "Usage: " << argv[0] << " <model.xmodel>" << std::endl;
        return -1;
    }}
    
    try {{
        // Initialize the model
        {model_name.upper()}Runner model(argv[1]);
        
        // Example input (adjust based on your model)
        std::vector<float> input = {{
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        }};
        
        // Run inference
        auto output = model.predict(input);
        
        // Print results
        std::cout << "Inference results:" << std::endl;
        for (size_t i = 0; i < output.size(); ++i) {{
            std::cout << "Output[" << i << "]: " << output[i] << std::endl;
        }}
        
    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }}
    
    return 0;
}}
"""
    
    with open(dpu_inference_file, 'w') as f:
        f.write(dpu_content)
    
    # Generate CMakeLists.txt for building
    cmake_file = output_dir / "CMakeLists.txt"
    cmake_content = f"""cmake_minimum_required(VERSION 3.12)
project({model_name}_dpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

# Vitis AI includes and libraries
set(VITIS_AI_ROOT $ENV{{CONDA_PREFIX}})
include_directories(${{VITIS_AI_ROOT}}/include)
link_directories(${{VITIS_AI_ROOT}}/lib)

# Add executable
add_executable({model_name}_dpu {model_name}_dpu.cpp)

# Link libraries
target_link_libraries({model_name}_dpu
    xir
    vart-runner
    vart-util
    vart-buffer-object
    ${{GLOG_LIBRARIES}}
)

target_include_directories({model_name}_dpu PRIVATE ${{GLOG_INCLUDE_DIRS}})
target_compile_options({model_name}_dpu PRIVATE ${{GLOG_CFLAGS_OTHER}})
"""
    
    with open(cmake_file, 'w') as f:
        f.write(cmake_content)
    
    # Generate model compilation script
    compile_script = output_dir / "compile_model.sh"
    compile_content = f"""#!/bin/bash

# Vitis AI Model Compilation Script for {model_name}

echo "Compiling model for FPGA deployment..."

# Set target device (adjust based on your FPGA)
TARGET="DPUCZDX8G_ISA1_B4096_MAX_BG2"

# Compile the model using Vitis AI compiler
vai_c_tensorflow2 \\
    --model {model_name}.pb \\
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \\
    --output_dir ./compiled_model \\
    --net_name {model_name} \\
    --options "{{\\\"input_shape\\\": \\\"1,15\\\"\\}}"

echo "Model compilation completed!"
echo "Generated files:"
echo "  - compiled_model/{model_name}.xmodel"
echo "  - compiled_model/meta.json"

# Build the inference application
echo "Building inference application..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo "Build completed!"
echo "Run inference with: ./build/{model_name}_dpu compiled_model/{model_name}.xmodel"
"""
    
    with open(compile_script, 'w') as f:
        f.write(compile_content)
    compile_script.chmod(0o755)
    
    # Generate deployment guide
    guide_file = output_dir / "DEPLOYMENT_GUIDE.md"
    guide_content = f"""# {model_name} FPGA Deployment Guide

## Prerequisites

1. **Vitis AI Environment**: Make sure you have Vitis AI installed and activated
   ```bash
   conda activate vitis-ai-tensorflow2
   ```

2. **Target FPGA**: This guide assumes ZCU104 board. Adjust arch.json for your target.

## Step 1: Convert PyTorch Model to ONNX/TensorFlow

If you haven't already, convert your PyTorch model to a format supported by Vitis AI:

```python
import torch
import torch.onnx

# Load your PyTorch model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 15)  # Adjust input shape
torch.onnx.export(model, dummy_input, "{model_name}.onnx")
```

## Step 2: Quantization

Quantize your model for FPGA deployment:

```bash
vai_q_tensorflow2 \\
    --model {model_name}.pb \\
    --input_frozen_graph {model_name}.pb \\
    --input_nodes input \\
    --output_nodes output \\
    --input_shapes ?,15 \\
    --calib_dataset calibration_data.h5
```

## Step 3: Compilation

Run the compilation script:

```bash
./compile_model.sh
```

## Step 4: Build and Run

```bash
cd build
make
./{model_name}_dpu ../compiled_model/{model_name}.xmodel
```

## Performance Optimization Tips

1. **Batch Processing**: Process multiple inputs together
2. **Pipeline**: Use async execution for better throughput
3. **Memory Management**: Reuse tensor buffers
4. **Fixed-Point**: Use INT8 quantization for better performance

## Troubleshooting

- **Compilation Errors**: Check input/output node names
- **Runtime Errors**: Verify input tensor shapes and data types
- **Performance Issues**: Profile using Vitis AI Profiler

## Hardware Specifications

- **Target**: ZCU104 (adjust for your board)
- **DPU**: DPUCZDX8G
- **Memory**: DDR4 (adjust based on model size)
"""
    
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Generated Vitis AI DPU inference code: {dpu_inference_file}")
    print(f"✅ Generated CMakeLists.txt: {cmake_file}")
    print(f"✅ Generated compilation script: {compile_script}")
    print(f"✅ Generated deployment guide: {guide_file}")
    
    return dpu_inference_file, cmake_file, compile_script, guide_file

def main():
    parser = argparse.ArgumentParser(description="Generate Vitis AI deployment files")
    parser.add_argument("input", help="Input JSON file with weights")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("-n", "--name", default="vad_model", help="Model name")
    
    args = parser.parse_args()
    
    try:
        # Load JSON weights
        with open(args.input, 'r') as f:
            weights_dict = json.load(f)
        
        print(f"Loaded weights from: {args.input}")
        
        # Generate Vitis AI files
        generate_vitis_ai_files(weights_dict, args.output, args.name)
        
        print(f"\n🚀 Vitis AI deployment files generated!")
        print(f"📁 Output directory: {args.output}")
        print(f"\nNext steps:")
        print(f"1. Convert your PyTorch model to TensorFlow/ONNX")
        print(f"2. Run quantization with vai_q_tensorflow2")
        print(f"3. Execute ./compile_model.sh")
        print(f"4. Build and run the inference application")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()