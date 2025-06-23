#!/bin/bash

# Vitis AI Model Compilation Script for vad_model

echo "Compiling model for FPGA deployment..."

# Set target device (adjust based on your FPGA)
TARGET="DPUCZDX8G_ISA1_B4096_MAX_BG2"

# Compile the model using Vitis AI compiler
vai_c_tensorflow2 \
    --model vad_model.pb \
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
    --output_dir ./compiled_model \
    --net_name vad_model \
    --options "{\"input_shape\": \"1,15\"\}"

echo "Model compilation completed!"
echo "Generated files:"
echo "  - compiled_model/vad_model.xmodel"
echo "  - compiled_model/meta.json"

# Build the inference application
echo "Building inference application..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo "Build completed!"
echo "Run inference with: ./build/vad_model_dpu compiled_model/vad_model.xmodel"
