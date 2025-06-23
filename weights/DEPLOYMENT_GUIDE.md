# vad_model FPGA Deployment Guide

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
torch.onnx.export(model, dummy_input, "vad_model.onnx")
```

## Step 2: Quantization

Quantize your model for FPGA deployment:

```bash
vai_q_tensorflow2 \
    --model vad_model.pb \
    --input_frozen_graph vad_model.pb \
    --input_nodes input \
    --output_nodes output \
    --input_shapes ?,15 \
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
./vad_model_dpu ../compiled_model/vad_model.xmodel
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
