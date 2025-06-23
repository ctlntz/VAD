import subprocess
import json
import argparse
import os
from pathlib import Path
import shutil

def get_available_targets():
    """Get list of available Vitis AI compilation targets"""
    
    targets = {
        'ZCU104': {
            'arch': '/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json',
            'description': 'Xilinx ZCU104 Development Board'
        },
        'ZCU102': {
            'arch': '/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json',
            'description': 'Xilinx ZCU102 Development Board'
        },
        'Ultra96': {
            'arch': '/opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/arch.json',
            'description': 'Avnet Ultra96-V2 Development Board'
        },
        'KV260': {
            'arch': '/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json',
            'description': 'Kria KV260 Vision AI Starter Kit'
        },
        'VCK190': {
            'arch': '/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json',
            'description': 'Versal VCK190 Evaluation Kit'
        }
    }
    
    # Check which arch files actually exist
    available_targets = {}
    for name, info in targets.items():
        if Path(info['arch']).exists():
            available_targets[name] = info
    
    return available_targets

def create_compile_script(quantized_model_path, target_board="ZCU104", model_name="vad_model", output_dir="compiled_model"):
    """
    Create compilation script for Vitis AI compiler
    
    Args:
        quantized_model_path (str): Path to quantized model (.pb file)
        target_board (str): Target FPGA board
        model_name (str): Name for the compiled model
        output_dir (str): Output directory for compiled model
    """
    
    available_targets = get_available_targets()
    
    if target_board not in available_targets:
        print(f"❌ Target board '{target_board}' not available")
        print(f"Available targets: {list(available_targets.keys())}")
        return None
    
    arch_path = available_targets[target_board]['arch']
    
    # Create compilation script
    compile_script = Path("compile_model.sh")
    
    with compile_script.open("w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Compilation script for {model_name}\n")
        f.write(f"echo 'Compiling model {model_name} for target {target_board}'\n")
        f.write(f"vitis_ai_compiler --model {quantized_model_path} --arch {arch_path} --output_dir {output_dir} --model_name {model_name}\n")
        f.write("if [ $? -ne 0 ]; then\n")
        f.write("  echo 'Compilation failed!'\n")
        f.write("  exit 1\n")
        f.write("fi\n")
        f.write("echo 'Compilation completed successfully!'\n")
    
    # Make the script executable
    compile_script.chmod(0o755)
    
    print(f"Compilation script created: {compile_script}")
    return compile_script

def run_compile_script(script_path):
    """Run the compilation script."""
    try:
        subprocess.run([str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Vitis AI Compilation Script Generator")
    parser.add_argument("quantized_model_path", type=str, help="Path to the quantized model (.pb file)")
    parser.add_argument("--target_board", type=str, default="ZCU104", help="Target FPGA board")
    parser.add_argument("--model_name", type=str, default="vad_model", help="Name for the compiled model")
    parser.add_argument("--output_dir", type=str, default="compiled_model", help="Output directory for compiled model")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the compile script
    script_path = create_compile_script(args.quantized_model_path, args.target_board, args.model_name, args.output_dir)
    
    if script_path:
        # Run the compile script
        success = run_compile_script(script_path)
        if success:
            print("Model compiled successfully.")
        else:
            print("Model compilation failed.")

if __name__ == "__main__":
    main()