#!/usr/bin/env python
import subprocess
import sys
import os

def install_package(package, additional_args=[]):
    """Install a Python package using pip."""
    print(f"Installing {package}...")
    cmd = [sys.executable, "-m", "pip", "install", package] + additional_args
    try:
        subprocess.check_call(cmd)
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def check_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA is available. CUDA version: {torch.version.cuda}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available.")
        return cuda_available
    except ImportError:
        print("PyTorch is not installed. Installing PyTorch...")
        install_package("torch")
        return check_cuda_available()

def main():
    print("Installing dependencies for medical LLM evaluation...")
    
    # Check if CUDA is available
    cuda_available = check_cuda_available()
    
    # Basic dependencies
    basic_deps = [
        "transformers",
        "nltk",
        "rouge-score",
        "pandas",
        "matplotlib",
        "tqdm"
    ]
    
    for dep in basic_deps:
        install_package(dep)
    
    # Download NLTK data
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Failed to download NLTK data: {e}")
    
    # Optional advanced packages
    print("\nWould you like to install advanced packages for faster inference? (y/n)")
    choice = input().lower()
    
    if choice.startswith('y'):
        # Try to install vllm
        print("\nAttempting to install vllm for faster inference...")
        vllm_success = install_package("vllm")
        
        # Try to install bitsandbytes for int8 quantization
        print("\nAttempting to install bitsandbytes for int8 quantization...")
        bitsandbytes_success = install_package("bitsandbytes")
        
        if vllm_success:
            print("\nvllm installed successfully.")
            print("You can use it with: python run_inference.py")
        else:
            print("\nvllm installation failed.")
            print("You can still use the script with: python run_inference.py --no_vllm")
        
        if bitsandbytes_success:
            print("\nbitsandbytes installed successfully.")
            print("You can use int8 quantization by default.")
        else:
            print("\nbitsandbytes installation failed.")
            print("The script will automatically fall back to bfloat16.")
    
    print("\nAll basic dependencies installed. You can now run the script with:")
    print("python run_inference.py --n_samples 3 --models deepseek")
    print("\nAdd --no_vllm to disable vllm if it's causing issues.")
    print("Add --no_int8 to disable int8 quantization if it's causing issues.")

if __name__ == "__main__":
    main() 