"""
Device utilities for macOS with MPS (Metal Performance Shaders) support.
Optimized for Apple Silicon Macs.
"""
import platform
import warnings
import os
import torch


def get_device_info():
    """Get device information for macOS systems."""
    info = {
        'platform': platform.system(),
        'mps_available': torch.backends.mps.is_available(),
        'cpu_threads': os.cpu_count() or 4,
    }
    
    # Determine recommended device
    if info['mps_available']:
        info['recommended_device'] = 'mps'
    else:
        info['recommended_device'] = 'cpu'
        warnings.warn("MPS not available. Falling back to CPU.")
    
    # macOS specific settings
    info['supports_distributed'] = False  # Not supported on MPS
    info['supports_cuda_graphs'] = False  # Not available on macOS
    
    return info


def print_device_info():
    """Print device information for macOS."""
    info = get_device_info()
    
    print("=== macOS Device Information ===")
    print(f"Platform: {info['platform']}")
    print(f"Recommended device: {info['recommended_device']}")
    print(f"MPS available: {info['mps_available']}")
    print(f"CPU threads: {info['cpu_threads']}")
    print(f"Supports distributed: {info['supports_distributed']}")
    print(f"Supports CUDA graphs: {info['supports_cuda_graphs']}")
    print("=" * 35)


def get_optimal_config_for_macos():
    """Get optimal configuration for macOS systems."""
    info = get_device_info()
    
    config = {
        'device': info['recommended_device'],
        'enforce_eager': True,  # Always use eager mode on macOS
        'tensor_parallel_size': 1,  # Single device only
        'gpu_memory_utilization': 0.8 if info['mps_available'] else 0.9,
        'max_model_len': 2048,  # Conservative for memory
        'max_num_seqs': 16,  # Reasonable batch size
    }
        
    return config


if __name__ == "__main__":
    print_device_info()
    print("\nOptimal config:")
    config = get_optimal_config_for_macos()
    for key, value in config.items():
        print(f"  {key}: {value}")
