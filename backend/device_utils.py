"""
Device detection utilities for crop counting backend
"""

def detect_available_device():
    """
    Detect the best available device for training
    Returns 'cpu' if no CUDA is available, or '0' if CUDA is available
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(f"CUDA available: {torch.cuda.device_count()} device(s)")
            return '0'  # Use first GPU
        else:
            print("CUDA not available, using CPU")
            return 'cpu'
    except ImportError:
        print("PyTorch not available, defaulting to CPU")
        return 'cpu'
    except Exception as e:
        print(f"Error detecting device: {e}")
        return 'cpu'

def get_recommended_batch_size(device='cpu'):
    """
    Get recommended batch size based on device and available memory
    """
    if device == 'cpu':
        return 2  # Slightly higher for better CPU utilization
    else:
        # For GPU, check memory and recommend optimal batch size
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory in GB
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
                
                # RTX 4060 has 8GB - optimize for this
                if gpu_memory_gb >= 7.5:  # RTX 4060 range
                    return 16  # Good balance for 8GB VRAM
                elif gpu_memory_gb >= 6:
                    return 12
                elif gpu_memory_gb >= 4:
                    return 8
                else:
                    return 4
        except Exception as e:
            print(f"Error detecting GPU memory: {e}")
        return 8  # Default for GPU

def get_recommended_workers(device='cpu'):
    """
    Get recommended number of workers based on device and CPU count
    """
    try:
        import os
        cpu_count = os.cpu_count() or 1
        if device == 'cpu':
            # For CPU training, use fewer workers to avoid competition
            return min(2, cpu_count // 2)
        else:
            # For GPU training, can use more workers for data loading
            return min(6, cpu_count)
    except:
        return 2  # Conservative fallback