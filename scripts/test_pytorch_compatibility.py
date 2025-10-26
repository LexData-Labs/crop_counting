#!/usr/bin/env python3
"""
Test script to verify PyTorch 2.6+ compatibility with YOLOv8 models
"""
import torch
import sys
import os

def test_pytorch_version():
    """Test PyTorch version and compatibility"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

def test_safe_globals():
    """Test safe globals registration"""
    try:
        # Test if we can register safe globals
        safe_globals = ['collections.OrderedDict']
        torch.serialization.add_safe_globals(safe_globals)
        print("✓ Safe globals registration works")
        return True
    except Exception as e:
        print(f"✗ Safe globals registration failed: {e}")
        return False

def test_yolo_import():
    """Test ultralytics import"""
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics import successful")
        return True
    except Exception as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False

def test_model_loading(model_name='yolov8n.pt'):
    """Test YOLO model loading"""
    try:
        # Import the safe loading function from our training script
        sys.path.append(os.path.dirname(__file__))
        from train_yolov8 import load_yolo_safely
        
        print(f"Testing model loading: {model_name}")
        model = load_yolo_safely(model_name)
        print(f"✓ Successfully loaded {model_name}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed for {model_name}: {str(e)[:200]}...")
        return False

def main():
    """Run all tests"""
    print("=== PyTorch Compatibility Test ===")
    
    test_pytorch_version()
    print()
    
    safe_globals_ok = test_safe_globals()
    print()
    
    yolo_import_ok = test_yolo_import()
    print()
    
    if safe_globals_ok and yolo_import_ok:
        print("Testing model loading...")
        # Test smaller models first
        for model in ['yolov8n.pt', 'yolov8s.pt']:
            success = test_model_loading(model)
            if success:
                print(f"✓ {model} works!")
                break
            print()
        
        # Test larger model if requested
        test_large = input("\nTest yolov8x.pt (may fail with PyTorch 2.6+)? (y/N): ").strip().lower()
        if test_large == 'y':
            test_model_loading('yolov8x.pt')
    else:
        print("Skipping model loading tests due to import failures")
    
    print("\n=== Test Complete ===")
    
if __name__ == "__main__":
    main()