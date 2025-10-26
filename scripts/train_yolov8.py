import argparse
from ultralytics import YOLO
import os
import torch
import sys
from pathlib import Path
import urllib.request
import tempfile
from multiprocessing import freeze_support

# Configure PyTorch for compatibility with Ultralytics models
try:
    # For PyTorch 2.6+, register comprehensive safe globals for Ultralytics model loading
    import torch.serialization
    if hasattr(torch.serialization, 'add_safe_globals'):
        # Core PyTorch classes
        safe_classes = [
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList, 
            torch.nn.modules.container.ModuleDict,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.dropout.Dropout,
        ]
        
        # Comprehensive Ultralytics classes
        try:
            # Main model classes
            from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
            safe_classes.extend([DetectionModel, SegmentationModel, ClassificationModel])
            
            # Core modules
            from ultralytics.nn.modules import (
                Conv, DWConv, GhostConv, LightConv, RepConv,
                Bottleneck, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x,
                SPP, SPPF, SPPCSPC, GhostBottleneck, Concat, Detect, Segment, Classify,
                RTDETRDecoder, v10Detect
            )
            safe_classes.extend([
                Conv, DWConv, GhostConv, LightConv, RepConv,
                Bottleneck, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x,
                SPP, SPPF, SPPCSPC, GhostBottleneck, Concat, Detect, Segment, Classify,
                RTDETRDecoder, v10Detect
            ])
            
            # Attention modules
            try:
                from ultralytics.nn.modules.attention import CBAM
                safe_classes.append(CBAM)
            except ImportError:
                pass
                
        except ImportError as e:
            print(f"Warning: Some Ultralytics modules not available: {e}")
            # Fallback - add basic classes that should always work
            try:
                from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
                safe_classes.extend([Conv, C2f, SPPF, Detect])
            except ImportError:
                pass
            
        # Add all safe globals
        torch.serialization.add_safe_globals(safe_classes)
        print(f"Added {len(safe_classes)} safe globals for PyTorch 2.6+ compatibility")
        
except Exception as e:
    print(f"Warning: Could not register safe globals: {e}")

# Alternative approach: Try to configure ultralytics to use weights_only=False if available
try:
    import ultralytics
    # This is a potential hook point if ultralytics supports it in future versions
except ImportError:
    pass

WEIGHTS_CACHE = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(WEIGHTS_CACHE, exist_ok=True)


def is_valid_weight_file(path: str) -> bool:
    """Basic validation to avoid using partial/corrupted downloads."""
    if not os.path.isfile(path):
        return False
    # Avoid using tmp/partial files that may be present
    if path.endswith('.partial'):
        return False
    # Basic size check (> 1 MB)
    try:
        return os.path.getsize(path) > 1_000_000
    except OSError:
        return False


def download_weights(model_name):
    """Download weights with proper validation and cleanup"""
    weights_path = os.path.join(WEIGHTS_CACHE, model_name)
    
    # Remove any existing corrupted/partial files
    for pattern in [weights_path, f"{weights_path}.*"]:
        try:
            import glob
            for file_path in glob.glob(pattern):
                if os.path.exists(file_path) and not is_valid_weight_file(file_path):
                    os.remove(file_path)
                    print(f"Removed corrupted file: {file_path}")
        except Exception:
            pass
    
    if not is_valid_weight_file(weights_path):
        print(f"Downloading fresh {model_name}...")
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        
        # Download to temporary file first, then move to avoid partial downloads
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            try:
                urllib.request.urlretrieve(url, tmp_file.name)
                if is_valid_weight_file(tmp_file.name):
                    # Move successful download to final location
                    if os.path.exists(weights_path):
                        os.remove(weights_path)
                    os.rename(tmp_file.name, weights_path)
                    print(f"Successfully downloaded {model_name}")
                else:
                    os.remove(tmp_file.name)
                    raise Exception("Downloaded file appears corrupted")
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(tmp_file.name):
                    os.remove(tmp_file.name)
                raise e
                
    return weights_path


def load_yolo_safely(model_path):
    """Load YOLO model with PyTorch 2.6+ compatibility and robust error handling"""
    basename = os.path.basename(model_path)
    is_official_model = basename in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    
    print(f"Attempting to load model: {model_path}")
    
    # Check if the file exists and is valid
    if not is_valid_weight_file(model_path):
        print(f"Invalid or missing weight file: {model_path}")
        if is_official_model:
            print("Attempting to download fresh official weights...")
            try:
                fresh_weights = download_weights(basename)
                print(f"Loading downloaded weights: {fresh_weights}")
                return load_yolo_with_fallbacks(fresh_weights)
            except Exception as e:
                print(f"Failed to download weights: {e}")
                raise RuntimeError(f"Could not download or load {basename}. Check your internet connection.")
        else:
            raise RuntimeError(f"Custom weight file not found or invalid: {model_path}")
    
    return load_yolo_with_fallbacks(model_path)


def load_yolo_with_fallbacks(model_path):
    """Load YOLO with multiple fallback methods for PyTorch 2.6+ compatibility"""
    basename = os.path.basename(model_path)
    is_official_model = basename in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    
    # Method 1: Try with PyTorch 2.8+ workaround first for official models
    if is_official_model:
        try:
            print("Attempting PyTorch 2.8+ compatible loading...")
            return load_yolo_pytorch28_workaround(model_path)
        except Exception as e1:
            print(f"PyTorch 2.8+ method failed: {str(e1)[:100]}...")
    
    # Method 2: Direct loading with registered safe globals
    try:
        print("Attempting direct model loading...")
        return YOLO(model_path)
    except Exception as e2:
        print(f"Direct loading failed: {str(e2)[:200]}...")
        
        # Method 3: For official models, try downloading fresh weights
        if is_official_model and model_path.endswith(basename):
            try:
                print("Falling back to fresh download...")
                fresh_weights = download_weights(basename)
                # Try the PyTorch 2.8+ method with fresh weights
                return load_yolo_pytorch28_workaround(fresh_weights)
            except Exception as e3:
                print(f"Fresh weights fallback failed: {str(e3)[:200]}...")
        
        # If all else fails, provide clear guidance
        error_msg = str(e2).lower()
        if "weights_only" in error_msg or "unpickling" in error_msg or "module" in error_msg:
            torch_version = torch.__version__
            raise RuntimeError(
                f"PyTorch {torch_version} compatibility issue. Recommended solutions:\n"
                f"1. Use smaller model: Change to yolov8n.pt or yolov8s.pt\n"
                f"2. Downgrade PyTorch: pip install torch==2.4.0 torchvision==0.19.0\n"
                f"3. Try: pip install ultralytics --upgrade\n"
                f"4. Delete all .pt files in scripts/weights/ and restart\n"
                f"Original error: {str(e2)[:200]}..."
            )
        else:
            raise RuntimeError(
                f"Failed to load YOLO model. Check file integrity.\n"
                f"Original error: {str(e2)[:200]}..."
            )


def load_yolo_pytorch28_workaround(model_path):
    """Workaround for PyTorch 2.8+ weights_only issue with trusted Ultralytics models"""
    import torch
    from ultralytics import YOLO
    import os
    
    # This is a workaround for trusted official Ultralytics models only
    basename = os.path.basename(model_path)
    
    # Method A: Try with safe globals context manager
    try:
        # Import the actual classes for safe globals (not strings)
        import collections
        extra_safe_classes = [collections.OrderedDict]
        
        # Try to import ultralytics classes
        try:
            from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, Bottleneck
            from ultralytics.nn.tasks import DetectionModel
            extra_safe_classes.extend([Conv, C2f, SPPF, Detect, Bottleneck, DetectionModel])
        except ImportError:
            pass
        
        with torch.serialization.safe_globals(extra_safe_classes):
            model = YOLO(model_path)
            print("Successfully loaded with safe globals context manager")
            return model
    except Exception as e1:
        print(f"Safe globals context failed: {str(e1)[:100]}...")
    
    # Method B: Use ultralytics hub loading (bypasses local file loading)
    try:
        print("Trying ultralytics hub model loading...")
        # For official models, try loading from hub instead of file
        hub_name = basename.replace('.pt', '')
        model = YOLO(f'{hub_name}.pt')  # This will download from ultralytics
        print(f"Successfully loaded {hub_name} from ultralytics hub")
        return model
    except Exception as e2:
        print(f"Hub loading failed: {str(e2)[:100]}...")
    
    # Method C: Monkey-patch torch.load temporarily
    original_torch_load = torch.load
    
    def safe_torch_load(*args, **kwargs):
        # For trusted Ultralytics models, we allow weights_only=False
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    try:
        # Temporarily replace torch.load
        torch.load = safe_torch_load
        model = YOLO(model_path)
        print("Successfully loaded with torch.load monkey patch")
        return model
    except Exception as e3:
        print(f"Monkey patch failed: {str(e3)[:100]}...")
        
        # Method C: Force download a clean copy and try again
        if basename in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
            try:
                # Remove potentially corrupted file
                if os.path.exists(model_path):
                    os.remove(model_path)
                    print(f"Removed potentially corrupted {model_path}")
                
                # Download fresh copy
                fresh_path = download_weights(basename)
                model = YOLO(fresh_path)
                print("Successfully loaded fresh downloaded model")
                return model
            except Exception as e3:
                print(f"Fresh download failed: {str(e3)[:100]}...")
        
        # Method D: Environment variable workaround
        try:
            print("Trying environment variable workaround...")
            import os
            # Set environment variable to allow unsafe loading
            original_env = os.environ.get('PYTORCH_ENABLE_UNSAFE_LOAD', None)
            os.environ['PYTORCH_ENABLE_UNSAFE_LOAD'] = '1'
            
            try:
                model = YOLO(model_path)
                print("Successfully loaded with environment variable workaround")
                return model
            finally:
                # Restore original environment
                if original_env is None:
                    os.environ.pop('PYTORCH_ENABLE_UNSAFE_LOAD', None)
                else:
                    os.environ['PYTORCH_ENABLE_UNSAFE_LOAD'] = original_env
                    
        except Exception as e4:
            print(f"Environment workaround failed: {str(e4)[:100]}...")
        
        # If all methods fail, raise the original exception
        raise e2
    finally:
        # Always restore original torch.load
        torch.load = original_torch_load


def main():
    """Main training function with proper argument parsing and model training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/dataset.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='pretrained weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--project', type=str, default='../models/experiments')
    parser.add_argument('--name', type=str, default='yolov8_crop_detect')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    
    # Windows-specific multiprocessing fix: reduce workers to prevent spawn issues
    if os.name == 'nt':  # Windows
        if args.workers > 0:
            original_workers = args.workers
            args.workers = min(args.workers, 2)  # Limit to max 2 workers on Windows
            if args.workers != original_workers:
                print(f"Windows detected: Reduced workers from {original_workers} to {args.workers} to prevent multiprocessing issues")
    
    os.makedirs(args.project, exist_ok=True)
    
    print(f"Loading model {args.model}...")
    # Use YOLO's built-in model loading which handles downloads automatically
    try:
        model = YOLO(args.model)  # YOLO will automatically download if needed
        print(f"Successfully loaded model: {args.model}")
    except Exception as e:
        print(f"Failed to load model {args.model}. Trying fallback...")
        print(f"Error: {e}")
        # Fallback to the complex loading method
        model = load_yolo_safely(args.model)
    
    # use task='detect' (default) for YOLO text labels
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=args.name, device=args.device, workers=args.workers)


if __name__ == '__main__':
    # Required for Windows multiprocessing support
    freeze_support()
    main()
