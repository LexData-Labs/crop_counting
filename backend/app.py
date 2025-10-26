import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import json
import threading
import time
from datetime import datetime
import uuid
import shutil
from werkzeug.utils import secure_filename
from device_utils import detect_available_device, get_recommended_batch_size, get_recommended_workers


app = Flask(__name__)
# Allow uploads up to 5GB
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 * 1024  # 20 GB
from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({'error': 'File too large. The server allows up to 20GB per upload. Please split your dataset or contact the administrator.'}), 413
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'txt'}
ALLOWED_MODEL_EXTENSIONS = {'pt'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def ensure_dataset_structure():
    """Ensure the dataset directory structure exists"""
    dataset_root = os.path.join(UPLOAD_FOLDER, 'dataset')
    required_dirs = [
        os.path.join(dataset_root, 'images', 'train'),
        os.path.join(dataset_root, 'images', 'val'),
        os.path.join(dataset_root, 'images', 'test'),
        os.path.join(dataset_root, 'labels', 'train'),
        os.path.join(dataset_root, 'labels', 'val'),
        os.path.join(dataset_root, 'labels', 'test')
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
    return required_dirs

def validate_dataset():
    """Check if dataset structure is valid"""
    required_dirs = [
        os.path.join(UPLOAD_FOLDER, 'dataset', 'images', 'train'),
        os.path.join(UPLOAD_FOLDER, 'dataset', 'images', 'val'),
        os.path.join(UPLOAD_FOLDER, 'dataset', 'labels', 'val')
    ]
    
    # Print directory contents for debugging
    for d in required_dirs:
        print(f"Checking directory: {d}")
        if os.path.exists(d):
            print(f"Files found: {os.listdir(d)}")
        else:
            print("Directory does not exist")
    
    missing = []
    for d in required_dirs:
        if not os.path.exists(d):
            missing.append(d)
        elif not os.listdir(d):
            missing.append(f"{d} (empty)")
            
    if missing:
        raise ValueError(f"Missing or empty required directories: {missing}. Please upload your dataset first.")
    return True

# Global state for tracking processes
training_processes = {}
detection_processes = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_model_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS

def run_training_script(config):
    """Run the training script with given configuration"""
    try:
        # Get absolute paths for better path resolution
        backend_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(backend_dir, '..'))
        script_path = os.path.join(project_root, 'scripts', 'train_yolov8.py')
        
        # Ensure experiments directory exists and is used as Ultralytics project folder
        project_dir = os.path.abspath(os.path.join(backend_dir, MODELS_FOLDER, 'experiments'))
        os.makedirs(project_dir, exist_ok=True)
        
        # Use absolute paths for better reliability
        data_path = os.path.abspath(config['data_path'])
        
        cmd = [
            'python', script_path,
            '--data', data_path,
            '--model', config['model'],
            '--epochs', str(config['epochs']),
            '--batch', str(config['batch_size']),
            '--imgsz', str(config['image_size']),
            '--project', project_dir,
            '--name', config['name'],
            '--device', config['device'],
            '--workers', str(config['workers'])
        ]
        
        print(f"Starting training with command: {' '.join(cmd)}")
        print(f"Working directory: {project_root}")
        print(f"Data path: {data_path}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout for better logging
            text=True,
            cwd=project_root,  # Run from project root for better path resolution
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        return process
    except Exception as e:
        print(f"Error starting training: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def run_detection_script(image_path, config):
    """Run detection on a single image using YOLO directly"""
    try:
        from ultralytics import YOLO
        
        # Load the model
        model_path = config['weights_path']
        if not os.path.exists(model_path):
            return {
                'success': False,
                'error': f'Model not found: {model_path}'
            }
            
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        
        # Run inference without saving initially
        print(f"Running detection on: {image_path}")
        results = model.predict(
            source=image_path,
            conf=config['confidence'],
            iou=config['iou'],
            save=False,  # Don't let YOLO handle saving
            verbose=False
        )
        
        # Count detections
        total_count = 0
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    total_count += len(result.boxes)
        
        print(f"Detection completed. Found {total_count} objects.")
        
        # Always save result image (with or without detections for visualization)
        result_image_path = None
        try:
            if results and len(results) > 0:
                result_image_path = save_detection_result(results[0], image_path)
            else:
                # Even if no results, copy the original image as result for display
                result_image_path = copy_original_as_result(image_path)
        except Exception as save_error:
            print(f"Warning: Could not save result image: {save_error}")
            # Don't fail the whole detection just because we can't save the image
        
        return {
            'success': True,
            'count': total_count,
            'result_path': result_image_path,
            'output': f'Detection completed. Found {total_count} objects.'
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Detection error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            'success': False,
            'error': error_msg
        }

def save_detection_result(result, original_image_path):
    """Save detection result with boxes drawn"""
    try:
        import cv2
        import numpy as np
        
        # Create results directory
        backend_dir = os.path.dirname(__file__)
        results_dir = os.path.join(backend_dir, RESULTS_FOLDER, 'detection')
        os.makedirs(results_dir, exist_ok=True)
        
        # Load original image
        image = cv2.imread(original_image_path)
        if image is None:
            return None
        
        # Draw detection boxes if any
        detection_count = 0
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            detection_count = len(boxes)
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw rectangle with thicker line
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw confidence score with background
                label = f'{conf:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw detection number
                number_label = f'#{i+1}'
                cv2.putText(image, number_label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add info text at the top of the image
        info_text = f'Detections: {detection_count}'
        cv2.rectangle(image, (10, 10), (300, 50), (0, 0, 0), -1)
        cv2.putText(image, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result image
        original_name = os.path.basename(original_image_path)
        name_without_ext = os.path.splitext(original_name)[0]
        ext = os.path.splitext(original_name)[1] or '.jpg'
        result_filename = f'result_{name_without_ext}{ext}'
        result_path = os.path.join(results_dir, result_filename)
        
        success = cv2.imwrite(result_path, image)
        if success:
            print(f"Saved result image: {result_path} ({detection_count} detections)")
            return result_path
        else:
            print(f"Failed to save result image: {result_path}")
            return None
        
    except Exception as e:
        print(f"Error saving detection result: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def copy_original_as_result(original_image_path):
    """Copy original image as result when no detections found"""
    try:
        import cv2
        
        # Create results directory
        backend_dir = os.path.dirname(__file__)
        results_dir = os.path.join(backend_dir, RESULTS_FOLDER, 'detection')
        os.makedirs(results_dir, exist_ok=True)
        
        # Load original image
        image = cv2.imread(original_image_path)
        if image is None:
            return None
        
        # Add "No detections found" text
        cv2.rectangle(image, (10, 10), (350, 50), (0, 0, 0), -1)
        cv2.putText(image, 'No detections found', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save result image
        original_name = os.path.basename(original_image_path)
        name_without_ext = os.path.splitext(original_name)[0]
        ext = os.path.splitext(original_name)[1] or '.jpg'
        result_filename = f'result_{name_without_ext}{ext}'
        result_path = os.path.join(results_dir, result_filename)
        
        success = cv2.imwrite(result_path, image)
        if success:
            print(f"Saved result image (no detections): {result_path}")
            return result_path
        else:
            print(f"Failed to save result image: {result_path}")
            return None
        
    except Exception as e:
        print(f"Error copying original as result: {e}")
        return None

def run_tile_inference_script(image_path, config):
    """Run tile inference on large images using YOLO directly (simplified version)"""
    try:
        from ultralytics import YOLO
        
        # Load the model
        model_path = config['weights_path']
        if not os.path.exists(model_path):
            return {
                'success': False,
                'error': f'Model not found: {model_path}'
            }
            
        print(f"Loading model for tile inference: {model_path}")
        model = YOLO(model_path)
        
        # For now, just run regular inference for simplicity
        # TODO: Implement proper tile inference later
        print(f"Running detection on large image: {image_path}")
        results = model.predict(
            source=image_path,
            conf=config['confidence'],
            iou=config['iou'],
            save=False,
            verbose=False
        )
        
        # Count detections
        total_count = 0
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    total_count += len(result.boxes)
        
        print(f"Detection completed. Found {total_count} objects.")
        
        # Always save result image (with or without detections)
        result_image_path = None
        try:
            if results and len(results) > 0:
                result_image_path = save_detection_result(results[0], image_path)
            else:
                # Even if no results, copy the original image as result for display
                result_image_path = copy_original_as_result(image_path)
        except Exception as save_error:
            print(f"Warning: Could not save result image: {save_error}")
        
        return {
            'success': True,
            'count': total_count,
            'result_path': result_image_path,
            'output': f'Large image detection completed. Found {total_count} objects.'
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Tile inference error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            'success': False,
            'error': error_msg
        }

# Removed complex NMS and save_detection_image functions to avoid issues

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/device/recommendations', methods=['GET'])
def get_device_recommendations():
    """Get recommended device and training configuration"""
    try:
        device = detect_available_device()
        batch_size = get_recommended_batch_size(device)
        workers = get_recommended_workers(device)
        
        # Additional recommendations based on device
        if device == 'cpu':
            recommended_config = {
                'device': device,
                'batch_size': batch_size,
                'workers': workers,
                'image_size': 320,  # Smaller for CPU
                'epochs': 50,
                'model': 'yolov8n.pt',  # Smallest model for CPU
                'message': 'CPU training detected. Using smaller model and batch size for better performance.'
            }
        else:
            # Get GPU name for specific optimizations
            gpu_name = "Unknown GPU"
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
            except:
                pass
                
            recommended_config = {
                'device': device,
                'batch_size': batch_size,
                'workers': workers,
                'image_size': 640,  # Standard for GPU
                'epochs': 100,
                'model': 'yolov8s.pt',  # Good balance for RTX 4060
                'message': f'GPU training available ({gpu_name}). Optimized for 8GB VRAM.'
            }
            
        return jsonify({
            'status': 'success',
            'recommendations': recommended_config,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'fallback': {
                'device': 'cpu',
                'batch_size': 1,
                'workers': 2,
                'image_size': 320,
                'epochs': 50,
                'model': 'yolov8n.pt',
                'message': 'Using safe fallback configuration.'
            },
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/training/health', methods=['GET'])
def training_health_check():
    """Check if training environment is ready"""
    try:
        # Check if ultralytics is available
        import ultralytics
        from ultralytics import YOLO
        
        # Check if training script exists
        backend_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(backend_dir, '..'))
        script_path = os.path.join(project_root, 'scripts', 'train_yolov8.py')
        
        checks = {
            'ultralytics_available': True,
            'training_script_exists': os.path.exists(script_path),
            'uploads_folder_exists': os.path.exists(os.path.join(backend_dir, UPLOAD_FOLDER)),
            'models_folder_exists': os.path.exists(os.path.join(backend_dir, MODELS_FOLDER))
        }
        
        all_good = all(checks.values())
        
        return jsonify({
            'status': 'ready' if all_good else 'not_ready',
            'checks': checks,
            'script_path': script_path,
            'timestamp': datetime.now().isoformat()
        })
        
    except ImportError as e:
        return jsonify({
            'status': 'error',
            'error': f'Missing dependencies: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload dataset files"""
    import traceback
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        # Ensure dataset structure exists
        ensure_dataset_structure()
        
        files = request.files.getlist('files')
        uploaded_files = []

        # Try to get relative paths from form if provided (frontend must send 'relative_paths' as a JSON array)
        relative_paths = request.form.get('relative_paths')
        rel_paths_list = None
        if relative_paths:
            import json as _json
            try:
                rel_paths_list = _json.loads(relative_paths)
            except Exception as ex:
                print('Failed to parse relative_paths:', ex)
                rel_paths_list = None

        for idx, file in enumerate(files):
            if file and allowed_file(file.filename):
                # If relative path is provided, use it; else fallback to filename
                rel_path = None
                if rel_paths_list and idx < len(rel_paths_list):
                    rel_path = rel_paths_list[idx]
                if rel_path:
                    # Map the file to correct dataset structure
                    parts = os.path.normpath(rel_path).replace('..', '').split(os.sep)
                    if 'train' in parts:
                        split = 'train'
                    elif 'val' in parts:
                        split = 'val'
                    elif 'test' in parts:
                        split = 'test'
                    else:
                        split = 'train'  # default to train if no split specified
                        
                    if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        save_dir = os.path.join(UPLOAD_FOLDER, 'dataset', 'images', split)
                    else:  # .txt files
                        save_dir = os.path.join(UPLOAD_FOLDER, 'dataset', 'labels', split)
                        
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, os.path.basename(file.filename))
                else:
                    filename = secure_filename(file.filename)
                    save_path = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    file.save(save_path)
                    uploaded_files.append({
                        'name': file.filename,
                        'size': os.path.getsize(save_path),
                        'path': save_path
                    })
                except Exception as file_ex:
                    print(f'Failed to save file {file.filename}:', file_ex)
                    print(traceback.format_exc())
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files
        })
    except Exception as e:
        print('Upload error:', e)
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        # First ensure and validate dataset structure
        ensure_dataset_structure()
        validate_dataset()

        config = request.json
        training_id = str(uuid.uuid4())
        
        # Validate that dataset exists before training
        required_dirs = [
            os.path.join(UPLOAD_FOLDER, 'dataset', 'images', 'train'),
            os.path.join(UPLOAD_FOLDER, 'dataset', 'images', 'val'),
            os.path.join(UPLOAD_FOLDER, 'dataset', 'labels', 'train'),
            os.path.join(UPLOAD_FOLDER, 'dataset', 'labels', 'val')
        ]
        
        missing_dirs = []
        empty_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
            elif len(os.listdir(dir_path)) == 0:
                empty_dirs.append(dir_path)
        
        if missing_dirs:
            return jsonify({
                'error': f'Missing required directories: {missing_dirs}. Please upload your dataset first.'
            }), 400
            
        if empty_dirs:
            return jsonify({
                'error': f'Empty directories found: {empty_dirs}. Please ensure you have uploaded both images and labels.'
            }), 400
        
        print(f"Dataset validation passed for training {training_id}")
        
        # Overwrite all YOLO label files to have class 0 (crop) for every annotation
        import glob
        label_dirs = [
            os.path.join(UPLOAD_FOLDER, 'dataset', 'labels', split)
            for split in ['train', 'val', 'test']
        ]
        for label_dir in label_dirs:
            if os.path.exists(label_dir):
                for txt_file in glob.glob(os.path.join(label_dir, '*.txt')):
                    new_lines = []
                    with open(txt_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Replace class index with 0
                                new_lines.append('0 ' + ' '.join(parts[1:]))
                    with open(txt_file, 'w') as f:
                        for l in new_lines:
                            f.write(l + '\n')

        # Always create dataset.yaml with absolute paths for better reliability
        backend_dir = os.path.dirname(__file__)
        uploads_abs_path = os.path.abspath(os.path.join(backend_dir, UPLOAD_FOLDER))
        data_path = os.path.join(uploads_abs_path, 'dataset.yaml')
        
        dataset_config = {
            'train': os.path.join(uploads_abs_path, 'dataset', 'images', 'train'),
            'val': os.path.join(uploads_abs_path, 'dataset', 'images', 'val'), 
            'test': os.path.join(uploads_abs_path, 'dataset', 'images', 'test'),
            'nc': 1,
            'names': ['crop']
        }
        with open(data_path, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f)

        config['data_path'] = data_path
        config['name'] = f'yolov8_crop_detect_{training_id[:8]}'
        
        # Auto-detect device if not specified or if invalid
        if 'device' not in config or config['device'] in ['', 'auto']:
            config['device'] = detect_available_device()
            print(f"Auto-detected device: {config['device']}")
        elif config['device'] == '0':  # Check if GPU is actually available
            try:
                import torch
                if not torch.cuda.is_available():
                    print("GPU requested but CUDA not available, falling back to CPU")
                    config['device'] = 'cpu'
            except ImportError:
                config['device'] = 'cpu'

        # Start training process
        process = run_training_script(config)
        if process:
            training_processes[training_id] = {
                'process': process,
                'config': config,
                'status': 'running',
                'start_time': datetime.now().isoformat(),
                'progress': 0,
                'total_epochs': int(config.get('epochs', 0)),
                'loss': 0.0,
                'accuracy': 0.0
            }
            
            # Start monitoring thread
            def monitor_training():
                backend_dir = os.path.dirname(__file__)
                results_csv = os.path.join(backend_dir, MODELS_FOLDER, 'experiments', config['name'], 'results.csv')
                last_epoch_reported = -1
                print(f"Monitoring training process {training_id}, looking for results at: {results_csv}")
                
                while process.poll() is None:
                    time.sleep(2)
                    
                    # Read process output for error detection
                    try:
                        # Check if process has output to read
                        if process.stdout and not process.stdout.closed:
                            line = process.stdout.readline()
                            if line:
                                print(f"Training output: {line.strip()}")
                                # Look for errors in output
                                if 'error' in line.lower() or 'exception' in line.lower():
                                    print(f"Training error detected: {line.strip()}")
                    except Exception as output_ex:
                        print(f"Error reading training output: {output_ex}")
                    
                    try:
                        if os.path.exists(results_csv):
                            # Read last non-header line
                            with open(results_csv, 'r') as f:
                                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                            if len(lines) >= 2:
                                header = [h.strip() for h in lines[0].split(',')]
                                last = [v.strip() for v in lines[-1].split(',')]
                                # Epoch number may be in first column named 'epoch'
                                data = {header[i]: last[i] for i in range(min(len(header), len(last)))}
                                # Update epoch
                                epoch_val = 0
                                if 'epoch' in data:
                                    try:
                                        epoch_val = int(float(data['epoch'])) + 1  # 0-index to 1-index
                                    except:
                                        pass
                                else:
                                    # Fallback: infer by row count minus header
                                    epoch_val = len(lines) - 1
                                # Update metrics
                                loss_val = 0.0
                                acc_val = 0.0
                                # Ultralytics typical columns: 'train/box_loss','metrics/mAP50-95(B)' etc.
                                for key in ['loss', 'train/loss', 'train/box_loss']:
                                    if key in data:
                                        try:
                                            loss_val = float(data[key])
                                            break
                                        except:
                                            pass
                                for key in ['accuracy', 'metrics/mAP50-95(B)', 'metrics/mAP50-95', 'metrics/mAP50(B)', 'metrics/mAP50']:
                                    if key in data:
                                        try:
                                            acc_val = float(data[key])
                                            break
                                        except:
                                            pass
                                if training_id in training_processes:
                                    training_processes[training_id]['progress'] = min(epoch_val, training_processes[training_id]['total_epochs'] or epoch_val)
                                    training_processes[training_id]['loss'] = loss_val
                                    training_processes[training_id]['accuracy'] = acc_val
                                    training_processes[training_id]['status'] = 'running'
                    except Exception as mon_ex:
                        print('Monitor training parse error:', mon_ex)
                
                # Training completed - check return code
                return_code = process.poll()
                if training_id in training_processes:
                    if return_code == 0:
                        training_processes[training_id]['status'] = 'completed'
                        print(f"Training {training_id} completed successfully")
                    else:
                        training_processes[training_id]['status'] = 'failed'
                        print(f"Training {training_id} failed with return code: {return_code}")
                        # Try to get error output
                        try:
                            if process.stdout:
                                remaining_output = process.stdout.read()
                                if remaining_output:
                                    print(f"Training error output: {remaining_output}")
                        except Exception as err_ex:
                            print(f"Could not read error output: {err_ex}")
                    
                    training_processes[training_id]['end_time'] = datetime.now().isoformat()
                
                # After training is done, clean up uploaded dataset
                try:
                    uploads_path = os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER)
                    if os.path.exists(uploads_path):
                        import shutil
                        shutil.rmtree(uploads_path, ignore_errors=True)
                        # Re-create empty uploads directory for future uploads
                        os.makedirs(uploads_path, exist_ok=True)
                        print(f"Cleaned up uploads directory at {uploads_path} after training {training_id} completed")
                except Exception as cleanup_ex:
                    print(f"Failed to clean uploads directory after training {training_id}: {cleanup_ex}")
            
            thread = threading.Thread(target=monitor_training)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'training_id': training_id,
                'status': 'started',
                'message': 'Training process started'
            })
        else:
            return jsonify({'error': 'Failed to start training'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/status/<training_id>', methods=['GET'])
def get_training_status(training_id):
    """Get training status"""
    if training_id not in training_processes:
        return jsonify({'error': 'Training not found'}), 404
    
    training_info = training_processes[training_id]
    process = training_info['process']
    # Check if process is still running
    if process.poll() is None:
        training_info['status'] = 'running'
    else:
        training_info['status'] = 'completed' if process.returncode == 0 else 'failed'

    # Compose response with serializable fields and friendly names
    resp = {k: v for k, v in training_info.items() if k != 'process'}
    # Backwards compatibility keys
    resp['progress'] = resp.get('progress', 0)
    resp['loss'] = float(resp.get('loss', 0.0))
    resp['accuracy'] = float(resp.get('accuracy', 0.0))
    resp['total_epochs'] = int(resp.get('total_epochs', 0))
    return jsonify(resp)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of trained models and available PT files for detection"""
    try:
        models = []
        backend_dir = os.path.dirname(__file__)
        
        # Get trained models from experiments folder
        experiments_path = os.path.join(backend_dir, MODELS_FOLDER, 'experiments')
        if os.path.exists(experiments_path):
            for model_dir in os.listdir(experiments_path):
                model_path = os.path.join(experiments_path, model_dir)
                
                # Check for best.pt (recommended) and last.pt
                for weight_file in ['best.pt', 'last.pt']:
                    weights_path = os.path.join(model_path, 'weights', weight_file)
                    if os.path.exists(weights_path):
                        stat = os.stat(weights_path)
                        models.append({
                            'id': f"{model_dir}_{weight_file.split('.')[0]}",
                            'name': f"{model_dir} ({weight_file})",
                            'path': weights_path,
                            'size': stat.st_size,
                            'type': 'trained',
                            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            'accuracy': 'N/A'  # Could be extracted from results.csv if needed
                        })
        
        # Get custom uploaded models
        custom_models_path = os.path.join(backend_dir, MODELS_FOLDER, 'custom')
        if os.path.exists(custom_models_path):
            for model_file in os.listdir(custom_models_path):
                if model_file.endswith('.pt'):
                    model_path = os.path.join(custom_models_path, model_file)
                    stat = os.stat(model_path)
                    models.append({
                        'id': f'custom_{model_file.split(".")[0]}',
                        'name': f'{model_file} (Custom)',
                        'path': model_path,
                        'size': stat.st_size,
                        'type': 'custom',
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'accuracy': 'Custom Model'
                    })
        
        # Get pretrained models from project root and backend folder
        pretrained_locations = [
            backend_dir,  # backend folder
            os.path.join(backend_dir, '..'),  # project root
            os.path.join(backend_dir, '..', 'scripts')  # scripts folder
        ]
        
        pretrained_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 'yolo11n.pt']
        
        for location in pretrained_locations:
            for model_name in pretrained_models:
                model_path = os.path.join(location, model_name)
                if os.path.exists(model_path):
                    stat = os.stat(model_path)
                    model_id = f"pretrained_{model_name.split('.')[0]}"
                    
                    # Avoid duplicates
                    if not any(m['id'] == model_id for m in models):
                        models.append({
                            'id': model_id,
                            'name': f"{model_name} (Pretrained)",
                            'path': model_path,
                            'size': stat.st_size,
                            'type': 'pretrained',
                            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            'accuracy': 'Varies by dataset'
                        })
        
        # Sort models: trained first, then custom, then pretrained, then by name
        type_priority = {'trained': 0, 'custom': 1, 'pretrained': 2}
        models.sort(key=lambda x: (type_priority.get(x['type'], 3), x['name']))
        
        return jsonify({
            'models': models,
            'total_count': len(models),
            'trained_count': len([m for m in models if m['type'] == 'trained']),
            'custom_count': len([m for m in models if m['type'] == 'custom']),
            'pretrained_count': len([m for m in models if m['type'] == 'pretrained'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/upload', methods=['POST'])
def upload_model():
    """Upload a trained model file"""
    try:
        if 'model' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        model_file = request.files['model']
        if model_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if model_file and allowed_model_file(model_file.filename):
            filename = secure_filename(model_file.filename)
            
            # Create custom models directory
            backend_dir = os.path.dirname(__file__)
            custom_models_dir = os.path.join(backend_dir, MODELS_FOLDER, 'custom')
            os.makedirs(custom_models_dir, exist_ok=True)
            
            # Save the model file
            model_path = os.path.join(custom_models_dir, filename)
            model_file.save(model_path)
            
            # Get file stats
            stat = os.stat(model_path)
            
            return jsonify({
                'message': 'Model uploaded successfully',
                'model': {
                    'id': f'custom_{filename.split(".")[0]}',
                    'name': f'{filename} (Custom)',
                    'path': model_path,
                    'size': stat.st_size,
                    'type': 'custom',
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                }
            })
        else:
            return jsonify({'error': 'Invalid file type. Only .pt files are allowed'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/single', methods=['POST'])
def detect_single():
    """Run detection on a single image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        config = request.form.to_dict()
        
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            backend_dir = os.path.dirname(__file__)
            upload_dir = os.path.join(backend_dir, UPLOAD_FOLDER)
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, filename)
            image_file.save(image_path)
            
            # Convert string values to appropriate types
            detection_config = {
                'weights_path': config.get('weights_path', ''),
                'confidence': float(config.get('confidence', 0.25)),
                'iou': float(config.get('iou', 0.45))
            }
            
            result = run_detection_script(image_path, detection_config)
            
            if result['success']:
                # Return both original image path and result image path if available
                response_data = {
                    'success': True,
                    'count': result['count'],
                    'image_path': image_path,
                    'output': result.get('output', '')
                }
                
                # Add result image path if available
                if result.get('result_path'):
                    # Convert absolute path to relative filename for API endpoint
                    result_filename = os.path.basename(result['result_path'])
                    response_data['result_image'] = f'/api/results/detection/{result_filename}'
                    
                return jsonify(response_data)
            else:
                return jsonify({'error': result['error']}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/tile', methods=['POST'])
def detect_tile():
    """Run tile inference on large images"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        config = request.form.to_dict()
        
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            backend_dir = os.path.dirname(__file__)
            upload_dir = os.path.join(backend_dir, UPLOAD_FOLDER)
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, filename)
            image_file.save(image_path)
            
            # Convert string values to appropriate types
            tile_config = {
                'weights_path': config.get('weights_path', ''),
                'tile_size': int(config.get('tile_size', 1024)),
                'overlap': int(config.get('overlap', 200)),
                'confidence': float(config.get('confidence', 0.25)),
                'iou': float(config.get('iou', 0.45)),
                'merge_iou': float(config.get('merge_iou', 0.5))
            }
            
            result = run_tile_inference_script(image_path, tile_config)
            
            if result['success']:
                # Return both original image path and result image path if available
                response_data = {
                    'success': True,
                    'count': result['count'],
                    'image_path': image_path,
                    'output': result.get('output', '')
                }
                
                # Add result image path if available
                if result.get('result_path'):
                    # Convert absolute path to relative filename for API endpoint
                    result_filename = os.path.basename(result['result_path'])
                    response_data['result_image'] = f'/api/results/detection/{result_filename}'
                    
                return jsonify(response_data)
            else:
                return jsonify({'error': result['error']}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<path:filepath>')
def get_result_image(filepath):
    """Get result image"""
    try:
        # Build the full path to the result file
        backend_dir = os.path.dirname(__file__)
        result_path = os.path.join(backend_dir, RESULTS_FOLDER, filepath)
        
        # Security check: ensure the path is within the results directory
        results_dir = os.path.abspath(os.path.join(backend_dir, RESULTS_FOLDER))
        abs_result_path = os.path.abspath(result_path)
        
        if not abs_result_path.startswith(results_dir):
            return jsonify({'error': 'Access denied'}), 403
        
        if os.path.exists(result_path):
            return send_file(result_path)
        else:
            return jsonify({'error': f'Result not found: {filepath}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Datasets route (must be defined before app.run)
@app.route('/api/datasets', methods=['GET'])
def list_datasets_api():
    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER)
        if not os.path.exists(uploads_dir):
            return jsonify({'datasets': []})
        datasets = []
        for name in os.listdir(uploads_dir):
            path = os.path.join(uploads_dir, name)
            if os.path.isdir(path):
                datasets.append(name)
        return jsonify({'datasets': datasets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
