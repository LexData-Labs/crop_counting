#!/usr/bin/env python3
"""
Start the crop counting system
- Backend: Flask API server
- Frontend: Serve the built React app
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check if trained model exists
    model_path = "D:\\New folder\\liitle\\2\\crop_counting\\backend\\models\\experiments\\yolov8n_crop_batch8_100epochs\\weights\\best.pt"
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Trained model not found at {model_path}")
        return False
    
    print(f"✅ Trained model found: {model_path}")
    print(f"   Model size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    
    # Check if backend exists
    backend_path = "D:\\New folder\\liitle\\2\\crop_counting\\backend\\app.py"
    if not os.path.exists(backend_path):
        print(f"❌ ERROR: Backend not found at {backend_path}")
        return False
    
    print("✅ Backend found")
    
    # Check if frontend build exists
    frontend_build = "D:\\New folder\\liitle\\2\\crop_counting\\frontend\\build\\index.html"
    if not os.path.exists(frontend_build):
        print(f"⚠️  Frontend build not found at {frontend_build}")
        print("   The system will run in API-only mode")
    else:
        print("✅ Frontend build found")
    
    return True

def update_model_path_in_backend():
    """Update the backend to use the correct trained model path"""
    print("🔧 Updating backend configuration...")
    
    # The path to our trained model
    trained_model_path = "D:\\\\New folder\\\\liitle\\\\2\\\\crop_counting\\\\backend\\\\models\\\\experiments\\\\yolov8n_crop_batch8_100epochs\\\\weights\\\\best.pt"
    
    # Create a simple config file for the model path
    clean_path = trained_model_path.replace("\\\\", "\\")
    config_content = f'''# Trained model configuration
TRAINED_MODEL_PATH = r"{clean_path}"

# Default detection settings optimized for crop counting
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 200
DEFAULT_MERGE_IOU = 0.5

import os
print("Using trained crop detection model:")
print(f"   Path: {{TRAINED_MODEL_PATH}}")
if os.path.exists(TRAINED_MODEL_PATH):
    print(f"   Size: {{os.path.getsize(TRAINED_MODEL_PATH) / 1024 / 1024:.1f}} MB")
else:
    print("   Status: NOT FOUND")
'''
    
    config_path = "D:\\New folder\\liitle\\2\\crop_counting\\backend\\model_config.py"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ Backend configuration updated")
    return config_path

def start_backend():
    """Start the Flask backend server"""
    print("🚀 Starting backend server...")
    
    try:
        # Change to backend directory
        os.chdir("D:\\New folder\\liitle\\2\\crop_counting\\backend")
        
        # Start Flask app
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor backend output
        def monitor_backend():
            print("📊 Backend Output:")
            print("-" * 50)
            for line in process.stdout:
                if line.strip():
                    print(f"[BACKEND] {line.strip()}")
                    # Look for success indicators
                    if "Running on" in line or "* Serving" in line:
                        print("✅ Backend server started successfully!")
                        print("🌐 Backend API available at: http://localhost:5000")
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_backend)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def test_backend_api():
    """Test if backend API is responding"""
    print("\n🔬 Testing backend API...")
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend API is responding!")
            print(f"   Status: {data.get('status')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"❌ Backend API error: {response.status_code}")
            return False
            
    except ImportError:
        print("⚠️  'requests' library not installed, skipping API test")
        print("   You can install it with: pip install requests")
        return True  # Don't fail just because requests isn't installed
        
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def show_usage_instructions():
    """Show instructions for using the system"""
    print("\n" + "🎉 CROP COUNTING SYSTEM STARTED!" + "\n" + "=" * 50)
    
    print("📡 BACKEND API:")
    print("   • Base URL: http://localhost:5000")
    print("   • Health Check: http://localhost:5000/api/health")
    print("   • Models List: http://localhost:5000/api/models")
    print("   • Device Info: http://localhost:5000/api/device/recommendations")
    
    print("\n🔧 API ENDPOINTS:")
    print("   • POST /api/detection/single - Single image detection")
    print("   • POST /api/detection/tile - Large image tile detection")
    print("   • GET /api/models - List available models")
    
    print("\n🏗️  TRAINED MODEL:")
    model_path = "D:\\New folder\\liitle\\2\\crop_counting\\backend\\models\\experiments\\yolov8n_crop_batch8_100epochs\\weights\\best.pt"
    print(f"   • Path: {model_path}")
    if os.path.exists(model_path):
        print(f"   • Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        print("   • Status: ✅ Ready for inference")
    
    print("\n💡 TESTING:")
    print("   • Use Postman, curl, or any REST client to test the API")
    print("   • Upload crop images to get detection counts")
    print("   • The model is trained for 100 epochs with 68.9% mAP50")
    
    print("\n⏹️  TO STOP:")
    print("   • Press Ctrl+C to stop the backend server")
    print("   • All processes will be terminated gracefully")
    
    print("\n🎯 READY FOR TESTING!")
    print("=" * 50)

def main():
    """Main function to start the crop counting system"""
    print("🌾 CROP COUNTING SYSTEM STARTUP")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        return False
    
    # Update backend configuration
    update_model_path_in_backend()
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend server.")
        return False
    
    # Test backend API
    if test_backend_api():
        show_usage_instructions()
    
    try:
        # Keep the system running
        print("\n🔄 System is running... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping crop counting system...")
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
        print("✅ System stopped successfully.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)