#!/usr/bin/env python3
"""
Debug detection API with a test image
"""

import requests
import json
import os
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test image"""
    # Create a simple 300x300 test image
    img = Image.new('RGB', (300, 300), color='green')
    
    # Add some simple shapes that might be detected
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw some rectangles
    draw.rectangle([50, 50, 100, 100], fill='red')
    draw.rectangle([150, 150, 200, 200], fill='blue') 
    draw.rectangle([200, 50, 250, 100], fill='yellow')
    
    # Save test image
    test_image_path = 'test_image.jpg'
    img.save(test_image_path)
    print(f"Created test image: {test_image_path}")
    return test_image_path

def test_detection_api():
    """Test the detection API with debugging"""
    try:
        # Create test image
        test_image_path = create_test_image()
        
        # Get available models first
        print("\n1. Getting available models...")
        models_response = requests.get('http://localhost:5000/api/models')
        
        if models_response.status_code != 200:
            print(f"❌ Models endpoint failed: {models_response.status_code}")
            print(f"Response: {models_response.text}")
            return
        
        models_data = models_response.json()
        models = models_data.get('models', [])
        trained_models = [m for m in models if m['type'] == 'trained']
        
        print(f"✅ Found {len(models)} total models")
        print(f"✅ Found {len(trained_models)} trained models")
        
        if not trained_models:
            print("❌ No trained models available!")
            return
        
        # Use the first trained model
        selected_model = trained_models[0]
        model_path = selected_model['path']
        print(f"Selected model: {selected_model['name']}")
        print(f"Model path: {model_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        
        # Test detection API
        print(f"\n2. Testing detection API...")
        
        with open(test_image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {
                'weights_path': model_path,
                'confidence': '0.25',
                'iou': '0.45'
            }
            
            print(f"Making request to: http://localhost:5000/api/detection/single")
            print(f"Form data: {data}")
            
            response = requests.post(
                'http://localhost:5000/api/detection/single',
                files=files,
                data=data,
                timeout=30
            )
        
        print(f"\n3. Response:")
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection successful!")
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            print("❌ Detection failed!")
            print(f"Error response: {response.text}")
            
            # Try to get more details from backend logs
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print("Raw error response:", response.text)
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"Cleaned up test image")
            
    except Exception as e:
        print(f"❌ Exception during test: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_detection_api()