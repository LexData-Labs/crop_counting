# Crop Counting AI - Complete Full-Stack Application

A comprehensive web application for automated crop counting using YOLOv8 deep learning models. This project includes a React frontend with Tailwind CSS and a Flask backend API that integrates with Python machine learning scripts.

## 🚀 Features

### Frontend (React + Tailwind CSS)
- **Dashboard**: Overview with statistics and quick actions
- **Dataset Upload**: Drag-and-drop file upload with progress tracking
- **Model Training**: Real-time training interface with configuration options
- **Model Management**: View, select, and manage trained models
- **Detection Interface**: Upload images and run crop detection
- **Results Visualization**: View detection results with statistics
- **Responsive Design**: Mobile-friendly interface

### Backend (Flask API)
- **File Upload API**: Handle dataset and image uploads
- **Training API**: Start and monitor model training processes
- **Detection API**: Run single image and tile inference
- **Model Management**: List and manage trained models
- **Real-time Progress**: WebSocket-like polling for training progress

### Machine Learning (Python Scripts)
- **YOLOv8 Training**: Complete training pipeline with Ultralytics
- **Single Image Detection**: Fast inference on individual images
- **Tile Inference**: Process large orthomosaics with tiling
- **NMS Merging**: Remove duplicate detections across tiles
- **Evaluation Metrics**: MAE, RMSE, and accuracy calculations

## 📁 Project Structure

```
crop-counting-ai/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API service layer
│   │   └── App.tsx          # Main application
│   ├── public/
│   └── package.json
├── backend/                  # Flask backend API
│   ├── app.py               # Main Flask application
│   ├── requirements.txt     # Python dependencies
│   └── uploads/             # Uploaded files
├── scripts/                 # Python ML scripts
│   ├── train_yolov8.py     # Model training
│   ├── infer_count.py      # Single image detection
│   ├── tile_inference.py   # Large image processing
│   ├── evaluate_counts.py  # Model evaluation
│   ├── merge_nms.py        # NMS merging
│   └── utils.py            # Utility functions
├── data/                    # Dataset storage
│   ├── images/             # Training/validation/test images
│   ├── labels/             # YOLO format labels
│   └── dataset.yaml        # Dataset configuration
├── models/                  # Trained model storage
└── requirements.txt         # Main Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- Python 3.8+
- CUDA (optional, for GPU training)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd crop-counting-ai
```

### 2. Install Dependencies

#### Frontend Dependencies
```bash
cd frontend
npm install
```

#### Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Python ML Dependencies
```bash
# Activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ML dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env.local` file in the frontend directory:
```env
REACT_APP_API_URL=http://localhost:5000/api
```

## 🚀 Running the Application

### Development Mode (Recommended)

#### Option 1: Run Everything Together
```bash
# Install concurrently for running both servers
npm install -g concurrently

# Start both frontend and backend
npm run start-dev
```

#### Option 2: Run Separately

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### Production Mode
```bash
# Build frontend
npm run build

# Serve frontend (using a static server)
# Backend runs on port 5000
```

## 📖 Usage Guide

### 1. Dataset Preparation
1. Go to **Dataset Upload** page
2. Upload images in `data/images/train/`, `data/images/val/`, `data/images/test/`
3. Upload corresponding YOLO label files in `data/labels/train/`, `data/labels/val/`, `data/labels/test/`
4. Ensure `data/dataset.yaml` is properly configured

### 2. Model Training
1. Go to **Training** page
2. Configure training parameters:
   - Model architecture (YOLOv8n/s/m/l/x)
   - Number of epochs
   - Batch size
   - Image size
   - Learning rate
   - Device (GPU/CPU)
3. Click **Start Training**
4. Monitor progress in real-time

### 3. Model Management
1. Go to **Models** page
2. View all trained models with metrics
3. Select active model for detection
4. Download or delete models as needed

### 4. Crop Detection
1. Go to **Detection** page
2. Select trained model and configure parameters
3. Upload images for detection
4. Click **Start Detection**
5. View results with crop counts and confidence scores

## 🔧 Configuration

### Training Configuration
- **Model Architecture**: Choose from YOLOv8 variants (nano to extra-large)
- **Epochs**: Number of training iterations (50-1000)
- **Batch Size**: Images per batch (8-64)
- **Image Size**: Input image resolution (320-2048)
- **Learning Rate**: Training step size (0.0001-1.0)

### Detection Configuration
- **Confidence Threshold**: Minimum detection confidence (0.1-0.9)
- **IoU Threshold**: Non-maximum suppression threshold (0.1-0.9)
- **Tile Size**: Size for tiling large images (512-2048)
- **Overlap**: Overlap between tiles (50-500)
- **Merge IoU**: IoU threshold for merging detections (0.1-0.9)

## 📊 API Endpoints

### Health Check
- `GET /api/health` - Check API status

### File Upload
- `POST /api/upload` - Upload dataset files

### Training
- `POST /api/training/start` - Start model training
- `GET /api/training/status/<id>` - Get training status

### Models
- `GET /api/models` - List trained models

### Detection
- `POST /api/detection/single` - Single image detection
- `POST /api/detection/tile` - Tile inference for large images

## 🎯 Key Features Explained

### Tiling Inference
For large orthomosaic images, the system automatically:
1. Divides the image into overlapping tiles
2. Runs detection on each tile
3. Converts tile coordinates to global coordinates
4. Merges overlapping detections using NMS

### Real-time Training Monitoring
- Live progress updates every 5 seconds
- Loss and accuracy tracking
- Estimated time remaining
- Automatic completion detection

### Responsive Design
- Mobile-first approach
- Touch-friendly interface
- Adaptive layouts for all screen sizes

## 🐛 Troubleshooting

### Common Issues

1. **Import errors in Python scripts**
   - Ensure virtual environment is activated
   - Install all requirements: `pip install -r requirements.txt`

2. **Frontend not connecting to backend**
   - Check if backend is running on port 5000
   - Verify `REACT_APP_API_URL` in `.env.local`

3. **Training fails to start**
   - Check if dataset files are properly uploaded
   - Verify `data/dataset.yaml` exists and is valid

4. **Detection returns no results**
   - Check confidence threshold settings
   - Ensure model weights file exists
   - Verify image format is supported

### Performance Tips

1. **GPU Training**: Use CUDA-enabled PyTorch for faster training
2. **Batch Size**: Increase batch size if you have more GPU memory
3. **Image Size**: Larger images provide better accuracy but slower processing
4. **Tile Size**: Larger tiles reduce processing time but may miss small objects

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [React](https://reactjs.org/) for the frontend framework
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [Flask](https://flask.palletsprojects.com/) for the backend API

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Happy Crop Counting! 🌱🤖**
#   c r o p _ c o u n t i n g  
 