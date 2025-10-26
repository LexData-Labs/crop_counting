<div align="center">

# 🌾 Crop Counting AI

### Automated Crop Detection & Counting using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.9+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.0+-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

A comprehensive full-stack web application for automated crop counting using state-of-the-art YOLOv8 deep learning models. Features a modern React frontend, robust Flask backend API, and powerful Python ML pipeline.

[Features](#-features) •
[Installation](#%EF%B8%8F-installation--setup) •
[Usage](#-usage-guide) •
[API](#-api-endpoints) •
[Contributing](#-contributing)

</div>

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#%EF%B8%8F-installation--setup)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [API Endpoints](#-api-endpoints)
- [Key Features Explained](#-key-features-explained)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Features

### 🎨 Frontend (React + TypeScript + Tailwind CSS)

- ✅ **Interactive Dashboard** - Real-time statistics and quick actions
- ✅ **Drag & Drop Upload** - Intuitive dataset and image upload with progress tracking
- ✅ **Live Training Monitor** - Real-time model training interface with live metrics
- ✅ **Model Management** - View, compare, and manage all trained models
- ✅ **Smart Detection** - Upload images and run crop detection with adjustable parameters
- ✅ **Visual Results** - Beautiful visualization of detection results with statistics
- ✅ **Fully Responsive** - Mobile-first design that works on all devices

### ⚙️ Backend (Flask REST API)

- 🔄 **File Upload System** - Robust handling of dataset and image uploads
- 🚀 **Training Pipeline** - Start, monitor, and manage model training processes
- 🎯 **Detection Engine** - Single image and tile-based inference
- 📊 **Model Registry** - Track and manage all trained model versions
- ⚡ **Real-time Updates** - Live progress tracking for long-running operations
- 🔒 **Error Handling** - Comprehensive error handling and validation

### 🤖 Machine Learning (YOLOv8 Pipeline)

- 🏋️ **Advanced Training** - Full YOLOv8 training pipeline with hyperparameter tuning
- 🖼️ **Single Image Inference** - Fast detection on individual images
- 🗺️ **Large Image Processing** - Tile-based inference for orthomosaics and large images
- 🔗 **Smart Merging** - NMS-based detection merging across tile boundaries
- 📈 **Comprehensive Metrics** - MAE, RMSE, accuracy, precision, and recall calculations
- 💾 **Model Versioning** - Automatic experiment tracking and model management

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="33%">

### Frontend
![React](https://img.shields.io/badge/-React-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/-TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/-Tailwind-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)
![Vite](https://img.shields.io/badge/-Vite-646CFF?style=flat-square&logo=vite&logoColor=white)

</td>
<td align="center" width="33%">

### Backend
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Flask-CORS](https://img.shields.io/badge/-Flask--CORS-000000?style=flat-square&logo=flask&logoColor=white)

</td>
<td align="center" width="33%">

### Machine Learning
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/-YOLOv8-00FFFF?style=flat-square)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</td>
</tr>
</table>

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

### 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v16 or higher) - [Download](https://nodejs.org/)
- **Python** 3.8+ - [Download](https://www.python.org/)
- **Git** - [Download](https://git-scm.com/)
- **CUDA** (optional, for GPU training) - [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/LexData-Labs/crop_counting.git
cd crop_counting
```

### 2️⃣ Install Dependencies

<details>
<summary><b>Frontend Setup</b></summary>

```bash
cd frontend
npm install
```

Create a `.env.local` file:
```env
REACT_APP_API_URL=http://localhost:5000/api
```

</details>

<details>
<summary><b>Backend Setup</b></summary>

```bash
cd backend
pip install -r requirements.txt
```

</details>

<details>
<summary><b>Python ML Scripts Setup</b></summary>

```bash
# Create and activate virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install ML dependencies
pip install -r requirements.txt
```

</details>

### 3️⃣ Dataset Configuration

Ensure your dataset follows the YOLO format:

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

## 🚀 Running the Application

### 🔧 Development Mode

You can run the frontend and backend separately in different terminals:

**Terminal 1 - Start Backend Server:**
```bash
cd backend
python app.py
```
> 🌐 Backend will run on `http://localhost:5000`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm start
```
> 🌐 Frontend will run on `http://localhost:3000`

### 🚢 Production Mode

```bash
# Build frontend
cd frontend
npm run build

# The build folder can be served with any static file server
# Backend continues to run on port 5000
```

### ✅ Verify Installation

Once both servers are running, navigate to `http://localhost:3000` and you should see the dashboard!

## 📖 Usage Guide

### 1️⃣ Dataset Preparation

1. Navigate to the **Dataset Upload** page
2. Organize your data in YOLO format:
   - Images: `data/images/{train,val,test}/`
   - Labels: `data/labels/{train,val,test}/`
3. Upload your dataset through the interface or directly to the folders
4. Verify `data/dataset.yaml` configuration

**Example `dataset.yaml`:**
```yaml
path: ./data
train: images/train
val: images/val
test: images/test
nc: 1  # number of classes
names: ['crop']
```

### 2️⃣ Model Training

1. Navigate to the **Training** page
2. Configure training parameters:

   | Parameter | Description | Recommended |
   |-----------|-------------|-------------|
   | Model Architecture | YOLOv8n/s/m/l/x | YOLOv8n for speed, YOLOv8x for accuracy |
   | Epochs | Training iterations | 100-300 |
   | Batch Size | Images per batch | 8-16 (GPU dependent) |
   | Image Size | Input resolution | 640 or 1280 |
   | Learning Rate | Step size | 0.01 (default) |
   | Device | GPU/CPU | GPU recommended |

3. Click **Start Training** and monitor progress in real-time
4. View training metrics, loss curves, and validation results

### 3️⃣ Model Management

1. Navigate to the **Models** page
2. View all trained models with performance metrics
3. Select the best model for deployment
4. Download model weights or delete unused models

### 4️⃣ Crop Detection

1. Navigate to the **Detection** page
2. Select your trained model
3. Configure detection parameters:
   - **Confidence Threshold**: Minimum detection confidence (default: 0.25)
   - **IoU Threshold**: NMS threshold (default: 0.45)
4. Upload image(s) for detection
5. Click **Start Detection**
6. View results with:
   - Annotated images with bounding boxes
   - Total crop count
   - Individual confidence scores
   - Detection statistics

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

### Health & Status
```http
GET /api/health
```
> Check API server status and availability

### File Management
```http
POST /api/upload
Content-Type: multipart/form-data
```
> Upload dataset files (images and labels)

### Training Operations
```http
POST /api/training/start
Content-Type: application/json
{
  "model": "yolov8n",
  "epochs": 100,
  "batch_size": 16,
  "img_size": 640
}
```
> Start a new training job

```http
GET /api/training/status/<training_id>
```
> Get real-time training progress and metrics

### Model Management
```http
GET /api/models
```
> List all trained models with metadata and metrics

### Detection & Inference
```http
POST /api/detection/single
Content-Type: multipart/form-data
```
> Run detection on a single image

```http
POST /api/detection/tile
Content-Type: multipart/form-data
{
  "tile_size": 1280,
  "overlap": 200,
  "merge_iou": 0.5
}
```
> Process large orthomosaic images using tile-based inference

## 🎯 Key Features Explained

### 🗺️ Tiling Inference for Large Images

Processing large orthomosaic images (e.g., drone imagery) requires special handling:

```
┌─────────────────────────────────┐
│    Large Orthomosaic Image      │
│  ┌────┬────┬────┬────┬────┐     │
│  │ T1 │ T2 │ T3 │ T4 │ T5 │     │  1. Split into overlapping tiles
│  ├────┼────┼────┼────┼────┤     │
│  │ T6 │ T7 │ T8 │ T9 │ T10│     │  2. Run detection on each tile
│  ├────┼────┼────┼────┼────┤     │
│  │ T11│ T12│ T13│ T14│ T15│     │  3. Convert coordinates to global
│  └────┴────┴────┴────┴────┘     │
│                                  │  4. Merge detections using NMS
└─────────────────────────────────┘
```

**Benefits:**
- ✅ Process images larger than GPU memory
- ✅ Maintain high detection accuracy
- ✅ Eliminate duplicate detections at tile boundaries
- ✅ Scalable to any image size

### ⚡ Real-time Training Monitoring

Track your model training with live updates:
- 📊 **Live Metrics**: Loss, accuracy, precision, recall
- ⏱️ **Progress Tracking**: Current epoch, ETA, time elapsed
- 📈 **Visualization**: Training and validation curves
- 🔔 **Notifications**: Completion alerts

### 📱 Responsive Design

Built with mobile-first approach:
- 💻 **Desktop**: Full-featured dashboard with multi-column layouts
- 📱 **Tablet**: Optimized touch interface
- 📞 **Mobile**: Streamlined single-column view
- 🎨 **Dark Mode Ready**: Modern UI with Tailwind CSS

## 🐛 Troubleshooting

### ❌ Common Issues & Solutions

<details>
<summary><b>Import errors in Python scripts</b></summary>

**Problem**: `ModuleNotFoundError` or import errors

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall all dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><b>Frontend not connecting to backend</b></summary>

**Problem**: API connection errors or CORS issues

**Solution**:
1. Verify backend is running on `http://localhost:5000`
2. Check `.env.local` has correct API URL:
   ```env
   REACT_APP_API_URL=http://localhost:5000/api
   ```
3. Ensure Flask-CORS is installed: `pip install flask-cors`

</details>

<details>
<summary><b>Training fails to start</b></summary>

**Problem**: Training won't begin or crashes immediately

**Solution**:
1. Verify `data/dataset.yaml` exists and is properly formatted
2. Check that dataset folders contain images and labels
3. Ensure YOLO format labels are correct (class x_center y_center width height)
4. Verify sufficient disk space for model checkpoints

</details>

<details>
<summary><b>Detection returns no results</b></summary>

**Problem**: No crops detected in images

**Solution**:
1. Lower confidence threshold (try 0.1-0.3)
2. Verify model weights file exists in `models/` directory
3. Check image format is supported (jpg, png, jpeg)
4. Ensure model was trained on similar data

</details>

<details>
<summary><b>CUDA out of memory errors</b></summary>

**Problem**: GPU memory errors during training

**Solution**:
1. Reduce batch size (try 8 or 4)
2. Reduce image size (try 640 instead of 1280)
3. Use a smaller model (yolov8n instead of yolov8x)
4. Clear GPU cache: `torch.cuda.empty_cache()`

</details>

### ⚡ Performance Tips

| Aspect | Recommendation | Impact |
|--------|----------------|--------|
| **GPU Training** | Use CUDA-enabled PyTorch | 10-50x faster training |
| **Batch Size** | Increase based on GPU memory | Faster training, better gradient estimates |
| **Image Size** | 640 for speed, 1280 for accuracy | Accuracy vs speed tradeoff |
| **Tile Size** | 1024-1280 for large images | Balance between speed and edge detection |
| **Model Selection** | YOLOv8n for real-time, YOLOv8x for accuracy | Speed vs accuracy tradeoff |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow existing code style and conventions
- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is built with amazing open-source technologies:

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection
- **[React](https://reactjs.org/)** - Modern UI library
- **[TypeScript](https://www.typescriptlang.org/)** - Type-safe JavaScript
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first CSS framework
- **[Flask](https://flask.palletsprojects.com/)** - Lightweight Python web framework
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

## 📞 Support & Contact

Need help? Here's how to get support:

- 📫 **Issues**: [Create an issue](https://github.com/LexData-Labs/crop_counting/issues)
- 📖 **Documentation**: Check the sections above
- 💬 **Discussions**: Start a [discussion](https://github.com/LexData-Labs/crop_counting/discussions)

## 🌟 Star History

If this project helped you, please consider giving it a ⭐!

---

<div align="center">

**Made with ❤️ by [LexData Labs](https://github.com/LexData-Labs)**

**Happy Crop Counting! 🌱🤖**

[![GitHub Stars](https://img.shields.io/github/stars/LexData-Labs/crop_counting?style=social)](https://github.com/LexData-Labs/crop_counting)
[![GitHub Forks](https://img.shields.io/github/forks/LexData-Labs/crop_counting?style=social)](https://github.com/LexData-Labs/crop_counting/fork)

</div>
