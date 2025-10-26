#!/bin/bash

echo "🌱 Starting Crop Counting AI Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create necessary directories
echo "Creating directories..."
mkdir -p backend/uploads
mkdir -p backend/models
mkdir -p backend/results
mkdir -p data/images/train
mkdir -p data/images/val
mkdir -p data/images/test
mkdir -p data/labels/train
mkdir -p data/labels/val
mkdir -p data/labels/test

# Start the application
echo "Starting application..."
echo "Frontend will be available at: http://localhost:3000"
echo "Backend API will be available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the application"

# Run both frontend and backend
npx concurrently "cd backend && python app.py" "cd frontend && npm start"
