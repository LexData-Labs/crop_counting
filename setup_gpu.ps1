# GPU Setup Script for Crop Counting AI
# This script installs PyTorch with CUDA support for your RTX 4060

Write-Host "🚀 Setting up GPU support for Crop Counting AI..." -ForegroundColor Green
Write-Host "GPU Detected: NVIDIA GeForce RTX 4060 (8GB VRAM)" -ForegroundColor Yellow
Write-Host "CUDA Version: 12.6" -ForegroundColor Yellow

# Step 1: Activate virtual environment
Write-Host "`n📦 Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Step 2: Uninstall CPU-only PyTorch
Write-Host "`n🗑️ Removing CPU-only PyTorch..." -ForegroundColor Cyan
python -m pip uninstall torch torchvision torchaudio -y

# Step 3: Install CUDA-enabled PyTorch
Write-Host "`n⚡ Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

# For CUDA 12.x, use the cu121 version (compatible with CUDA 12.6)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 4: Verify installation
Write-Host "`n✅ Verifying GPU installation..." -ForegroundColor Cyan
python -c "
import torch
print('=' * 50)
print('PyTorch GPU Installation Verification')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print('✅ GPU setup successful!')
else:
    print('❌ GPU setup failed - CUDA not available')
print('=' * 50)
"

Write-Host "`n🎯 GPU setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Start backend: cd backend && python app.py" -ForegroundColor White
Write-Host "2. Start frontend: cd frontend && npm start" -ForegroundColor White  
Write-Host "3. Try training with GPU acceleration! 🚀" -ForegroundColor White