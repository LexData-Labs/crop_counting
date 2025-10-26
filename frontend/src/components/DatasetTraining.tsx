import React, { useState, useCallback, useEffect } from 'react';
import { 
  CloudArrowUpIcon as CloudUploadIcon, 
  DocumentIcon, 
  PhotoIcon, 
  PlayIcon, 
  PauseIcon, 
  StopIcon, 
  CogIcon,
  ClockIcon,
  CheckCircleIcon,
  ChevronRightIcon,
  ChevronLeftIcon
} from '@heroicons/react/24/outline';
import { getDatasets, startTraining, getTrainingStatus, getModels } from '../services/api';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'uploading' | 'completed' | 'error';
  progress: number;
}

interface TrainingConfig {
  model: string;
  epochs: number;
  batchSize: number;
  imageSize: number;
  learningRate: number;
  device: string;
  workers: number;
}

interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  timeRemaining: string;
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error' | 'failed';
}

type Step = 'upload' | 'training';

const DatasetTraining = () => {
  const [currentStep, setCurrentStep] = useState<Step>('upload');
  
  // Dataset Upload State
  const [datasets, setDatasets] = useState<string[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  
  // Training State
  const [config, setConfig] = useState<TrainingConfig>(() => {
    const savedConfig = localStorage.getItem('trainingConfig');
    return savedConfig ? JSON.parse(savedConfig) : {
      model: 'yolov8n.pt',
      epochs: 50,
      batchSize: 16,
      imageSize: 1024,
      learningRate: 0.01,
      device: '0',
      workers: 6
    };
  });

  const [progress, setProgress] = useState<TrainingProgress>({
    epoch: 0,
    totalEpochs: 50,
    loss: 0,
    accuracy: 0,
    timeRemaining: '0h 0m',
    status: 'idle'
  });

  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [configApplied, setConfigApplied] = useState<boolean>(false);

  // Save config to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('trainingConfig', JSON.stringify(config));
  }, [config]);

  // Load datasets and models
  const fetchDatasets = useCallback(async () => {
    setLoadingDatasets(true);
    setDatasetsError(null);
    try {
      const resp = await getDatasets();
      const list = (resp as any)?.data?.datasets || [];
      setDatasets(list);
    } catch (err: any) {
      const msg = err?.response?.data?.error || err?.message || 'Failed to load datasets';
      setDatasetsError(msg);
    } finally {
      setLoadingDatasets(false);
    }
  }, []);

  const fetchModels = async () => {
    try {
      const res = await getModels();
      if (res.data && res.data.models) {
        setTrainingHistory(res.data.models.map((m: any, idx: number) => ({
          id: idx + 1,
          name: m.name,
          accuracy: m.accuracy || 0,
          loss: m.loss || 0,
          date: m.created ? m.created.split('T')[0] : '',
          status: 'completed',
        })));
      }
    } catch (err) {
      setErrorMsg('Failed to fetch model history');
    }
  };

  useEffect(() => {
    fetchDatasets();
    fetchModels();
  }, [fetchDatasets]);

  // Upload handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      handleFiles(files);
    }
  }, []);

  // Helper to split array into chunks
  function chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const results: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      results.push(array.slice(i, i + chunkSize));
    }
    return results;
  }

  const handleFiles = async (files: File[]) => {
    setIsUploading(true);
    setUploadProgress(0);
    const relPaths = files.map(f => (f as any).webkitRelativePath || f.name);
    const newFiles: UploadedFile[] = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading',
      progress: 0
    }));
    setUploadedFiles(newFiles);

    const BATCH_SIZE = 500;
    const fileChunks = chunkArray(files, BATCH_SIZE);
    const relPathChunks = chunkArray(relPaths, BATCH_SIZE);
    let uploadedCount = 0;
    let hasError = false;

    for (let i = 0; i < fileChunks.length; i++) {
      const chunk = fileChunks[i];
      const relChunk = relPathChunks[i];
      await new Promise<void>((resolve) => {
        const formData = new FormData();
        chunk.forEach(file => {
          formData.append('files', file);
        });
        formData.append('relative_paths', JSON.stringify(relChunk));
        const xhr = new XMLHttpRequest();
        xhr.open('POST', process.env.REACT_APP_API_URL ? process.env.REACT_APP_API_URL + '/upload' : 'http://localhost:5000/api/upload', true);
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const chunkProgress = Math.round((event.loaded / event.total) * 100);
            const overall = Math.round(((uploadedCount + (chunkProgress / 100) * chunk.length) / files.length) * 100);
            setUploadProgress(overall);
          }
        };
        xhr.onload = () => {
          uploadedCount += chunk.length;
          setUploadProgress(Math.round((uploadedCount / files.length) * 100));
          if (xhr.status === 200) {
            setUploadedFiles(prev => prev.map((f, idx) =>
              idx < uploadedCount ? { ...f, status: 'completed', progress: 100 } : f
            ));
          } else {
            hasError = true;
            setUploadedFiles(prev => prev.map((f, idx) =>
              idx >= uploadedCount && idx < uploadedCount + chunk.length ? { ...f, status: 'error', progress: 0 } : f
            ));
          }
          resolve();
        };
        xhr.onerror = () => {
          hasError = true;
          setUploadedFiles(prev => prev.map((f, idx) =>
            idx >= uploadedCount && idx < uploadedCount + chunk.length ? { ...f, status: 'error', progress: 0 } : f
          ));
          resolve();
        };
        xhr.send(formData);
      });
      if (hasError) break;
    }
    setIsUploading(false);
    setUploadProgress(100);
    try { await fetchDatasets(); } catch {}
  };

  // Training handlers
  const startTrainingProcess = async () => {
    try {
      setProgress(prev => ({ ...prev, status: 'training', totalEpochs: config.epochs }));
      setErrorMsg(null);
      const response = await startTraining({
        model: config.model,
        epochs: config.epochs,
        batch_size: config.batchSize,
        image_size: config.imageSize,
        learning_rate: config.learningRate,
        device: config.device,
        workers: config.workers
      });
      const trainingId = response.data.training_id;
      const pollStatus = async () => {
        try {
          const statusResponse = await getTrainingStatus(trainingId);
          const trainingData = statusResponse.data;
          setProgress(prev => ({
            ...prev,
            epoch: trainingData.progress || 0,
            totalEpochs: (trainingData.total_epochs ?? prev.totalEpochs) || prev.totalEpochs,
            status: trainingData.status,
            loss: trainingData.loss || 0,
            accuracy: trainingData.accuracy || 0
          }));
          if (trainingData.status === 'running') {
            setTimeout(pollStatus, 5000);
          } else if (trainingData.status === 'completed') {
            fetchModels();
          } else if (trainingData.status === 'failed') {
            setErrorMsg('Training failed. Check backend logs for details.');
          }
        } catch (error) {
          setErrorMsg('Error polling training status');
          setProgress(prev => ({ ...prev, status: 'error' }));
        }
      };
      pollStatus();
    } catch (error: any) {
      setErrorMsg('Error starting training: ' + (error?.response?.data?.error || error.message));
      setProgress(prev => ({ ...prev, status: 'error' }));
    }
  };

  const pauseTraining = () => {
    setProgress(prev => ({ ...prev, status: 'paused' }));
  };

  const stopTraining = () => {
    setProgress(prev => ({ ...prev, status: 'idle', epoch: 0 }));
  };

  // Helper functions
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) {
      return <PhotoIcon className="h-8 w-8 text-blue-500" />;
    }
    return <DocumentIcon className="h-8 w-8 text-gray-500" />;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'uploading':
      case 'training':
        return 'text-blue-600';
      case 'paused':
        return 'text-yellow-600';
      case 'completed':
        return 'text-green-600';
      case 'error':
      case 'failed':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'training':
        return <PlayIcon className="h-5 w-5" />;
      case 'paused':
        return <PauseIcon className="h-5 w-5" />;
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5" />;
      default:
        return <StopIcon className="h-5 w-5" />;
    }
  };

  const modelOptions = [
    { value: 'yolov8n.pt', label: 'YOLOv8 Nano (Fastest, Recommended)' },
    { value: 'yolov8s.pt', label: 'YOLOv8 Small (Good Balance)' },
    { value: 'yolov8m.pt', label: 'YOLOv8 Medium (Better Accuracy)' },
    { value: 'yolov8l.pt', label: 'YOLOv8 Large (High Accuracy)' },
    { value: 'yolov8x.pt', label: 'YOLOv8 Extra Large (Highest Accuracy, May require PyTorch 2.4)' },
  ];

  const deviceOptions = [
    { value: '0', label: 'GPU (CUDA)' },
    { value: 'cpu', label: 'CPU' },
  ];

  const hasUploadedData = datasets.length > 0 || uploadedFiles.some(f => f.status === 'completed');

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dataset & Training</h1>
        <p className="mt-2 text-gray-600">Upload your dataset and train YOLOv8 models in one seamless workflow</p>
      </div>

      {/* Step Progress Indicator */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
              currentStep === 'upload' ? 'bg-primary-600 text-white' : 
              hasUploadedData ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'
            }`}>
              {hasUploadedData && currentStep !== 'upload' ? '✓' : '1'}
            </div>
            <span className={`font-medium ${currentStep === 'upload' ? 'text-primary-600' : hasUploadedData ? 'text-green-600' : 'text-gray-600'}`}>
              Upload Dataset
            </span>
          </div>
          
          <ChevronRightIcon className="h-5 w-5 text-gray-400" />
          
          <div className="flex items-center space-x-4">
            <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
              currentStep === 'training' ? 'bg-primary-600 text-white' : 
              !hasUploadedData ? 'bg-gray-300 text-gray-600' : 'bg-gray-200 text-gray-700'
            }`}>
              2
            </div>
            <span className={`font-medium ${currentStep === 'training' ? 'text-primary-600' : !hasUploadedData ? 'text-gray-400' : 'text-gray-600'}`}>
              Configure & Train
            </span>
          </div>
        </div>
        
        {/* Step Navigation */}
        <div className="flex justify-between">
          <button
            onClick={() => setCurrentStep('upload')}
            className={`btn-secondary flex items-center ${currentStep === 'upload' ? 'bg-primary-100 border-primary-300' : ''}`}
          >
            <ChevronLeftIcon className="h-4 w-4 mr-2" />
            Upload Step
          </button>
          <button
            onClick={() => setCurrentStep('training')}
            disabled={!hasUploadedData}
            className={`btn-primary flex items-center ${!hasUploadedData ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            Training Step
            <ChevronRightIcon className="h-4 w-4 ml-2" />
          </button>
        </div>
      </div>

      {/* Dataset Upload Step */}
      {currentStep === 'upload' && (
        <>
          {/* Upload Area */}
          <div className="card">
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors duration-200 ${
                isDragging
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <CloudUploadIcon className="mx-auto h-12 w-12 text-gray-400" />
              <div className="mt-4">
                <label htmlFor="file-upload" className="cursor-pointer">
                  <span className="mt-2 block text-sm font-medium text-gray-900">
                    Drop files or folders here or click to upload
                  </span>
                  <span className="mt-1 block text-sm text-gray-500">
                    Images (JPG, PNG) and YOLO label files (TXT). You can select a folder for large datasets.
                  </span>
                </label>
                <input
                  id="file-upload"
                  name="file-upload"
                  type="file"
                  className="sr-only"
                  multiple
                  accept="image/*,.txt"
                  // @ts-ignore
                  webkitdirectory="true"
                  directory="true"
                  onChange={handleFileInput}
                />
              </div>
            </div>
          </div>

          {/* Upload Progress */}
          {uploadedFiles.length > 0 && (
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Upload Progress</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Progress</span>
                  <span className={`text-xs ${uploadProgress === 100 ? 'text-green-600' : 'text-blue-600'}`}>{uploadProgress}%</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <div className="mt-2 text-xs text-gray-500">
                  {isUploading
                    ? 'Uploading...'
                    : uploadedFiles.every(f => f.status === 'completed')
                      ? 'Upload completed'
                      : uploadedFiles.some(f => f.status === 'error')
                        ? 'Upload failed'
                        : ''}
                </div>
              </div>
            </div>
          )}

          {/* Uploaded Datasets */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Uploaded Datasets</h3>
              <button onClick={fetchDatasets} className="btn-secondary text-sm" disabled={loadingDatasets}>
                {loadingDatasets ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
            {loadingDatasets ? (
              <div className="text-gray-500">Loading...</div>
            ) : datasetsError ? (
              <div className="text-red-600">{datasetsError}</div>
            ) : datasets.length === 0 ? (
              <div className="text-gray-500">No datasets found.</div>
            ) : (
              <ul className="list-disc pl-6">
                {datasets.map(ds => (
                  <li key={ds} className="text-gray-800">{ds}</li>
                ))}
              </ul>
            )}
          </div>

          {/* Dataset Structure Info */}
          <div className="card">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Dataset Structure</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-2">
                Your dataset should follow this structure:
              </p>
              <pre className="text-xs text-gray-800 bg-white p-3 rounded border overflow-x-auto">
{`data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/   # YOLO txt files
    ├── val/
    └── test/`}
              </pre>
              <p className="text-sm text-gray-600 mt-2">
                Each image should have a corresponding .txt file with the same name containing YOLO format annotations.
              </p>
            </div>
          </div>
        </>
      )}

      {/* Training Step */}
      {currentStep === 'training' && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Training Configuration */}
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Training Configuration</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Model Architecture
                  </label>
                  <select
                    value={config.model}
                    onChange={(e) => setConfig(prev => ({ ...prev, model: e.target.value }))}
                    className="input-field"
                  >
                    {modelOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Epochs
                    </label>
                    <input
                      type="number"
                      value={config.epochs}
                      onChange={(e) => setConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                      className="input-field"
                      min="1"
                      max="1000"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Batch Size
                    </label>
                    <input
                      type="number"
                      value={config.batchSize}
                      onChange={(e) => setConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                      className="input-field"
                      min="1"
                      max="64"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Image Size
                    </label>
                    <input
                      type="number"
                      value={config.imageSize}
                      onChange={(e) => setConfig(prev => ({ ...prev, imageSize: parseInt(e.target.value) }))}
                      className="input-field"
                      min="320"
                      max="2048"
                      step="32"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Learning Rate
                    </label>
                    <input
                      type="number"
                      value={config.learningRate}
                      onChange={(e) => setConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                      className="input-field"
                      min="0.0001"
                      max="1"
                      step="0.001"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Device
                    </label>
                    <select
                      value={config.device}
                      onChange={(e) => setConfig(prev => ({ ...prev, device: e.target.value }))}
                      className="input-field"
                    >
                      {deviceOptions.map(option => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Workers
                    </label>
                    <input
                      type="number"
                      value={config.workers}
                      onChange={(e) => setConfig(prev => ({ ...prev, workers: parseInt(e.target.value) }))}
                      className="input-field"
                      min="1"
                      max="16"
                    />
                  </div>
                </div>

                {/* Save & Apply Config Button */}
                <div className="pt-4 border-t border-gray-200">
                  <button
                    onClick={() => {
                      // Save configuration to localStorage
                      localStorage.setItem('trainingConfig', JSON.stringify(config));
                      
                      // Apply configuration to progress tracker
                      setProgress(prev => ({ 
                        ...prev, 
                        totalEpochs: config.epochs,
                        // Reset progress if config changes during idle state
                        ...(prev.status === 'idle' && { epoch: 0, loss: 0, accuracy: 0, timeRemaining: '0h 0m' })
                      }));
                      
                      // Show success feedback
                      setConfigApplied(true);
                      setTimeout(() => setConfigApplied(false), 3000); // Hide feedback after 3 seconds
                    }}
                    className={`w-full ${configApplied ? "bg-green-600 hover:bg-green-700" : "bg-blue-600 hover:bg-blue-700"} text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center shadow-sm`}
                    title="Save configuration to localStorage and apply to training progress"
                  >
                    {configApplied ? (
                      <>
                        <CheckCircleIcon className="h-4 w-4 mr-2" />
                        Configuration Saved & Applied!
                      </>
                    ) : (
                      <>
                        <CogIcon className="h-4 w-4 mr-2" />
                        Save & Apply Configuration
                      </>
                    )}
                  </button>
                  {configApplied && (
                    <p className="text-sm text-green-600 mt-2 text-center font-medium">
                      ✓ Configuration saved to browser storage and applied to progress tracker
                    </p>
                  )}
                  <p className="text-xs text-gray-500 mt-2 text-center">
                    Configuration is automatically saved when modified, but click above to apply changes to the progress panel
                  </p>
                </div>
              </div>
            </div>

            {/* Training Progress */}
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Training Progress</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Status</span>
                  <span className={`flex items-center ${getStatusColor(progress.status)}`}>
                    {getStatusIcon(progress.status)}
                    <span className="ml-1 capitalize">{progress.status}</span>
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Epoch</span>
                  <span className="text-sm text-gray-900">
                    {progress.epoch} / {progress.totalEpochs}
                  </span>
                </div>

                <div>
                  <div className="flex justify-between text-sm text-gray-700 mb-1">
                    <span>Progress</span>
                    <span>{Math.round((progress.epoch / progress.totalEpochs) * 100)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${(progress.epoch / progress.totalEpochs) * 100}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm font-medium text-gray-700">Loss</span>
                    <p className="text-lg font-semibold text-gray-900">{progress.loss}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-700">Accuracy</span>
                    <p className="text-lg font-semibold text-gray-900">{(progress.accuracy * 100).toFixed(1)}%</p>
                  </div>
                </div>

                <div className="flex items-center text-sm text-gray-700">
                  <ClockIcon className="h-4 w-4 mr-2" />
                  <span>Time Remaining: {progress.timeRemaining}</span>
                </div>

                <div className="flex space-x-2 pt-4">
                  {progress.status === 'idle' && (
                    <button onClick={startTrainingProcess} className="btn-primary flex items-center">
                      <PlayIcon className="h-4 w-4 mr-2" />
                      Start Training
                    </button>
                  )}
                  {progress.status === 'training' && (
                    <button onClick={pauseTraining} className="btn-warning flex items-center">
                      <PauseIcon className="h-4 w-4 mr-2" />
                      Pause
                    </button>
                  )}
                  {progress.status === 'paused' && (
                    <button onClick={startTrainingProcess} className="btn-primary flex items-center">
                      <PlayIcon className="h-4 w-4 mr-2" />
                      Resume
                    </button>
                  )}
                  {(progress.status === 'training' || progress.status === 'paused') && (
                    <button onClick={stopTraining} className="btn-error flex items-center">
                      <StopIcon className="h-4 w-4 mr-2" />
                      Stop
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Training History */}
          <div className="card">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Training History</h3>
            {errorMsg && <div className="text-red-600 mb-2">{errorMsg}</div>}
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Model Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Accuracy
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Loss
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {trainingHistory.length === 0 ? (
                    <tr><td colSpan={5} className="text-center py-4 text-gray-500">No models found</td></tr>
                  ) : trainingHistory.map((model) => (
                    <tr key={model.id}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {model.name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {(model.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {model.loss}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {model.date}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          model.status === 'completed' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {model.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default DatasetTraining;