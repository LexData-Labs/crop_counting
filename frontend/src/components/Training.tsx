import React, { useState, useEffect } from 'react';
import { 
  PlayIcon, 
  PauseIcon, 
  StopIcon, 
  CogIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { startTraining, getTrainingStatus, getModels } from '../services/api';

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

const Training = () => {
  // Load initial config from localStorage or use defaults
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

  // Save config to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('trainingConfig', JSON.stringify(config));
  }, [config]);

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

  // Fetch model history from backend
  const fetchModels = async () => {
    try {
      const res = await getModels();
      if (res.data && res.data.models) {
        // Map backend models to table format
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

  React.useEffect(() => {
    fetchModels();
  }, []);

  const modelOptions = [
    { value: 'yolov8n.pt', label: 'YOLOv8 Nano (Fastest)' },
    { value: 'yolov8s.pt', label: 'YOLOv8 Small' },
    { value: 'yolov8m.pt', label: 'YOLOv8 Medium' },
    { value: 'yolov8l.pt', label: 'YOLOv8 Large' },
    { value: 'yolov8x.pt', label: 'YOLOv8 Extra Large (Most Accurate)' },
  ];

  const deviceOptions = [
    { value: '0', label: 'GPU (CUDA)' },
    { value: 'cpu', label: 'CPU' },
  ];

  const startTrainingProcess = async () => {
    try {
      // Immediately reflect the configured total epochs in the progress panel
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
      // Poll for training status
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
            setTimeout(pollStatus, 5000); // Poll every 5 seconds
          } else if (trainingData.status === 'completed') {
            fetchModels(); // Refresh model list
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

  const simulateTraining = () => {
    let currentEpoch = 0;
    const interval = setInterval(() => {
      currentEpoch += 1;
      const loss = Math.max(0.1, 1.0 - (currentEpoch / config.epochs) * 0.8);
      const accuracy = Math.min(0.95, (currentEpoch / config.epochs) * 0.9);
      const timeRemaining = Math.max(0, config.epochs - currentEpoch) * 2; // 2 minutes per epoch
      
      setProgress(prev => ({
        ...prev,
        epoch: currentEpoch,
        loss: parseFloat(loss.toFixed(3)),
        accuracy: parseFloat(accuracy.toFixed(3)),
        timeRemaining: `${Math.floor(timeRemaining / 60)}h ${Math.floor(timeRemaining % 60)}m`
      }));

      if (currentEpoch >= config.epochs) {
        clearInterval(interval);
        setProgress(prev => ({ ...prev, status: 'completed' }));
        // Add to training history
        const newModel = {
          id: trainingHistory.length + 1,
          name: `yolov8_crop_detect_v${trainingHistory.length + 1}`,
          accuracy: parseFloat(accuracy.toFixed(3)),
          loss: parseFloat(loss.toFixed(3)),
          date: new Date().toISOString().split('T')[0],
          status: 'completed'
        };
        setTrainingHistory(prev => [newModel, ...prev]);
      }
    }, 1000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Model Training</h1>
        <p className="mt-2 text-gray-600">Configure and train YOLOv8 models for crop detection</p>
      </div>

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

            {/* Apply Config to Progress */}
            <div className="pt-2">
              <button
                onClick={() => {
                  setProgress(prev => ({ ...prev, totalEpochs: config.epochs }));
                  setConfigApplied(true);
                  setTimeout(() => setConfigApplied(false), 2000); // Hide feedback after 2 seconds
                }}
                className={configApplied ? "btn-success" : "btn-secondary"}
                title="Apply current configuration values (e.g., epochs) to the progress panel"
              >
                {configApplied ? '✓ Applied!' : 'Apply to Progress'}
              </button>
              {configApplied && (
                <p className="text-sm text-green-600 mt-1">
                  Configuration applied to training progress panel
                </p>
              )}
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
    </div>
  );
};

export default Training;
