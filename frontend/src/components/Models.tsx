import React, { useState } from 'react';
import { 
  BeakerIcon, 
  EyeIcon, 
  TrashIcon, 
  ArrowDownTrayIcon as DownloadIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

interface Model {
  id: string;
  name: string;
  architecture: string;
  accuracy: number;
  loss: number;
  size: string;
  date: string;
  status: 'trained' | 'training' | 'error';
  epochs: number;
  batchSize: number;
  imageSize: number;
}

const Models = () => {
  const [models, setModels] = useState<Model[]>([
    {
      id: '1',
      name: 'yolov8_crop_detect_v1',
      architecture: 'YOLOv8 Nano',
      accuracy: 0.89,
      loss: 0.15,
      size: '6.2 MB',
      date: '2024-01-15',
      status: 'trained',
      epochs: 50,
      batchSize: 16,
      imageSize: 1024
    },
    {
      id: '2',
      name: 'yolov8_crop_detect_v2',
      architecture: 'YOLOv8 Small',
      accuracy: 0.92,
      loss: 0.12,
      size: '21.5 MB',
      date: '2024-01-14',
      status: 'trained',
      epochs: 75,
      batchSize: 12,
      imageSize: 1280
    },
    {
      id: '3',
      name: 'yolov8_crop_detect_v3',
      architecture: 'YOLOv8 Medium',
      accuracy: 0.87,
      loss: 0.18,
      size: '49.7 MB',
      date: '2024-01-13',
      status: 'trained',
      epochs: 100,
      batchSize: 8,
      imageSize: 1024
    }
  ]);

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState<string | null>(null);

  const deleteModel = (modelId: string) => {
    setModels(prev => prev.filter(model => model.id !== modelId));
    if (selectedModel === modelId) {
      setSelectedModel(null);
    }
  };

  const downloadModel = (model: Model) => {
    // Simulate download
    console.log(`Downloading ${model.name}...`);
    // In a real app, this would trigger a download
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'trained':
        return 'text-green-600 bg-green-100';
      case 'training':
        return 'text-blue-600 bg-blue-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'trained':
        return <CheckCircleIcon className="h-4 w-4" />;
      case 'training':
        return <ClockIcon className="h-4 w-4" />;
      case 'error':
        return <TrashIcon className="h-4 w-4" />;
      default:
        return <BeakerIcon className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Trained Models</h1>
        <p className="mt-2 text-gray-600">Manage and select your trained YOLOv8 models for crop detection</p>
      </div>

      {/* Model Selection */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Select Active Model</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              className={`border-2 rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                selectedModel === model.id
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => setSelectedModel(model.id)}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-gray-900">{model.name}</h4>
                <span className={`inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(model.status)}`}>
                  {getStatusIcon(model.status)}
                  <span className="ml-1 capitalize">{model.status}</span>
                </span>
              </div>
              <p className="text-sm text-gray-600 mb-2">{model.architecture}</p>
              <div className="flex justify-between text-sm text-gray-500">
                <span>Accuracy: {(model.accuracy * 100).toFixed(1)}%</span>
                <span>{model.size}</span>
              </div>
            </div>
          ))}
        </div>
        {selectedModel && (
          <div className="mt-4 p-4 bg-primary-50 rounded-lg">
            <p className="text-sm text-primary-700">
              <strong>Selected:</strong> {models.find(m => m.id === selectedModel)?.name}
            </p>
          </div>
        )}
      </div>

      {/* Model Details Table */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Model Details</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Architecture
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Loss
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Size
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {models.map((model) => (
                <tr key={model.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <BeakerIcon className="h-5 w-5 text-gray-400 mr-3" />
                      <div>
                        <div className="text-sm font-medium text-gray-900">{model.name}</div>
                        <div className="text-sm text-gray-500">Epochs: {model.epochs}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {model.architecture}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <ChartBarIcon className="h-4 w-4 text-green-500 mr-2" />
                      <span className="text-sm text-gray-900">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {model.loss}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {model.size}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {model.date}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex space-x-2">
                      <button
                        onClick={() => setShowDetails(showDetails === model.id ? null : model.id)}
                        className="text-primary-600 hover:text-primary-900"
                      >
                        <EyeIcon className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => downloadModel(model)}
                        className="text-green-600 hover:text-green-900"
                      >
                        <DownloadIcon className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => deleteModel(model.id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Model Information */}
      {showDetails && (
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Model Details</h3>
          {(() => {
            const model = models.find(m => m.id === showDetails);
            if (!model) return null;
            
            return (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Training Configuration</h4>
                  <dl className="space-y-1">
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Architecture:</dt>
                      <dd className="text-sm text-gray-900">{model.architecture}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Epochs:</dt>
                      <dd className="text-sm text-gray-900">{model.epochs}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Batch Size:</dt>
                      <dd className="text-sm text-gray-900">{model.batchSize}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Image Size:</dt>
                      <dd className="text-sm text-gray-900">{model.imageSize}px</dd>
                    </div>
                  </dl>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Performance Metrics</h4>
                  <dl className="space-y-1">
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Accuracy:</dt>
                      <dd className="text-sm text-gray-900">{(model.accuracy * 100).toFixed(1)}%</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Loss:</dt>
                      <dd className="text-sm text-gray-900">{model.loss}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Model Size:</dt>
                      <dd className="text-sm text-gray-900">{model.size}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-500">Status:</dt>
                      <dd className="text-sm text-gray-900 capitalize">{model.status}</dd>
                    </div>
                  </dl>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Actions</h4>
                  <div className="space-y-2">
                    <button
                      onClick={() => setSelectedModel(model.id)}
                      className="w-full btn-primary text-sm"
                    >
                      Select as Active
                    </button>
                    <button
                      onClick={() => downloadModel(model)}
                      className="w-full btn-secondary text-sm"
                    >
                      Download Model
                    </button>
                    <button
                      onClick={() => deleteModel(model.id)}
                      className="w-full btn-error text-sm"
                    >
                      Delete Model
                    </button>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};

export default Models;
