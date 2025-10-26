import React, { useState, useCallback, useEffect } from 'react';
import { 
  EyeIcon, 
  CloudArrowUpIcon as CloudUploadIcon, 
  PlayIcon, 
  ArrowDownTrayIcon as DownloadIcon,
  PhotoIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  PlusIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { detectSingle, detectTile, getModels, uploadModel, getResultImage } from '../services/api';

interface Model {
  id: string;
  name: string;
  path: string;
  size: number;
  type: 'trained' | 'custom' | 'pretrained';
  created: string;
  accuracy: string;
}

interface DetectionConfig {
  model: string;
  confidence: number;
  iou: number;
  tileSize: number;
  overlap: number;
  mergeIou: number;
}

interface DetectionResult {
  id: string;
  imageName: string;
  count: number;
  confidence: number;
  processingTime: number;
  timestamp: string;
  imageUrl?: string;
  resultUrl?: string;
}

const Detection = () => {
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  
  const [config, setConfig] = useState<DetectionConfig>({
    model: '',
    confidence: 0.25,
    iou: 0.45,
    tileSize: 1024,
    overlap: 200,
    mergeIou: 0.5
  });

  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [results, setResults] = useState<DetectionResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<DetectionResult | null>(null);
  const [uploadingModel, setUploadingModel] = useState(false);
  const [showModelUpload, setShowModelUpload] = useState(false);

  // Load available models on component mount
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoadingModels(true);
      setModelsError(null);
      const response = await getModels();
      const models = response.data.models || [];
      setAvailableModels(models);
      
      // Set default model to first available trained model, or first model if no trained models
      if (models.length > 0 && !config.model) {
        const trainedModels = models.filter((m: Model) => m.type === 'trained');
        const defaultModel = trainedModels.length > 0 ? trainedModels[0] : models[0];
        setConfig(prev => ({ ...prev, model: defaultModel.path }));
      }
    } catch (error: any) {
      console.error('Failed to load models:', error);
      setModelsError(error.response?.data?.error || 'Failed to load models');
    } finally {
      setLoadingModels(false);
    }
  };

  const handleModelUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pt')) {
      alert('Please select a valid .pt model file');
      return;
    }

    try {
      setUploadingModel(true);
      const response = await uploadModel(file);
      
      // Add the new model to the list
      const newModel = response.data.model;
      setAvailableModels(prev => [newModel, ...prev]);
      
      // Select the newly uploaded model
      setConfig(prev => ({ ...prev, model: newModel.path }));
      
      setShowModelUpload(false);
      alert('Model uploaded successfully!');
    } catch (error: any) {
      console.error('Failed to upload model:', error);
      alert(error.response?.data?.error || 'Failed to upload model');
    } finally {
      setUploadingModel(false);
      // Reset the file input
      event.target.value = '';
    }
  };

  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setUploadedImages(prev => [...prev, ...files]);
    }
  }, []);

  const removeImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index));
  };

  const startDetection = async () => {
    if (uploadedImages.length === 0) {
      alert('Please upload images first');
      return;
    }

    if (!config.model) {
      alert('Please select a model first');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);
    const newResults: DetectionResult[] = [];

    try {
      for (let i = 0; i < uploadedImages.length; i++) {
        const startTime = Date.now();
        
        // Use tile inference for large images, single detection for smaller ones
        const isLargeImage = uploadedImages[i].size > 5 * 1024 * 1024; // 5MB threshold
        
        const detectionConfig = {
          weights_path: config.model,
          confidence: config.confidence,
          iou: config.iou,
          tile_size: config.tileSize,
          overlap: config.overlap,
          merge_iou: config.mergeIou
        };

        let response;
        try {
          if (isLargeImage) {
            console.log('Using tile inference for large image:', uploadedImages[i].name);
            response = await detectTile(uploadedImages[i], detectionConfig);
          } else {
            console.log('Using single detection for image:', uploadedImages[i].name);
            response = await detectSingle(uploadedImages[i], detectionConfig);
          }

          const processingTime = (Date.now() - startTime) / 1000;
          setProcessingProgress(((i + 1) / uploadedImages.length) * 100);

          if (response.data.success) {
            const result: DetectionResult = {
              id: Math.random().toString(36).substr(2, 9),
              imageName: uploadedImages[i].name,
              count: response.data.count,
              confidence: config.confidence,
              processingTime: processingTime,
              timestamp: new Date().toISOString(),
              imageUrl: URL.createObjectURL(uploadedImages[i]),
              resultUrl: response.data.result_image || response.data.image_path
            };

            newResults.push(result);
            setResults(prev => [result, ...prev]);
          } else {
            console.error('Detection failed for', uploadedImages[i].name, response.data);
            alert(`Detection failed for ${uploadedImages[i].name}: ${response.data.error || 'Unknown error'}`);
          }
        } catch (imageError: any) {
          console.error('Error processing image', uploadedImages[i].name, imageError);
          alert(`Error processing ${uploadedImages[i].name}: ${imageError.response?.data?.error || imageError.message}`);
        }
      }
      
      if (newResults.length > 0) {
        alert(`Successfully processed ${newResults.length} out of ${uploadedImages.length} images`);
      }
    } catch (error: any) {
      console.error('Detection error:', error);
      alert('Detection process failed: ' + (error.response?.data?.error || error.message));
    } finally {
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  };

  const downloadResult = async (result: DetectionResult) => {
    try {
      if (result.resultUrl) {
        // Try to download the result image from the server
        try {
          const filename = result.resultUrl.split('/').pop() || result.imageName;
          const response = await getResultImage(filename);
          
          // Create blob URL and trigger download
          const blob = new Blob([response.data]);
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `result_${result.imageName}`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        } catch (imageError) {
          console.error('Failed to download result image:', imageError);
          // Fallback: download the original image
          downloadOriginalImage(result);
        }
      } else {
        // Download original image as fallback
        downloadOriginalImage(result);
      }
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download result');
    }
  };

  const downloadOriginalImage = (result: DetectionResult) => {
    if (result.imageUrl) {
      const a = document.createElement('a');
      a.href = result.imageUrl;
      a.download = result.imageName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const exportResults = () => {
    const csvContent = [
      'Image Name,Count,Confidence,Processing Time,Timestamp',
      ...results.map(r => `${r.imageName},${r.count},${r.confidence.toFixed(3)},${r.processingTime.toFixed(2)}s,${r.timestamp}`)
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'detection_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const getTotalCount = () => {
    return results.reduce((sum, result) => sum + result.count, 0);
  };

  const getAverageConfidence = () => {
    if (results.length === 0) return 0;
    return results.reduce((sum, result) => sum + result.confidence, 0) / results.length;
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Crop Detection</h1>
        <p className="mt-2 text-gray-600">Detect and count crops in your images using trained YOLOv8 models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Detection Configuration */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium text-gray-900">Detection Settings</h3>
              <button
                onClick={() => setShowModelUpload(!showModelUpload)}
                className="btn-secondary flex items-center text-sm"
                title="Upload custom model"
              >
                <PlusIcon className="h-4 w-4 mr-1" />
                Model
              </button>
            </div>
            
            {/* Model Upload Section */}
            {showModelUpload && (
              <div className="mb-4 p-4 border-2 border-dashed border-gray-300 rounded-lg">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Upload Custom Model</h4>
                <input
                  type="file"
                  accept=".pt"
                  onChange={handleModelUpload}
                  disabled={uploadingModel}
                  className="input-field mb-2"
                />
                {uploadingModel && (
                  <p className="text-sm text-blue-600">Uploading model...</p>
                )}
                <p className="text-xs text-gray-500">
                  Upload a trained YOLOv8/YOLOv11 .pt model file
                </p>
              </div>
            )}
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model {loadingModels && <span className="text-sm text-gray-500">(Loading...)</span>}
                </label>
                
                {modelsError ? (
                  <div className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg mb-2">
                    <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
                    <div>
                      <p className="text-sm text-red-700">{modelsError}</p>
                      <button 
                        onClick={loadModels}
                        className="text-sm text-red-600 hover:text-red-800 underline mt-1"
                      >
                        Retry
                      </button>
                    </div>
                  </div>
                ) : (
                  <select
                    value={config.model}
                    onChange={(e) => setConfig(prev => ({ ...prev, model: e.target.value }))}
                    className="input-field"
                    disabled={loadingModels}
                  >
                    <option value="">Select a model...</option>
                    {availableModels.map(model => (
                      <option key={model.id} value={model.path}>
                        {model.name} - {model.type === 'trained' ? model.accuracy : `${model.type} model`}
                      </option>
                    ))}
                  </select>
                )}
                
                {availableModels.length > 0 && (
                  <div className="mt-2 text-xs text-gray-500">
                    Available: {availableModels.filter(m => m.type === 'trained').length} trained, 
                    {availableModels.filter(m => m.type === 'custom').length} custom, 
                    {availableModels.filter(m => m.type === 'pretrained').length} pretrained
                  </div>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence Threshold: {config.confidence}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={config.confidence}
                  onChange={(e) => setConfig(prev => ({ ...prev, confidence: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  IoU Threshold: {config.iou}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={config.iou}
                  onChange={(e) => setConfig(prev => ({ ...prev, iou: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Tile Size
                  </label>
                  <input
                    type="number"
                    value={config.tileSize}
                    onChange={(e) => setConfig(prev => ({ ...prev, tileSize: parseInt(e.target.value) }))}
                    className="input-field"
                    min="512"
                    max="2048"
                    step="32"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Overlap
                  </label>
                  <input
                    type="number"
                    value={config.overlap}
                    onChange={(e) => setConfig(prev => ({ ...prev, overlap: parseInt(e.target.value) }))}
                    className="input-field"
                    min="50"
                    max="500"
                    step="10"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Merge IoU: {config.mergeIou}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={config.mergeIou}
                  onChange={(e) => setConfig(prev => ({ ...prev, mergeIou: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Image Upload and Processing */}
        <div className="lg:col-span-2 space-y-6">
          {/* Image Upload */}
          <div className="card">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Upload Images</h3>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <CloudUploadIcon className="mx-auto h-12 w-12 text-gray-400" />
              <div className="mt-4">
                <label htmlFor="image-upload" className="cursor-pointer">
                  <span className="mt-2 block text-sm font-medium text-gray-900">
                    Click to upload images or drag and drop
                  </span>
                  <span className="mt-1 block text-sm text-gray-500">
                    PNG, JPG, JPEG up to 10MB each
                  </span>
                </label>
                <input
                  id="image-upload"
                  name="image-upload"
                  type="file"
                  className="sr-only"
                  multiple
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </div>
            </div>

            {/* Uploaded Images */}
            {uploadedImages.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Uploaded Images ({uploadedImages.length})</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {uploadedImages.map((image, index) => (
                    <div key={index} className="relative group">
                      <img
                        src={URL.createObjectURL(image)}
                        alt={image.name}
                        className="w-full h-24 object-cover rounded-lg"
                      />
                      <button
                        onClick={() => removeImage(index)}
                        className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        ×
                      </button>
                      <p className="text-xs text-gray-500 mt-1 truncate">{image.name}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Processing Controls */}
            <div className="mt-6 flex justify-between items-center">
              <div className="flex space-x-2">
                <button
                  onClick={startDetection}
                  disabled={uploadedImages.length === 0 || isProcessing || !config.model || loadingModels}
                  className="btn-primary flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <PlayIcon className="h-4 w-4 mr-2" />
                  {isProcessing ? 'Processing...' : 'Start Detection'}
                </button>
                {results.length > 0 && (
                  <button
                    onClick={exportResults}
                    className="btn-secondary flex items-center"
                  >
                    <DownloadIcon className="h-4 w-4 mr-2" />
                    Export Results
                  </button>
                )}
              </div>
              
              {/* Status messages */}
              <div className="text-sm text-gray-500">
                {loadingModels && 'Loading models...'}
                {!config.model && !loadingModels && availableModels.length > 0 && 'Select a model'}
                {uploadedImages.length === 0 && 'Upload images to start'}
                {availableModels.length === 0 && !loadingModels && !modelsError && 'No models available'}
              </div>
            </div>

            {/* Processing Progress */}
            {isProcessing && (
              <div className="mt-4">
                <div className="flex justify-between text-sm text-gray-700 mb-1">
                  <span>Processing Images...</span>
                  <span>{Math.round(processingProgress)}%</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${processingProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Results Summary */}
          {results.length > 0 && (
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Detection Summary</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary-600">{results.length}</div>
                  <div className="text-sm text-gray-500">Images Processed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{getTotalCount()}</div>
                  <div className="text-sm text-gray-500">Total Crops Detected</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">{(getAverageConfidence() * 100).toFixed(1)}%</div>
                  <div className="text-sm text-gray-500">Avg Confidence</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Detection Results */}
      {results.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Detection Results</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Image
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Count
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {results.map((result) => (
                  <tr key={result.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {result.resultUrl ? (
                          <img 
                            src={`http://localhost:5000${result.resultUrl}`}
                            alt={`Result for ${result.imageName}`}
                            className="h-12 w-12 object-cover rounded border cursor-pointer hover:scale-105 transition-transform"
                            onClick={() => setSelectedResult(result)}
                            onError={(e) => {
                              // Fallback to icon if image fails to load
                              const target = e.target as HTMLImageElement;
                              target.style.display = 'none';
                              target.nextElementSibling?.classList.remove('hidden');
                            }}
                          />
                        ) : null}
                        <PhotoIcon className={`h-8 w-8 text-gray-400 ${result.resultUrl ? 'hidden' : ''}`} />
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{result.imageName}</div>
                      <div className="text-sm text-gray-500">{result.timestamp}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <ChartBarIcon className="h-4 w-4 text-green-500 mr-2" />
                        <span className="text-sm font-medium text-gray-900">{result.count}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-gray-900">{(result.confidence * 100).toFixed(1)}%</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-sm text-gray-500">
                        <ClockIcon className="h-4 w-4 mr-1" />
                        {result.processingTime.toFixed(2)}s
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setSelectedResult(result)}
                          className="text-primary-600 hover:text-primary-900"
                        >
                          <EyeIcon className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => downloadResult(result)}
                          className="text-green-600 hover:text-green-900"
                        >
                          <DownloadIcon className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Result Detail Modal */}
      {selectedResult && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-10 mx-auto p-6 border w-11/12 md:w-4/5 lg:w-3/4 xl:w-2/3 shadow-lg rounded-md bg-white max-h-[90vh] overflow-y-auto">
            <div className="mt-3">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-900">Detection Result</h3>
                <button
                  onClick={() => setSelectedResult(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-900">Image: {selectedResult.imageName}</h4>
                  <p className="text-sm text-gray-500">Processed on {selectedResult.timestamp}</p>
                </div>
                
                {/* Result Image Display */}
                <div className="space-y-3">
                  <h5 className="font-medium text-gray-700">Detection Result with Bounding Boxes:</h5>
                  <div className="flex justify-center">
                    {selectedResult.resultUrl ? (
                      <img 
                        src={`http://localhost:5000${selectedResult.resultUrl}`}
                        alt={`Detection result for ${selectedResult.imageName}`}
                        className="max-w-full max-h-96 object-contain rounded-lg shadow-md border"
                        onError={(e) => {
                          console.error('Failed to load result image:', selectedResult.resultUrl);
                          const target = e.target as HTMLImageElement;
                          target.src = selectedResult.imageUrl || '';
                          target.alt = 'Original image (result image failed to load)';
                        }}
                      />
                    ) : selectedResult.imageUrl ? (
                      <div className="text-center">
                        <img 
                          src={selectedResult.imageUrl}
                          alt={selectedResult.imageName}
                          className="max-w-full max-h-96 object-contain rounded-lg shadow-md border"
                        />
                        <p className="text-sm text-gray-500 mt-2">Original image (no result image available)</p>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-48 bg-gray-100 rounded-lg">
                        <PhotoIcon className="h-16 w-16 text-gray-400" />
                        <span className="ml-2 text-gray-500">No image available</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm font-medium text-gray-700">Crops Detected:</span>
                    <p className="text-lg font-semibold text-gray-900">{selectedResult.count}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-700">Confidence:</span>
                    <p className="text-lg font-semibold text-gray-900">{(selectedResult.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={() => setSelectedResult(null)}
                    className="btn-secondary"
                  >
                    Close
                  </button>
                  <button
                    onClick={() => downloadResult(selectedResult)}
                    className="btn-primary"
                  >
                    Download Result
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Detection;
