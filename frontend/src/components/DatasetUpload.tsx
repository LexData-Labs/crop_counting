import React, { useState, useCallback, useEffect } from 'react';
import { getDatasets } from '../services/api';
import { CloudArrowUpIcon as CloudUploadIcon, DocumentIcon, PhotoIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'uploading' | 'completed' | 'error';
  progress: number;
}

const DatasetUpload = () => {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0); // 0-100
  const [isUploading, setIsUploading] = useState(false);

  // Load list of stored datasets
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

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

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
            // Progress for this chunk only
            const chunkProgress = Math.round((event.loaded / event.total) * 100);
            // Overall progress
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
    // Refresh the datasets list after upload completes
    try { await fetchDatasets(); } catch {}
  };

  const simulateUpload = (fileId: string) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 30;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        setUploadedFiles(prev => 
          prev.map(file => 
            file.id === fileId 
              ? { ...file, status: 'completed', progress: 100 }
              : file
          )
        );
      } else {
        setUploadedFiles(prev => 
          prev.map(file => 
            file.id === fileId 
              ? { ...file, progress }
              : file
          )
        );
      }
    }, 200);
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
  };

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
        return 'text-blue-600';
      case 'completed':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
  <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dataset Upload</h1>
        <p className="mt-2 text-gray-600">Upload your crop images and corresponding YOLO label files</p>
      </div>

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

      {/* Actions */}
      <div className="flex justify-end space-x-4">
        <button className="btn-secondary">
          Cancel
        </button>
        <button 
          className="btn-primary"
          disabled={uploadedFiles.length === 0 || uploadedFiles.some(f => f.status === 'uploading')}
        >
          Process Dataset
        </button>
      </div>
    </div>
  );
};

export default DatasetUpload;
