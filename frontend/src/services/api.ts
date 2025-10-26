import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Datasets
export const getDatasets = () => api.get('/datasets');
// (removed duplicate getModels export)

// Health check
export const healthCheck = () => api.get('/health');


// Training
export const startTraining = (config: any) => api.post('/training/start', config);
export const getTrainingStatus = (trainingId: string) => api.get(`/training/status/${trainingId}`);

// Models
export const getModels = () => api.get('/models');
export const uploadModel = (modelFile: File) => {
  const formData = new FormData();
  formData.append('model', modelFile);
  return api.post('/models/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Detection
export const detectSingle = (image: File, config: any) => {
  const formData = new FormData();
  formData.append('image', image);
  Object.keys(config).forEach(key => {
    formData.append(key, config[key]);
  });
  return api.post('/detection/single', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const detectTile = (image: File, config: any) => {
  const formData = new FormData();
  formData.append('image', image);
  Object.keys(config).forEach(key => {
    formData.append(key, config[key]);
  });
  return api.post('/detection/tile', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

// Results
export const getResultImage = (filename: string) => 
  api.get(`/results/${filename}`, { responseType: 'blob' });

export default api;
