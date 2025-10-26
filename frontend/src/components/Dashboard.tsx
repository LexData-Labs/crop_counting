import React, { useState, useEffect } from 'react';
import { 
  CloudArrowUpIcon as CloudUploadIcon, 
  CogIcon, 
  EyeIcon, 
  ChartBarIcon,
  BeakerIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalDatasets: 0,
    trainedModels: 0,
    totalDetections: 0,
    accuracy: 0
  });

  const [recentActivity, setRecentActivity] = useState([
    { id: 1, type: 'upload', message: 'Dataset uploaded successfully', time: '2 hours ago', status: 'success' },
    { id: 2, type: 'training', message: 'Model training completed', time: '1 day ago', status: 'success' },
    { id: 3, type: 'detection', message: 'Detection analysis finished', time: '2 days ago', status: 'success' },
  ]);

  const quickActions = [
    {
      title: 'Upload Dataset',
      description: 'Upload images and labels for training',
      icon: CloudUploadIcon,
      href: '/upload',
      color: 'bg-blue-500 hover:bg-blue-600'
    },
    {
      title: 'Train Model',
      description: 'Start training a new YOLOv8 model',
      icon: CogIcon,
      href: '/training',
      color: 'bg-green-500 hover:bg-green-600'
    },
    {
      title: 'Run Detection',
      description: 'Detect crops in uploaded images',
      icon: EyeIcon,
      href: '/detection',
      color: 'bg-purple-500 hover:bg-purple-600'
    },
    {
      title: 'View Models',
      description: 'Manage and select trained models',
      icon: BeakerIcon,
      href: '/models',
      color: 'bg-orange-500 hover:bg-orange-600'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Crop Counter AI Dashboard</h1>
        <p className="mt-2 text-gray-600">Train and deploy YOLOv8 models for automated crop counting</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CloudUploadIcon className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Total Datasets</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.totalDatasets}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <BeakerIcon className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Trained Models</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.trainedModels}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <EyeIcon className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Total Detections</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.totalDetections}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-orange-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Avg Accuracy</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.accuracy}%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {quickActions.map((action) => (
            <a
              key={action.title}
              href={action.href}
              className="card hover:shadow-lg transition-shadow duration-200 cursor-pointer"
            >
              <div className="flex items-center mb-4">
                <div className={`p-3 rounded-lg ${action.color} text-white`}>
                  <action.icon className="h-6 w-6" />
                </div>
                <h3 className="ml-3 text-lg font-medium text-gray-900">{action.title}</h3>
              </div>
              <p className="text-gray-600">{action.description}</p>
            </a>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Activity</h2>
        <div className="card">
          <div className="flow-root">
            <ul className="-mb-8">
              {recentActivity.map((activity, activityIdx) => (
                <li key={activity.id}>
                  <div className="relative pb-8">
                    {activityIdx !== recentActivity.length - 1 ? (
                      <span
                        className="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200"
                        aria-hidden="true"
                      />
                    ) : null}
                    <div className="relative flex space-x-3">
                      <div>
                        <span
                          className={`h-8 w-8 rounded-full flex items-center justify-center ring-8 ring-white ${
                            activity.status === 'success'
                              ? 'bg-green-500'
                              : activity.status === 'warning'
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                          }`}
                        >
                          {activity.status === 'success' ? (
                            <CheckCircleIcon className="h-5 w-5 text-white" />
                          ) : (
                            <ExclamationTriangleIcon className="h-5 w-5 text-white" />
                          )}
                        </span>
                      </div>
                      <div className="min-w-0 flex-1 pt-1.5 flex justify-between space-x-4">
                        <div>
                          <p className="text-sm text-gray-500">{activity.message}</p>
                        </div>
                        <div className="text-right text-sm whitespace-nowrap text-gray-500">
                          {activity.time}
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
