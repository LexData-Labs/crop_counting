import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { 
  HomeIcon, 
  CogIcon, 
  EyeIcon, 
  ChartBarIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';
import DatasetTraining from './components/DatasetTraining';
import Detection from './components/Detection';
import Dashboard from './components/Dashboard';
import Models from './components/Models';

const Navigation = () => {
  const location = useLocation();
  
  const navItems = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Dataset & Training', href: '/dataset-training', icon: CogIcon },
    { name: 'Models', href: '/models', icon: BeakerIcon },
    { name: 'Detection', href: '/detection', icon: EyeIcon },
    { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  ];

  return (
    <nav className="bg-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <h1 className="text-xl font-bold text-primary-600">Crop Counter AI</h1>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              {navItems.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`${
                      isActive
                        ? 'border-primary-500 text-primary-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                  >
                    <item.icon className="w-4 h-4 mr-2" />
                    {item.name}
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dataset-training" element={<DatasetTraining />} />
            <Route path="/models" element={<Models />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/analytics" element={<div className="text-center py-12"><h2 className="text-2xl font-bold text-gray-900">Analytics Coming Soon</h2></div>} />
            {/* Legacy route redirects */}
            <Route path="/upload" element={<Navigate to="/dataset-training" replace />} />
            <Route path="/training" element={<Navigate to="/dataset-training" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;