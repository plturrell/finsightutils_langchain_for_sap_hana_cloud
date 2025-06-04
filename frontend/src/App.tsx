import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Dashboard from './pages/Dashboard';
import Search from './pages/Search';
import Benchmark from './pages/Benchmark';
import Settings from './pages/Settings';
import Developer from './pages/Developer';
import Layout from './components/Layout';
import { ErrorProvider } from './context/ErrorContext';

const App: React.FC = () => {
  return (
    <ErrorProvider>
      <Router>
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="search" element={<Search />} />
              <Route path="benchmark" element={<Benchmark />} />
              <Route path="settings" element={<Settings />} />
              <Route path="developer" element={<Developer />} />
            </Route>
          </Routes>
        </Box>
      </Router>
    </ErrorProvider>
  );
};

export default App;