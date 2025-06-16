import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Dashboard from './pages/Dashboard';
import Search from './pages/Search';
import Benchmark from './pages/Benchmark';
import Settings from './pages/Settings';
import Developer from './pages/Developer';
import Reasoning from './pages/Reasoning';
import DataPipeline from './pages/DataPipeline';
import VectorCreation from './pages/VectorCreation';
import VectorExploration from './pages/VectorExploration';
import Layout from './components/Layout';
import PageTransition from './components/PageTransition';
import { ErrorProvider } from './context/ErrorContext';
import { AnimationProvider } from '@finsightdev/ui-animations';
// Keep local AnimationToggle for now, we'll replace it in the next step
import AnimationToggle from './components/AnimationToggle';
import { EnhancedComponentProvider } from './utils/applyEnhancements';
import { EmptyStateReplacer, defaultEmptyStateDetector } from './utils/emptyStateReplacer';

const App: React.FC = () => {
  return (
    <ErrorProvider>
      <AnimationProvider storagePrefix="langchain-hana">
        <EnhancedComponentProvider>
          <EmptyStateReplacer isEmptyState={defaultEmptyStateDetector}>
            <Router>
              <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <AnimationToggle positionTop={false} />
                <Routes>
                  <Route path="/" element={<Layout />}>
                    <Route index element={
                      <PageTransition>
                        <Dashboard />
                      </PageTransition>
                    } />
                    <Route path="search" element={
                      <PageTransition>
                        <Search />
                      </PageTransition>
                    } />
                    <Route path="benchmark" element={
                      <PageTransition>
                        <Benchmark />
                      </PageTransition>
                    } />
                    <Route path="settings" element={
                      <PageTransition>
                        <Settings />
                      </PageTransition>
                    } />
                    <Route path="developer" element={
                      <PageTransition>
                        <Developer />
                      </PageTransition>
                    } />
                    <Route path="reasoning" element={
                      <PageTransition>
                        <Reasoning />
                      </PageTransition>
                    } />
                    <Route path="data-pipeline" element={
                      <PageTransition>
                        <DataPipeline />
                      </PageTransition>
                    } />
                    <Route path="vector-creation" element={
                      <PageTransition>
                        <VectorCreation />
                      </PageTransition>
                    } />
                    <Route path="vector-exploration" element={
                      <PageTransition>
                        <VectorExploration />
                      </PageTransition>
                    } />
                    <Route path="vector-exploration/:vectorTable" element={
                      <PageTransition>
                        <VectorExploration />
                      </PageTransition>
                    } />
                  </Route>
                </Routes>
              </Box>
            </Router>
          </EmptyStateReplacer>
        </EnhancedComponentProvider>
      </AnimationProvider>
    </ErrorProvider>
  );
};

export default App;