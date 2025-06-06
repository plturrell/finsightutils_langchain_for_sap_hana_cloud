import React, { useState, useEffect, useRef } from 'react';

/**
 * React component wrapper for the EmbeddingVisualizer
 * 
 * This component provides a React interface to the EmbeddingVisualizer class,
 * allowing it to be easily integrated into React applications.
 */
const EmbeddingVisualizerReact = ({
  points = [],
  metadata = null,
  similarities = null,
  queryPoint = null,
  clusters = null,
  width = '100%',
  height = '500px',
  method = '3d',
  colorScheme = 'similarity',
  showLabels = true,
  showAxes = true,
  showGrid = true,
  colorblindMode = false,
  autoRotate = false,
  onPointSelected = null,
  className = '',
  style = {}
}) => {
  const containerRef = useRef(null);
  const visualizerRef = useRef(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [selectedPoints, setSelectedPoints] = useState([]);

  // Initialize visualizer when component mounts
  useEffect(() => {
    // Check if EmbeddingVisualizer is available
    if (typeof window === 'undefined' || !window.EmbeddingVisualizer) {
      // Load EmbeddingVisualizer script
      const script = document.createElement('script');
      script.src = '/components/visualization/EmbeddingVisualizer.js';
      script.async = true;
      script.onload = initializeVisualizer;
      document.body.appendChild(script);
      
      return () => {
        document.body.removeChild(script);
      };
    } else {
      initializeVisualizer();
    }
  }, []);

  // Initialize the visualizer
  const initializeVisualizer = () => {
    if (!containerRef.current || !window.EmbeddingVisualizer) return;
    
    // Create visualizer instance
    const containerId = containerRef.current.id;
    const options = {
      width,
      height,
      method,
      colorScheme,
      showLabels,
      showAxes,
      showGrid,
      colorblindFriendly: colorblindMode,
      autoRotate,
    };
    
    visualizerRef.current = new window.EmbeddingVisualizer(containerId, options);
    setIsInitialized(true);
    
    // Start a polling interval to check for selected points
    const checkSelectionInterval = setInterval(() => {
      if (visualizerRef.current) {
        const newSelected = visualizerRef.current.getSelectedPoints();
        if (JSON.stringify(newSelected) !== JSON.stringify(selectedPoints)) {
          setSelectedPoints(newSelected);
          if (onPointSelected) {
            onPointSelected(newSelected);
          }
        }
      } else {
        clearInterval(checkSelectionInterval);
      }
    }, 500);
    
    return () => {
      clearInterval(checkSelectionInterval);
    };
  };

  // Load data when points or visualizer change
  useEffect(() => {
    if (isInitialized && visualizerRef.current && points && points.length > 0) {
      visualizerRef.current.loadData(points, metadata, similarities, queryPoint, clusters);
    }
  }, [isInitialized, points, metadata, similarities, queryPoint, clusters]);

  // Update options when they change
  useEffect(() => {
    if (isInitialized && visualizerRef.current) {
      visualizerRef.current.updateOptions({
        method,
        colorScheme,
        showLabels,
        showAxes,
        showGrid,
        colorblindFriendly: colorblindMode,
        autoRotate,
      });
    }
  }, [isInitialized, method, colorScheme, showLabels, showAxes, showGrid, colorblindMode, autoRotate]);

  // Create a unique ID for the container if none exists
  const containerId = useRef(`embedding-viz-${Math.random().toString(36).substring(2, 10)}`);

  return (
    <div
      ref={containerRef}
      id={containerId.current}
      className={`embedding-visualizer-container ${className}`}
      style={{
        width,
        height,
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        ...style
      }}
    >
      {!isInitialized && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0,0,0,0.03)',
            fontSize: '14px',
            color: '#666',
          }}
        >
          Initializing visualization...
        </div>
      )}
    </div>
  );
};

export default EmbeddingVisualizerReact;