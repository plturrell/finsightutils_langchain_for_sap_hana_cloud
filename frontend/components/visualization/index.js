/**
 * Vector Embedding Visualization Components
 * 
 * This module exports visualization components for displaying vector embeddings
 * in 2D and 3D formats with support for TensorRT optimization information.
 */

// Export the JavaScript class for direct use
export { default as EmbeddingVisualizer } from './EmbeddingVisualizer';

// Export the React component for React applications
export { default as EmbeddingVisualizerReact } from './EmbeddingVisualizerReact';

// Export additional visualization utilities
export { default as TensorRTInfoPanel } from './TensorRTInfoPanel';
export { default as VisualizationControls } from './VisualizationControls';
export { default as VectorMetricsPanel } from './VectorMetricsPanel';