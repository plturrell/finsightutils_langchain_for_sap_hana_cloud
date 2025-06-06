/**
 * Example: Interactive vector visualization for SAP HANA Cloud LangChain integration
 * 
 * This example demonstrates:
 * 1. Creating an interactive 3D visualization of vector embeddings
 * 2. Streaming search results with real-time updates
 * 3. Implementing accessibility features
 * 4. Responsive design for mobile and desktop
 * 
 * Requirements:
 * - React
 * - Three.js and react-three-fiber
 * - D3.js for dimensionality reduction
 * - Material UI for components
 */

import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import * as THREE from 'three';
import * as d3 from 'd3';
import { 
  Box, 
  Typography, 
  TextField, 
  Button, 
  CircularProgress, 
  Paper, 
  Slider, 
  FormControlLabel,
  Switch,
  useMediaQuery,
  useTheme,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  PanTool as PanToolIcon,
  AccessibilityNew as AccessibilityIcon,
} from '@mui/icons-material';
import { vectorStoreService } from '../api/services';

// UMAP parameters for dimensionality reduction
const UMAP_PARAMS = {
  nComponents: 3,          // 3D visualization
  nNeighbors: 15,          // Local neighborhood size
  minDist: 0.1,            // Minimum distance between points
  spread: 1.0,             // Global spread of the embeddings
  random: 42,              // Random seed for reproducibility
};

// Accessibility color schemes
const COLOR_SCHEMES = {
  default: {
    background: '#f5f5f5',
    queryPoint: '#ff3030',
    documentPoints: '#2196f3',
    selectedPoint: '#4caf50',
    textColor: '#000000',
    gridColor: '#cccccc',
  },
  highContrast: {
    background: '#000000',
    queryPoint: '#ff0000',
    documentPoints: '#ffffff',
    selectedPoint: '#00ff00',
    textColor: '#ffffff',
    gridColor: '#444444',
  },
  colorblindFriendly: {
    background: '#f5f5f5',
    queryPoint: '#e69f00',
    documentPoints: '#56b4e9',
    selectedPoint: '#009e73',
    textColor: '#000000',
    gridColor: '#cccccc',
  },
};

/**
 * Point cloud component for vector visualization
 */
const VectorPointCloud = ({
  points,
  queryPoint,
  selectedPointIndex,
  setSelectedPointIndex,
  similarities,
  colorScheme,
}) => {
  const pointsRef = useRef();
  const queryPointRef = useRef();
  
  // Update positions when points change
  useEffect(() => {
    if (pointsRef.current && points.length > 0) {
      // Update point positions
      const positions = new Float32Array(points.length * 3);
      const colors = new Float32Array(points.length * 3);
      const sizes = new Float32Array(points.length);
      
      points.forEach((point, i) => {
        positions[i * 3] = point[0];
        positions[i * 3 + 1] = point[1];
        positions[i * 3 + 2] = point[2];
        
        // Color based on similarity if available
        const similarity = similarities && similarities[i] ? similarities[i] : 0.5;
        
        // Convert similarity to color
        // Higher similarity = more intense color
        const color = new THREE.Color(colorScheme.documentPoints);
        color.lerp(new THREE.Color(colorScheme.selectedPoint), similarity);
        
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
        
        // Size based on similarity
        sizes[i] = 0.2 + (similarity * 0.3);
      });
      
      pointsRef.current.geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(positions, 3)
      );
      
      pointsRef.current.geometry.setAttribute(
        'color',
        new THREE.BufferAttribute(colors, 3)
      );
      
      pointsRef.current.geometry.setAttribute(
        'size',
        new THREE.BufferAttribute(sizes, 1)
      );
      
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
      pointsRef.current.geometry.attributes.color.needsUpdate = true;
      pointsRef.current.geometry.attributes.size.needsUpdate = true;
    }
    
    // Update query point if available
    if (queryPointRef.current && queryPoint) {
      queryPointRef.current.position.set(
        queryPoint[0],
        queryPoint[1],
        queryPoint[2]
      );
    }
  }, [points, queryPoint, similarities, selectedPointIndex, colorScheme]);
  
  // Handle point selection
  const handlePointClick = (event) => {
    // Find closest point to the click
    if (pointsRef.current && points.length > 0) {
      event.stopPropagation();
      
      // Get mouse position in normalized device coordinates
      const mouse = new THREE.Vector2(
        (event.offsetX / event.target.clientWidth) * 2 - 1,
        -(event.offsetY / event.target.clientHeight) * 2 + 1
      );
      
      // Raycasting to find the closest point
      const raycaster = new THREE.Raycaster();
      raycaster.params.Points.threshold = 0.2;
      raycaster.setFromCamera(mouse, event.camera);
      
      const intersects = raycaster.intersectObject(pointsRef.current);
      
      if (intersects.length > 0) {
        const index = intersects[0].index;
        setSelectedPointIndex(index);
      }
    }
  };
  
  return (
    <>
      {/* Document points */}
      <points ref={pointsRef} onClick={handlePointClick}>
        <bufferGeometry />
        <pointsMaterial
          vertexColors
          size={0.5}
          sizeAttenuation
          transparent
          alphaTest={0.5}
        />
      </points>
      
      {/* Query point (larger and highlighted) */}
      {queryPoint && (
        <mesh ref={queryPointRef} position={[queryPoint[0], queryPoint[1], queryPoint[2]]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial color={colorScheme.queryPoint} />
        </mesh>
      )}
      
      {/* Selected point highlight */}
      {selectedPointIndex !== null && points[selectedPointIndex] && (
        <mesh position={[
          points[selectedPointIndex][0],
          points[selectedPointIndex][1],
          points[selectedPointIndex][2]
        ]}>
          <sphereGeometry args={[0.25, 16, 16]} />
          <meshBasicMaterial color={colorScheme.selectedPoint} wireframe />
        </mesh>
      )}
      
      {/* Grid for better spatial perception */}
      <gridHelper args={[20, 20, colorScheme.gridColor, colorScheme.gridColor]} />
      
      {/* Axes labels for orientation */}
      <Text position={[10, 0, 0]} color={colorScheme.textColor} fontSize={0.5}>
        X
      </Text>
      <Text position={[0, 10, 0]} color={colorScheme.textColor} fontSize={0.5}>
        Y
      </Text>
      <Text position={[0, 0, 10]} color={colorScheme.textColor} fontSize={0.5}>
        Z
      </Text>
    </>
  );
};

/**
 * Main visualization component
 */
const VectorVisualization = ({ height = '70vh' }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  // State for search and visualization
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [points, setPoints] = useState([]);
  const [queryPoint, setQueryPoint] = useState(null);
  const [selectedPointIndex, setSelectedPointIndex] = useState(null);
  const [resultCount, setResultCount] = useState(10);
  const [diversityFactor, setDiversityFactor] = useState(0.5);
  const [useTensorRT, setUseTensorRT] = useState(true);
  
  // Accessibility options
  const [showLabels, setShowLabels] = useState(true);
  const [colorScheme, setColorScheme] = useState(COLOR_SCHEMES.default);
  const [useHighContrast, setUseHighContrast] = useState(false);
  const [isColorblindMode, setIsColorblindMode] = useState(false);
  
  // Update color scheme when accessibility options change
  useEffect(() => {
    if (useHighContrast) {
      setColorScheme(COLOR_SCHEMES.highContrast);
    } else if (isColorblindMode) {
      setColorScheme(COLOR_SCHEMES.colorblindFriendly);
    } else {
      setColorScheme(COLOR_SCHEMES.default);
    }
  }, [useHighContrast, isColorblindMode]);
  
  // Function to reduce dimensions using UMAP-like algorithm
  // (In a production environment, use actual UMAP from a library or API)
  const reduceDimensions = (embeddings, queryEmbedding = null) => {
    // For this example, we'll use d3's force simulation as a simple alternative
    // In production, use a proper UMAP implementation
    
    // Combine all embeddings
    const allEmbeddings = queryEmbedding 
      ? [queryEmbedding, ...embeddings] 
      : [...embeddings];
    
    // Create nodes for the force simulation
    const nodes = allEmbeddings.map((_, i) => ({ id: i }));
    
    // Create links based on cosine similarity
    const links = [];
    for (let i = 0; i < allEmbeddings.length; i++) {
      for (let j = i + 1; j < allEmbeddings.length; j++) {
        // Calculate cosine similarity
        const embA = allEmbeddings[i];
        const embB = allEmbeddings[j];
        
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let k = 0; k < embA.length; k++) {
          dotProduct += embA[k] * embB[k];
          normA += embA[k] * embA[k];
          normB += embB[k] * embB[k];
        }
        
        const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        
        links.push({
          source: i,
          target: j,
          value: similarity * 10  // Scale for force strength
        });
      }
    }
    
    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).strength(d => d.value))
      .force('charge', d3.forceManyBody().strength(-50))
      .force('center', d3.forceCenter())
      .force('x', d3.forceX())
      .force('y', d3.forceY())
      .force('z', d3.forceZ())
      .stop();
    
    // Run the simulation
    for (let i = 0; i < 300; i++) simulation.tick();
    
    // Extract 3D positions
    const positions = nodes.map(node => [node.x, node.y, node.z || 0]);
    
    // Return positions and separate query position if provided
    if (queryEmbedding) {
      return {
        queryPosition: positions[0],
        documentPositions: positions.slice(1)
      };
    }
    
    return {
      documentPositions: positions
    };
  };
  
  // Perform search and visualize results
  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    setResults([]);
    setPoints([]);
    setQueryPoint(null);
    setSelectedPointIndex(null);
    
    try {
      // Determine which search method to use
      const searchMethod = diversityFactor < 1 ? 'mmr' : 'similarity';
      let response;
      
      if (searchMethod === 'mmr') {
        // Use MMR search for diverse results
        response = await vectorStoreService.mmrQuery(
          query,
          resultCount,
          resultCount * 3,  // fetch_k = 3x result count
          diversityFactor,
          null,  // filter
          useTensorRT
        );
      } else {
        // Use standard similarity search
        response = await vectorStoreService.query(
          query,
          resultCount,
          null,  // filter
          useTensorRT
        );
      }
      
      // Extract results and embeddings
      const searchResults = response.data.results;
      const queryEmbedding = response.data.query_embedding;
      
      // Extract document embeddings (for a real implementation, you would need to
      // request the embeddings from the backend as well)
      const documentEmbeddings = searchResults.map(result => 
        result.embedding || new Array(queryEmbedding.length).fill(0).map(() => Math.random())
      );
      
      // Extract similarities
      const similarities = searchResults.map(result => result.score);
      
      // Reduce dimensions for visualization
      const { queryPosition, documentPositions } = reduceDimensions(
        documentEmbeddings,
        queryEmbedding
      );
      
      // Update state with results and visualization data
      setResults(searchResults.map(result => result.document));
      setPoints(documentPositions);
      setQueryPoint(queryPosition);
      
      // For streaming visualization effect, animate points appearing
      const pointsPerStep = 3;
      const steps = Math.ceil(documentPositions.length / pointsPerStep);
      
      for (let i = 0; i < steps; i++) {
        const visiblePoints = documentPositions.slice(0, (i + 1) * pointsPerStep);
        setPoints([...visiblePoints]);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
    } catch (err) {
      console.error('Search error:', err);
      setError('Failed to perform search. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle keyboard navigation
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      // Navigate through results
      if (results.length === 0) return;
      
      const newIndex = selectedPointIndex === null
        ? 0
        : (selectedPointIndex + (e.key === 'ArrowUp' ? -1 : 1) + results.length) % results.length;
      
      setSelectedPointIndex(newIndex);
      e.preventDefault();
    }
  };
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Search controls */}
      <Paper
        elevation={2}
        sx={{
          p: 2,
          mb: 2,
          borderRadius: 2,
          backgroundColor: colorScheme.background,
          color: colorScheme.textColor,
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 2 }}>
          <TextField
            fullWidth
            label="Search Query"
            variant="outlined"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            InputProps={{
              startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />,
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
              },
              '& .MuiInputLabel-root': {
                color: colorScheme.textColor,
              },
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: colorScheme.textColor,
              },
            }}
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
            sx={{
              minWidth: 120,
              borderRadius: 2,
              alignSelf: isMobile ? 'stretch' : 'auto',
            }}
          >
            Search
          </Button>
        </Box>
        
        {/* Advanced options */}
        <Box sx={{ mt: 2 }}>
          <Typography gutterBottom>
            Results: {resultCount}
          </Typography>
          <Slider
            value={resultCount}
            onChange={(_, value) => setResultCount(value)}
            min={5}
            max={50}
            step={5}
            marks={[
              { value: 5, label: '5' },
              { value: 25, label: '25' },
              { value: 50, label: '50' },
            ]}
            disabled={loading}
            sx={{
              color: colorScheme.documentPoints,
              '& .MuiSlider-markLabel': {
                color: colorScheme.textColor,
              },
            }}
          />
          
          <Typography gutterBottom sx={{ mt: 1 }}>
            Diversity Factor: {diversityFactor.toFixed(1)}
          </Typography>
          <Slider
            value={diversityFactor}
            onChange={(_, value) => setDiversityFactor(value)}
            min={0}
            max={1}
            step={0.1}
            marks={[
              { value: 0, label: 'Diverse' },
              { value: 0.5, label: 'Balanced' },
              { value: 1, label: 'Relevant' },
            ]}
            disabled={loading}
            sx={{
              color: colorScheme.documentPoints,
              '& .MuiSlider-markLabel': {
                color: colorScheme.textColor,
              },
            }}
          />
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', mt: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={useTensorRT}
                  onChange={(e) => setUseTensorRT(e.target.checked)}
                  color="primary"
                />
              }
              label="Use GPU Acceleration"
            />
            
            {/* Accessibility controls */}
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Tooltip title="Toggle Labels">
                <IconButton onClick={() => setShowLabels(!showLabels)}>
                  {showLabels ? <VisibilityIcon /> : <VisibilityOffIcon />}
                </IconButton>
              </Tooltip>
              
              <Tooltip title="High Contrast Mode">
                <IconButton onClick={() => setUseHighContrast(!useHighContrast)}>
                  <AccessibilityIcon 
                    color={useHighContrast ? 'primary' : 'action'} 
                  />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Colorblind Friendly Mode">
                <IconButton onClick={() => setIsColorblindMode(!isColorblindMode)}>
                  <VisibilityIcon 
                    color={isColorblindMode ? 'primary' : 'action'} 
                  />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Box>
      </Paper>
      
      {/* Visualization area */}
      <Box
        sx={{
          position: 'relative',
          flex: 1,
          height: height,
          borderRadius: 2,
          overflow: 'hidden',
          backgroundColor: colorScheme.background,
        }}
      >
        <Canvas camera={{ position: [0, 0, 15], fov: 50 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          
          <VectorPointCloud
            points={points}
            queryPoint={queryPoint}
            selectedPointIndex={selectedPointIndex}
            setSelectedPointIndex={setSelectedPointIndex}
            similarities={results.map(result => result.score)}
            colorScheme={colorScheme}
          />
          
          {/* Camera controls */}
          <OrbitControls makeDefault />
          
          {/* Point labels if enabled */}
          {showLabels && results.length > 0 && points.length > 0 && (
            <group>
              {results.map((result, idx) => (
                points[idx] && (
                  <Html
                    key={idx}
                    position={[points[idx][0], points[idx][1], points[idx][2]]}
                    distanceFactor={10}
                    occlude
                    zIndexRange={[100, 0]}
                    style={{
                      display: selectedPointIndex === idx || selectedPointIndex === null ? 'block' : 'none',
                      padding: '6px',
                      background: 'rgba(0,0,0,0.8)',
                      color: 'white',
                      borderRadius: '4px',
                      fontSize: '12px',
                      maxWidth: '200px',
                      pointerEvents: 'none',
                    }}
                  >
                    {result.metadata?.title || `Document ${idx + 1}`}
                  </Html>
                )
              ))}
            </group>
          )}
        </Canvas>
        
        {/* Visualization controls */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            right: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
          }}
        >
          <Tooltip title="Zoom In">
            <IconButton
              sx={{ bgcolor: 'rgba(255,255,255,0.7)' }}
              onClick={() => {
                // Access camera from OrbitControls and zoom in
                const controls = document.querySelector('canvas')?.__r3f?.controls;
                if (controls) {
                  controls.dollyIn(1.2);
                  controls.update();
                }
              }}
            >
              <ZoomInIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Zoom Out">
            <IconButton
              sx={{ bgcolor: 'rgba(255,255,255,0.7)' }}
              onClick={() => {
                // Access camera from OrbitControls and zoom out
                const controls = document.querySelector('canvas')?.__r3f?.controls;
                if (controls) {
                  controls.dollyOut(1.2);
                  controls.update();
                }
              }}
            >
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Reset View">
            <IconButton
              sx={{ bgcolor: 'rgba(255,255,255,0.7)' }}
              onClick={() => {
                // Reset camera position
                const controls = document.querySelector('canvas')?.__r3f?.controls;
                if (controls) {
                  controls.reset();
                }
              }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      {/* Results panel */}
      {results.length > 0 && (
        <Paper
          elevation={2}
          sx={{
            mt: 2,
            p: 2,
            borderRadius: 2,
            maxHeight: '300px',
            overflow: 'auto',
            backgroundColor: colorScheme.background,
            color: colorScheme.textColor,
          }}
        >
          <Typography variant="h6" gutterBottom>
            Search Results
          </Typography>
          
          {results.map((doc, idx) => (
            <Box
              key={idx}
              sx={{
                p: 1,
                my: 1,
                borderRadius: 1,
                border: '1px solid',
                borderColor: selectedPointIndex === idx 
                  ? colorScheme.selectedPoint
                  : 'rgba(0,0,0,0.1)',
                backgroundColor: selectedPointIndex === idx 
                  ? `${colorScheme.selectedPoint}22`
                  : 'transparent',
                cursor: 'pointer',
                '&:hover': {
                  backgroundColor: `${colorScheme.documentPoints}22`,
                },
              }}
              onClick={() => setSelectedPointIndex(idx)}
              tabIndex={0}
              role="button"
              aria-pressed={selectedPointIndex === idx}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  setSelectedPointIndex(idx);
                  e.preventDefault();
                }
              }}
            >
              <Typography variant="subtitle2">
                {doc.metadata?.title || `Document ${idx + 1}`}
                {doc.score && (
                  <Typography 
                    component="span" 
                    variant="caption" 
                    sx={{ ml: 1, color: colorScheme.documentPoints }}
                  >
                    ({(doc.score * 100).toFixed(1)}% match)
                  </Typography>
                )}
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>
                {doc.page_content?.substring(0, 150)}
                {doc.page_content?.length > 150 ? '...' : ''}
              </Typography>
            </Box>
          ))}
        </Paper>
      )}
    </Box>
  );
};

export default VectorVisualization;