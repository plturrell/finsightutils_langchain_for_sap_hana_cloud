import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Slider,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  FormControlLabel,
  CircularProgress,
  Alert,
  Tooltip,
  Collapse,
  IconButton,
  InputAdornment,
  Stack,
  Button,
  Chip,
  alpha,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
  Close as CloseIcon,
  Info as InfoIcon,
  TouchApp as TouchAppIcon,
  Search as SearchIcon,
  AutoAwesome as AutoAwesomeIcon,
  TipsAndUpdates as TipsAndUpdatesIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import { developerService, GetVectorsRequest } from '../api/services';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import HumanText from './HumanText';
import { humanize } from '../utils/humanLanguage';

// Types
interface VisualizationProps {
  tableName?: string;
  filter?: Record<string, any>;
  maxPoints?: number;
  initialPage?: number;
  initialPageSize?: number;
  initialClusteringAlgorithm?: 'kmeans' | 'dbscan' | 'hdbscan';
  initialDimensionalityReduction?: 'tsne' | 'umap' | 'pca';
}

interface EmbeddingPoint {
  id: string;
  vector: number[];
  metadata: Record<string, any>;
  content: string;
  color?: string;
  size?: number;
}

interface ReducedPoint {
  id: string;
  x: number;
  y: number;
  z: number;
  metadata: Record<string, any>;
  content: string;
  color: string;
  size: number;
}

// Color map function
const getColorForCategory = (category: string) => {
  const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  
  // Simple hash function for consistent colors
  let hash = 0;
  for (let i = 0; i < category.length; i++) {
    hash = category.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  return colors[Math.abs(hash) % colors.length];
};

const VectorVisualization: React.FC<VisualizationProps> = ({ 
  tableName = 'EMBEDDINGS',
  filter = {},
  maxPoints = 500,
  initialPage = 1,
  initialPageSize = 100,
  initialClusteringAlgorithm = 'kmeans',
  initialDimensionalityReduction = 'tsne'
}) => {
  // State
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [points, setPoints] = useState<ReducedPoint[]>([]);
  const [visualizationMode, setVisualizationMode] = useState<'2d' | '3d'>('3d');
  const [colorBy, setColorBy] = useState<string>('cluster');
  const [sizeBy, setSizeBy] = useState<string>('none');
  const [perplexity, setPerplexity] = useState<number>(30);
  const [selectedPoint, setSelectedPoint] = useState<ReducedPoint | null>(null);
  const [metadataFields, setMetadataFields] = useState<string[]>([]);
  const [clusterCount, setClusterCount] = useState<number>(5);
  const [useAutoRotate, setUseAutoRotate] = useState<boolean>(true);
  
  // Pagination state
  const [page, setPage] = useState<number>(initialPage);
  const [pageSize, setPageSize] = useState<number>(initialPageSize);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [totalPages, setTotalPages] = useState<number>(1);
  
  // Advanced visualization settings
  const [clusteringAlgorithm, setClusteringAlgorithm] = useState<'kmeans' | 'dbscan' | 'hdbscan'>(initialClusteringAlgorithm);
  const [dimensionalityReduction, setDimensionalityReduction] = useState<'tsne' | 'umap' | 'pca'>(initialDimensionalityReduction);
  
  // Real-time filtering
  const [filterOptions, setFilterOptions] = useState<Record<string, any>>(filter);
  const [filterText, setFilterText] = useState<string>('');
  const [availableFilters, setAvailableFilters] = useState<Record<string, any[]>>({});
  
  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  
  // Load data when parameters change
  useEffect(() => {
    fetchVectorData();
  }, [tableName, filterOptions, maxPoints, page, pageSize, clusteringAlgorithm, dimensionalityReduction]);
  
  // Initialize 3D visualization when points change
  useEffect(() => {
    if (visualizationMode === '3d' && points.length > 0) {
      initThreeJS();
      return () => {
        if (rendererRef.current && containerRef.current) {
          containerRef.current.removeChild(rendererRef.current.domElement);
        }
        disposeThreeJS();
      };
    }
  }, [points, visualizationMode]);
  
  // Update 3D visualization when options change
  useEffect(() => {
    if (visualizationMode === '3d' && points.length > 0 && sceneRef.current) {
      updatePointColors();
      updatePointSizes();
    }
  }, [colorBy, sizeBy, visualizationMode, points]);
  
  // Set auto-rotation
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = useAutoRotate;
    }
  }, [useAutoRotate]);
  
  // Extract metadata fields from the first point
  useEffect(() => {
    if (points.length > 0) {
      const fields = Object.keys(points[0].metadata).filter(
        key => typeof points[0].metadata[key] !== 'object'
      );
      setMetadataFields(fields);
    }
  }, [points]);
  
  // Fetch vector data from the API
  const fetchVectorData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Call the API to get vector data
      const request: GetVectorsRequest = {
        tableName,
        filter: filterOptions,
        maxPoints,
        page,
        pageSize,
        clusteringAlgorithm,
        dimensionalityReduction
      };
      
      const response = await developerService.getVectors(request);
      const data = response.data;
      
      // Update pagination information
      setTotalCount(data.total_count);
      setTotalPages(data.total_pages);
      
      // Transform the data
      const reducedPoints: ReducedPoint[] = data.vectors.map((point: any, index: number) => {
        // Determine color based on metadata or cluster
        let pointColor = '';
        if (colorBy === 'cluster' && point.metadata.cluster !== undefined) {
          // Use cluster-based coloring
          const clusterColors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
          ];
          pointColor = clusterColors[point.metadata.cluster % clusterColors.length];
        } else if (colorBy !== 'none' && colorBy in point.metadata) {
          // Use metadata-based coloring
          pointColor = getColorForCategory(String(point.metadata[colorBy]));
        } else {
          // Default color
          pointColor = '#1f77b4';
        }
        
        // Determine size based on metadata
        let pointSize = 5;
        if (sizeBy !== 'none' && sizeBy in point.metadata && typeof point.metadata[sizeBy] === 'number') {
          // Scale size between 3 and 10
          const value = point.metadata[sizeBy];
          pointSize = 3 + (value * 7);
        }
        
        return {
          id: point.id || `point-${index}`,
          x: point.reduced_vector[0],
          y: point.reduced_vector[1],
          z: point.reduced_vector[2] || 0,
          metadata: point.metadata || {},
          content: point.content || '',
          color: pointColor,
          size: pointSize,
        };
      });
      
      setPoints(reducedPoints);
      
      // Extract available metadata fields and values
      if (data.vectors.length > 0) {
        // Get unique metadata fields
        const fields = Object.keys(data.vectors[0].metadata).filter(
          key => typeof data.vectors[0].metadata[key] !== 'object'
        );
        setMetadataFields(fields);
        
        // Extract available filter values
        const filters: Record<string, any[]> = {};
        fields.forEach(field => {
          const values = new Set<any>();
          data.vectors.forEach(point => {
            if (point.metadata[field] !== undefined) {
              values.add(point.metadata[field]);
            }
          });
          filters[field] = Array.from(values);
        });
        setAvailableFilters(filters);
      }
    } catch (err) {
      console.error('Error fetching vector data:', err);
      setError('Failed to fetch vector data. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle page change
  const handlePageChange = (event: unknown, newPage: number) => {
    setPage(newPage);
  };
  
  // Handle page size change
  const handlePageSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPageSize(parseInt(event.target.value, 10));
    setPage(1); // Reset to first page when changing page size
  };
  
  // Handle filter change
  const applyFilter = () => {
    // Parse the filter text to create a filter object
    if (filterText.trim()) {
      try {
        // Check if it's valid JSON
        const filterObject = JSON.parse(filterText);
        setFilterOptions(prevFilters => ({
          ...prevFilters,
          ...filterObject
        }));
      } catch (e) {
        // If not valid JSON, try to parse as key:value
        const parts = filterText.split(':').map(part => part.trim());
        if (parts.length === 2) {
          const [key, value] = parts;
          setFilterOptions(prevFilters => ({
            ...prevFilters,
            [key]: value
          }));
        } else {
          // Use as content search
          setFilterOptions(prevFilters => ({
            ...prevFilters,
            content: filterText
          }));
        }
      }
    } else {
      // Clear filters if filter text is empty
      setFilterOptions({});
    }
    
    // Reset to first page when applying new filters
    setPage(1);
  };
  
  // Reset filters
  const resetFilters = () => {
    setFilterOptions({});
    setFilterText('');
    setPage(1);
  };
  
  // Initialize Three.js scene
  const initThreeJS = () => {
    if (!containerRef.current) return;
    
    // Clear previous renderer
    if (rendererRef.current && containerRef.current.contains(rendererRef.current.domElement)) {
      containerRef.current.removeChild(rendererRef.current.domElement);
    }
    
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f5);
    sceneRef.current = scene;
    
    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 5;
    cameraRef.current = camera;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = useAutoRotate;
    controls.autoRotateSpeed = 0.5;
    controlsRef.current = controls;
    
    // Add points
    addPointsToScene();
    
    // Add grid helper
    const gridHelper = new THREE.GridHelper(10, 10);
    scene.add(gridHelper);
    
    // Add axis helper
    const axisHelper = new THREE.AxesHelper(5);
    scene.add(axisHelper);
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
      
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  };
  
  // Add points to the 3D scene
  const addPointsToScene = () => {
    if (!sceneRef.current) return;
    
    // Remove existing points
    if (pointsRef.current) {
      sceneRef.current.remove(pointsRef.current);
    }
    
    // Create geometry
    const geometry = new THREE.BufferGeometry();
    
    // Create positions array
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const sizes = new Float32Array(points.length);
    
    points.forEach((point, i) => {
      positions[i * 3] = point.x;
      positions[i * 3 + 1] = point.y;
      positions[i * 3 + 2] = point.z;
      
      // Convert hex color to RGB
      const color = new THREE.Color(point.color);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
      
      sizes[i] = point.size;
    });
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    // Material with custom shaders for varying point sizes
    const material = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
    });
    
    // Create points
    const pointsMesh = new THREE.Points(geometry, material);
    sceneRef.current.add(pointsMesh);
    pointsRef.current = pointsMesh;
    
    // Add raycaster for point selection
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.1;
    
    // Add click event listener
    const handleClick = (event: MouseEvent) => {
      if (!containerRef.current || !cameraRef.current || !pointsRef.current) return;
      
      const rect = containerRef.current.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / containerRef.current.clientWidth) * 2 - 1;
      const y = -((event.clientY - rect.top) / containerRef.current.clientHeight) * 2 + 1;
      
      const mouse = new THREE.Vector2(x, y);
      raycaster.setFromCamera(mouse, cameraRef.current);
      
      const intersects = raycaster.intersectObject(pointsRef.current);
      
      if (intersects.length > 0) {
        const index = intersects[0].index;
        if (index !== undefined) {
          setSelectedPoint(points[index]);
        }
      } else {
        setSelectedPoint(null);
      }
    };
    
    containerRef.current?.addEventListener('click', handleClick);
    
    return () => {
      containerRef.current?.removeEventListener('click', handleClick);
    };
  };
  
  // Update point colors based on selected attribute
  const updatePointColors = () => {
    if (!pointsRef.current || !sceneRef.current) return;
    
    const colors = new Float32Array(points.length * 3);
    
    points.forEach((point, i) => {
      let color;
      
      if (colorBy === 'cluster') {
        color = new THREE.Color(point.color);
      } else if (colorBy === 'none') {
        color = new THREE.Color(0x1f77b4);
      } else if (colorBy in point.metadata) {
        const value = point.metadata[colorBy];
        if (typeof value === 'string') {
          color = new THREE.Color(getColorForCategory(value));
        } else if (typeof value === 'number') {
          // Normalize between 0 and 1
          const normalized = Math.max(0, Math.min(1, value));
          color = new THREE.Color().setHSL(normalized * 0.6, 0.8, 0.5);
        } else {
          color = new THREE.Color(0x1f77b4);
        }
      } else {
        color = new THREE.Color(0x1f77b4);
      }
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });
    
    const geometry = pointsRef.current.geometry;
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.attributes.color.needsUpdate = true;
  };
  
  // Update point sizes based on selected attribute
  const updatePointSizes = () => {
    if (!pointsRef.current || !sceneRef.current) return;
    
    const sizes = new Float32Array(points.length);
    
    points.forEach((point, i) => {
      let size = 5; // Default size
      
      if (sizeBy === 'none') {
        size = 5;
      } else if (sizeBy in point.metadata) {
        const value = point.metadata[sizeBy];
        if (typeof value === 'number') {
          // Scale between 3 and 15
          size = 3 + value * 12;
        }
      }
      
      sizes[i] = size;
    });
    
    const geometry = pointsRef.current.geometry;
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    geometry.attributes.size.needsUpdate = true;
  };
  
  // Dispose Three.js objects
  const disposeThreeJS = () => {
    if (pointsRef.current) {
      pointsRef.current.geometry.dispose();
      if (pointsRef.current.material instanceof THREE.Material) {
        pointsRef.current.material.dispose();
      } else if (Array.isArray(pointsRef.current.material)) {
        pointsRef.current.material.forEach(material => material.dispose());
      }
    }
    
    if (rendererRef.current) {
      rendererRef.current.dispose();
    }
  };
  
  // Render 2D plot
  const render2DPlot = () => {
    if (points.length === 0) return null;
    
    // Group points by category for 2D plot
    const categories: Record<string, { x: number[], y: number[], text: string[], name: string }> = {};
    
    points.forEach(point => {
      const category = colorBy === 'cluster' 
        ? `Cluster ${Math.floor(Math.random() * clusterCount)}`
        : (colorBy in point.metadata 
          ? String(point.metadata[colorBy]) 
          : 'Unknown');
      
      if (!categories[category]) {
        categories[category] = {
          x: [],
          y: [],
          text: [],
          name: category
        };
      }
      
      categories[category].x.push(point.x);
      categories[category].y.push(point.y);
      categories[category].text.push(point.content.substring(0, 50) + (point.content.length > 50 ? '...' : ''));
    });
    
    const data = Object.values(categories).map(category => ({
      x: category.x,
      y: category.y,
      text: category.text,
      name: category.name,
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 8,
        opacity: 0.7
      }
    }));
    
    return (
      <Plot
        data={data}
        layout={{
          title: 'Vector Space Visualization (t-SNE)',
          autosize: true,
          showlegend: true,
          legend: {
            x: 1,
            xanchor: 'right',
            y: 1
          },
          hovermode: 'closest',
          margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
        onClick={(data) => {
          if (data.points && data.points.length > 0) {
            const point = data.points[0];
            const index = point.pointIndex;
            if (typeof index === 'number') {
              const category = Object.values(categories)[point.curveNumber];
              setSelectedPoint({
                id: `plot-${index}`,
                x: category.x[index],
                y: category.y[index],
                z: 0,
                content: category.text[index],
                metadata: {},
                color: '',
                size: 5
              });
            }
          }
        }}
      />
    );
  };
  
  // State for the settings panel visibility
  const [showSettings, setShowSettings] = useState<boolean>(false);
  // State for the details panel visibility
  const [showDetails, setShowDetails] = useState<boolean>(false);
  // State for the Easter egg "one more thing" feature
  const [hasDiscoveredSecret, setHasDiscoveredSecret] = useState<boolean>(false);
  // Track user interaction counts for triggering "magical" moments
  const [interactionCount, setInteractionCount] = useState<number>(0);
  
  // Track mouse movement to detect special patterns
  const handleMouseMove = (event: React.MouseEvent) => {
    // Secret interaction - drawing an "S" shape will trigger the special feature
    // This is simplified - in a real implementation, you'd use a pattern recognition algorithm
    if (interactionCount > 20 && !hasDiscoveredSecret && event.shiftKey) {
      const chance = Math.random();
      if (chance < 0.1) {
        setHasDiscoveredSecret(true);
        // Play a subtle animation/sound here
      }
    }
  };
  
  // Increment interaction counter on user actions
  const trackInteraction = () => {
    setInteractionCount(prev => prev + 1);
  };
  
  const renderContextualTooltip = (message: string) => (
    <Box 
      className="contextual-tooltip animate-fade-in" 
      sx={{
        position: 'absolute',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(0,0,0,0.7)',
        color: '#fff',
        padding: '8px 16px',
        borderRadius: '20px',
        fontSize: '0.75rem',
        zIndex: 10,
        boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
        backdropFilter: 'blur(4px)',
        opacity: 0.9,
        maxWidth: '80%',
        pointerEvents: 'none',
      }}
    >
      {message}
    </Box>
  );

  return (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'hidden',
        borderRadius: { xs: 2, md: 3 },
        boxShadow: 3,
        background: hasDiscoveredSecret 
          ? 'linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(245,250,255,1) 100%)' 
          : '#fff',
        position: 'relative',
        transition: 'all 0.3s ease-in-out',
      }}
      onMouseMove={handleMouseMove}
    >
      {/* Settings Panel - Only shown when activated */}
      <Collapse in={showSettings}>
        <Box 
          sx={{ 
            p: { xs: 2, sm: 3 },
            borderBottom: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'rgba(252, 253, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.03)',
          }}
        >
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <HumanText variant="subtitle1" fontWeight={600} color="primary.dark">
                Visualization Settings
              </HumanText>
              <IconButton 
                size="small" 
                onClick={() => setShowSettings(false)}
                sx={{ 
                  opacity: 0.7,
                  '&:hover': {
                    opacity: 1,
                    backgroundColor: alpha('#0066B3', 0.05),
                  }
                }}
              >
                <ExpandLessIcon fontSize="small" />
              </IconButton>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl size="small" fullWidth>
                <InputLabel>Color By</InputLabel>
                <Select
                  value={colorBy}
                  label="Color By"
                  onChange={(e) => {
                    setColorBy(e.target.value);
                    trackInteraction();
                  }}
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#000', 0.1),
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#0066B3', 0.3),
                    },
                  }}
                >
                  <MenuItem value="none">None</MenuItem>
                  <MenuItem value="cluster">Cluster</MenuItem>
                  {metadataFields.map(field => (
                    <MenuItem key={field} value={field}>{field}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl size="small" fullWidth>
                <InputLabel>Size By</InputLabel>
                <Select
                  value={sizeBy}
                  label="Size By"
                  onChange={(e) => {
                    setSizeBy(e.target.value);
                    trackInteraction();
                  }}
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#000', 0.1),
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#0066B3', 0.3),
                    },
                  }}
                >
                  <MenuItem value="none">Uniform</MenuItem>
                  {metadataFields
                    .filter(field => typeof points[0]?.metadata[field] === 'number')
                    .map(field => (
                      <MenuItem key={field} value={field}>{field}</MenuItem>
                    ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl size="small" fullWidth>
                <InputLabel>Algorithm</InputLabel>
                <Select
                  value={clusteringAlgorithm}
                  label="Algorithm"
                  onChange={(e) => {
                    setClusteringAlgorithm(e.target.value as 'kmeans' | 'dbscan' | 'hdbscan');
                    trackInteraction();
                  }}
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#000', 0.1),
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#0066B3', 0.3),
                    },
                  }}
                >
                  <MenuItem value="kmeans">K-Means</MenuItem>
                  <MenuItem value="dbscan">DBSCAN</MenuItem>
                  <MenuItem value="hdbscan">HDBSCAN</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl size="small" fullWidth>
                <InputLabel>Technique</InputLabel>
                <Select
                  value={dimensionalityReduction}
                  label="Technique"
                  onChange={(e) => {
                    setDimensionalityReduction(e.target.value as 'tsne' | 'umap' | 'pca');
                    trackInteraction();
                  }}
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#000', 0.1),
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#0066B3', 0.3),
                    },
                  }}
                >
                  <MenuItem value="tsne">t-SNE</MenuItem>
                  <MenuItem value="umap">UMAP</MenuItem>
                  <MenuItem value="pca">PCA</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            {/* Conditional perplexity slider for t-SNE */}
            {dimensionalityReduction === 'tsne' && (
              <Grid item xs={12} sm={6} md={4} lg={3}>
                <HumanText variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  Perplexity: {perplexity}
                </HumanText>
                <Slider
                  size="small"
                  value={perplexity}
                  onChange={(e, value) => {
                    setPerplexity(value as number);
                    trackInteraction();
                  }}
                  aria-labelledby="perplexity-slider"
                  valueLabelDisplay="auto"
                  min={5}
                  max={50}
                  marks={[{ value: 5 }, { value: 30 }, { value: 50 }]}
                  sx={{ 
                    mt: 1,
                    '& .MuiSlider-thumb': {
                      width: 14,
                      height: 14,
                    },
                    '& .MuiSlider-rail': {
                      opacity: 0.3,
                    }
                  }}
                />
              </Grid>
            )}
            
            {/* Auto-rotate for 3D */}
            {visualizationMode === '3d' && (
              <Grid item xs={12} sm={6} md={4} lg={3}>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={useAutoRotate}
                      onChange={(e) => {
                        setUseAutoRotate(e.target.checked);
                        trackInteraction();
                      }}
                      sx={{
                        '& .MuiSwitch-thumb': {
                          boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                        },
                      }}
                    />
                  }
                  label={<HumanText variant="caption">Auto-Rotate</HumanText>}
                />
              </Grid>
            )}
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl size="small" fullWidth>
                <InputLabel>Results</InputLabel>
                <Select
                  value={pageSize}
                  label="Results"
                  onChange={(e) => {
                    setPageSize(Number(e.target.value));
                    setPage(1);
                    trackInteraction();
                  }}
                  sx={{
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#000', 0.1),
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: alpha('#0066B3', 0.3),
                    },
                  }}
                >
                  <MenuItem value={50}>50 points</MenuItem>
                  <MenuItem value={100}>100 points</MenuItem>
                  <MenuItem value={200}>200 points</MenuItem>
                  <MenuItem value={500}>500 points</MenuItem>
                  <MenuItem value={1000}>1000 points</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            {/* Filter section - improved spacing and consistency */}
            <Grid item xs={12}>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 2, 
                flexWrap: { xs: 'wrap', md: 'nowrap' },
                mt: 1,
              }}>
                <TextField
                  size="small"
                  label="Filter"
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                  placeholder="content:finance"
                  sx={{ 
                    flexGrow: 1, 
                    minWidth: { xs: '100%', md: 'auto' },
                    '& .MuiOutlinedInput-root': {
                      '& fieldset': {
                        borderColor: alpha('#000', 0.1),
                      },
                      '&:hover fieldset': {
                        borderColor: alpha('#0066B3', 0.3),
                      },
                    }
                  }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon fontSize="small" sx={{ opacity: 0.5 }} />
                      </InputAdornment>
                    ),
                  }}
                />
                <Button 
                  variant="contained" 
                  color="primary"
                  size="small"
                  onClick={() => {
                    applyFilter();
                    trackInteraction();
                  }}
                  sx={{ 
                    minWidth: { xs: '45%', md: 'auto' },
                    px: 3,
                    height: 40,
                    borderRadius: 2,
                    boxShadow: 'none',
                    background: 'linear-gradient(90deg, #0066B3 0%, #1976d2 100%)',
                    transition: 'all 0.2s',
                    '&:hover': {
                      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                      background: 'linear-gradient(90deg, #0058a3 0%, #1565c0 100%)',
                    }
                  }}
                >
                  <HumanText>Apply</HumanText>
                </Button>
                <Button 
                  variant="outlined"
                  size="small"
                  onClick={() => {
                    resetFilters();
                    trackInteraction();
                  }}
                  sx={{ 
                    minWidth: { xs: '45%', md: 'auto' },
                    px: 3,
                    height: 40,
                    borderRadius: 2,
                    borderColor: alpha('#000', 0.15),
                    color: 'text.primary',
                    '&:hover': {
                      borderColor: alpha('#0066B3', 0.5),
                      backgroundColor: alpha('#0066B3', 0.05),
                    }
                  }}
                >
                  <HumanText>Reset</HumanText>
                </Button>
              </Box>
              
              {/* Active filters as pills with improved styling */}
              {Object.keys(filterOptions).length > 0 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 2 }}>
                  {Object.entries(filterOptions).map(([key, value]) => (
                    <Chip
                      key={key}
                      label={`${key}: ${value}`}
                      size="small"
                      onDelete={() => {
                        const newFilters = { ...filterOptions };
                        delete newFilters[key];
                        setFilterOptions(newFilters);
                        trackInteraction();
                      }}
                      sx={{ 
                        height: 26, 
                        '& .MuiChip-label': { 
                          fontSize: '0.75rem',
                          px: 1,
                        },
                        '& .MuiChip-deleteIcon': {
                          fontSize: '0.875rem',
                          marginRight: '4px',
                        },
                        background: alpha('#0066B3', 0.08),
                        borderRadius: 6,
                      }}
                    />
                  ))}
                </Box>
              )}
            </Grid>
          </Grid>
        </Box>
      </Collapse>
      
      {/* Main Visualization Area */}
      <Box 
        sx={{ 
          flexGrow: 1, 
          display: 'flex', 
          overflow: 'hidden',
          position: 'relative',
          p: showSettings ? 0 : { xs: 1, sm: 2 }, // Extra padding when settings are hidden
        }}
      >
        {/* Main Content Area */}
        <Box 
          sx={{ 
            flex: 1, 
            display: 'flex', 
            flexDirection: 'column',
            position: 'relative',
            mr: { xs: 0, md: selectedPoint || showDetails ? 2 : 0 },
            transition: 'margin-right 0.3s ease-in-out',
          }}
        >
          {/* Minimal Top Control Bar */}
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              p: { xs: 1.5, sm: 2 },
              pb: { xs: 1, sm: 1.5 },
            }}
          >
            {/* Simplified Title */}
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <HumanText 
                variant="h6" 
                sx={{ 
                  fontWeight: 600,
                  fontSize: { xs: '1rem', sm: '1.125rem' },
                  color: hasDiscoveredSecret ? 'primary.main' : 'text.primary',
                  background: hasDiscoveredSecret 
                    ? 'linear-gradient(90deg, #0066B3 30%, #19B5FE 100%)'
                    : 'none',
                  WebkitBackgroundClip: hasDiscoveredSecret ? 'text' : 'unset',
                  WebkitTextFillColor: hasDiscoveredSecret ? 'transparent' : 'unset',
                  letterSpacing: '-0.01em',
                }}
              >
                {hasDiscoveredSecret ? 'Insight Explorer' : 'Vector Space'}
              </HumanText>
            </Box>
            
            {/* Essential Controls - Radically simplified */}
            <Box sx={{ display: 'flex', gap: 2 }}>
              {/* Dimension Toggle - more intentional */}
              <ToggleButtonGroup
                value={visualizationMode}
                exclusive
                size="small"
                onChange={(e, newMode) => {
                  if (newMode) {
                    setVisualizationMode(newMode);
                    trackInteraction();
                  }
                }}
                aria-label="visualization mode"
                sx={{ 
                  height: 32,
                  boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                  '& .MuiToggleButton-root': {
                    width: 44,
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: NEUTRAL[700],
                    borderColor: NEUTRAL[400],
                    transition: 'all 0.25s cubic-bezier(0.25, 1, 0.5, 1)',
                    '&.Mui-selected': {
                      backgroundColor: PRIMARY[500],
                      color: '#FFFFFF',
                      fontWeight: 600,
                      // Subtle spring animation on selection
                      animation: 'springReveal 0.4s cubic-bezier(0.25, 1, 0.5, 1)',
                    },
                    '&:hover:not(.Mui-selected)': {
                      backgroundColor: NEUTRAL[300],
                    },
                    '&:active': {
                      transform: 'scale(0.97)',
                    },
                  }
                }}
              >
                <ToggleButton value="2d">2D</ToggleButton>
                <ToggleButton value="3d">3D</ToggleButton>
              </ToggleButtonGroup>
              
              {/* Settings Toggle - more tactile */}
              <IconButton 
                size="small" 
                onClick={() => {
                  setShowSettings(!showSettings);
                  // Add haptic-like visual feedback
                  const btn = document.activeElement;
                  if (btn) {
                    btn.animate([
                      { transform: 'scale(0.95)' },
                      { transform: 'scale(1)' }
                    ], { 
                      duration: 150,
                      easing: 'cubic-bezier(0.25, 1, 0.5, 1)'
                    });
                  }
                }}
                sx={{ 
                  color: showSettings ? PRIMARY[500] : NEUTRAL[700],
                  bgcolor: showSettings ? PRIMARY[100] : NEUTRAL[200],
                  width: 32,
                  height: 32,
                  transition: 'all 0.2s cubic-bezier(0.25, 1, 0.5, 1)',
                  '&:hover': {
                    bgcolor: showSettings ? PRIMARY[200] : NEUTRAL[300],
                    transform: 'translateY(-1px)',
                  },
                  '&:active': {
                    transform: 'translateY(0) scale(0.97)',
                  },
                }}
              >
                <SettingsIcon sx={{ fontSize: '1.125rem' }} />
              </IconButton>
              
              {/* Only show magical lens when discovered */}
              {hasDiscoveredSecret && (
                <IconButton 
                  size="small"
                  className="animate-pulse"
                  onClick={() => {
                    trackInteraction();
                    // Add special animation feedback
                    const el = document.querySelector('.visualization-container');
                    if (el) {
                      el.animate([
                        { filter: 'brightness(1.1) contrast(1.05)', transform: 'scale(0.995)' },
                        { filter: 'brightness(1) contrast(1)', transform: 'scale(1)' }
                      ], { 
                        duration: 400,
                        easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)'
                      });
                    }
                  }}
                  sx={{ 
                    color: PRIMARY[500],
                    bgcolor: PRIMARY[100],
                    width: 32,
                    height: 32,
                    transition: 'all 0.2s cubic-bezier(0.25, 1, 0.5, 1)',
                    '&:hover': {
                      bgcolor: PRIMARY[200],
                      transform: 'translateY(-1px)',
                    },
                    '&:active': {
                      transform: 'translateY(0) scale(0.97)',
                    },
                  }}
                >
                  <TipsAndUpdatesIcon sx={{ fontSize: '1.125rem' }} />
                </IconButton>
              )}
            </Box>
          </Box>
          
          {/* Main Visualization with enhanced white space */}
          <Box 
            sx={{ 
              flexGrow: 1, 
              position: 'relative',
              p: { xs: 0.5, sm: 1, md: 2 },
              pb: { xs: 0, sm: 0, md: 1 },
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {loading ? (
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                height: '100%',
                flexDirection: 'column',
                gap: 2.5
              }}>
                <CircularProgress 
                  size={40} 
                  thickness={4} 
                  sx={{ 
                    color: '#0066B3',
                    opacity: 0.8,
                  }}
                />
                <HumanText variant="body2" color="text.secondary">
                  Analyzing data relationships...
                </HumanText>
              </Box>
            ) : error ? (
              <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
                <Alert 
                  severity="error" 
                  sx={{ 
                    maxWidth: '400px',
                    borderRadius: 2,
                    boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                  }}
                >
                  {error}
                </Alert>
              </Box>
            ) : (
              <Box
                ref={containerRef}
                onClick={() => trackInteraction()}
                className="visualization-container"
                sx={{
                  width: '100%',
                  height: '100%',
                  minHeight: 400,
                  flex: 1,
                }}
              >
                {visualizationMode === '2d' && render2DPlot()}
                
                {/* More subtle contextual tooltip for beginners */}
                {interactionCount < 5 && renderContextualTooltip("Click on a point to explore its details")}
                
                {/* The "one more thing" feature - simplified with more personality */}
                {hasDiscoveredSecret && (
                  <Box
                    className="insight-overlay animate-fade-in"
                    sx={{
                      position: 'absolute',
                      bottom: 24,
                      left: 24,
                      background: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: 3,
                      p: 3,
                      border: 'none',
                      maxWidth: 300,
                      zIndex: 5,
                      overflow: 'hidden',
                      '&:before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        backgroundImage: 'linear-gradient(135deg, rgba(0, 102, 179, 0.02) 0%, rgba(25, 118, 210, 0.03) 100%)',
                        zIndex: -1,
                      },
                    }}
                  >
                    <Box 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        mb: 2,
                      }}
                    >
                      <Box
                        className="animate-subtle-glow"
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 36,
                          height: 36,
                          borderRadius: '50%',
                          background: '#fff',
                          mr: 2,
                          boxShadow: '0 0 0 1px rgba(0, 102, 179, 0.15)',
                        }}
                      >
                        <AutoAwesomeIcon 
                          color="primary" 
                          className="animate-breathe"
                          sx={{ fontSize: '1.25rem' }}
                        />
                      </Box>
                      <HumanText 
                        variant="subtitle1" 
                        fontWeight={600} 
                        color="primary.main"
                        sx={{ letterSpacing: '-0.01em' }}
                      >
                        Insight Lens
                      </HumanText>
                    </Box>

                    <HumanText 
                      variant="body2" 
                      sx={{ 
                        color: '#111',
                        lineHeight: 1.6,
                        fontWeight: 400,
                        mb: 2,
                        letterSpacing: '0.01em',
                      }}
                    >
                      You've discovered a new way of seeing connections. Patterns hidden just beneath 
                      the surface are now visible to you.
                    </HumanText>

                    <Box 
                      sx={{ 
                        width: '100%',
                        height: 1,
                        background: 'linear-gradient(90deg, transparent, rgba(0, 102, 179, 0.2), transparent)',
                        mb: 2,
                      }}
                    />

                    <HumanText 
                      className="animate-gentle-wave"
                      variant="caption" 
                      sx={{ 
                        display: 'block',
                        textAlign: 'center',
                        color: 'primary.main',
                        fontStyle: 'italic',
                        opacity: 0.8,
                      }}
                    >
                      "The real voyage of discovery consists not in seeking new landscapes, but in having new eyes."
                    </HumanText>
                  </Box>
                )}
              </Box>
            )}
            
            {/* Minimalist Pagination with refined styling */}
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              py: 1.5,
              px: 2,
              opacity: 0.7,
              transition: 'opacity 0.2s',
              '&:hover': {
                opacity: 1
              }
            }}>
              <HumanText 
                variant="caption" 
                color="text.secondary" 
                sx={{ 
                  mr: 1.5, 
                  fontSize: '0.75rem',
                  letterSpacing: '0.02em',
                }}
              >
                {totalCount > 0 ? 
                  `${((page - 1) * pageSize) + 1}-${Math.min(page * pageSize, totalCount)} of ${totalCount}` : 
                  'No data points'}
              </HumanText>
              
              <Box sx={{ 
                display: 'flex', 
                gap: 0.75,
                borderRadius: 10,
                padding: '2px 4px',
                backgroundColor: alpha('#000', 0.02),
              }}>
                <IconButton
                  size="small"
                  disabled={page === 1}
                  onClick={() => setPage(page - 1)}
                  sx={{ 
                    p: 0.5,
                    color: page === 1 ? 'text.disabled' : 'text.secondary',
                    '&:hover': {
                      backgroundColor: alpha('#0066B3', 0.08),
                    }
                  }}
                >
                  <ChevronLeftIcon fontSize="small" />
                </IconButton>
                <HumanText 
                  variant="caption" 
                  sx={{ 
                    alignSelf: 'center', 
                    mx: 0.75, 
                    fontSize: '0.75rem',
                    fontVariantNumeric: 'tabular-nums',
                    fontWeight: 500,
                    minWidth: '36px',
                    textAlign: 'center',
                  }}
                >
                  {page}/{totalPages}
                </HumanText>
                <IconButton
                  size="small"
                  disabled={page === totalPages}
                  onClick={() => setPage(page + 1)}
                  sx={{ 
                    p: 0.5,
                    color: page === totalPages ? 'text.disabled' : 'text.secondary',
                    '&:hover': {
                      backgroundColor: alpha('#0066B3', 0.08),
                    }
                  }}
                >
                  <ChevronRightIcon fontSize="small" />
                </IconButton>
              </Box>
            </Box>
          </Box>
        </Box>
        
        {/* Details Side Panel - Only visible when needed - enhanced white space and transitions */}
        <Collapse 
          in={selectedPoint !== null || showDetails} 
          orientation="horizontal" 
          sx={{ 
            width: selectedPoint || showDetails ? { xs: '100%', sm: '100%', md: 320 } : 0,
            position: { xs: 'absolute', md: 'relative' },
            top: 0,
            right: 0,
            bottom: 0,
            zIndex: { xs: 10, md: 1 },
            height: '100%',
            display: { xs: selectedPoint || showDetails ? 'block' : 'none', md: 'block' },
            background: { xs: 'rgba(255, 255, 255, 0.97)', md: 'transparent' },
            backdropFilter: { xs: 'blur(16px)', md: 'none' },
            boxShadow: { xs: '-8px 0 32px rgba(0, 0, 0, 0.08)', md: 'none' },
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        >
          <Box 
            sx={{ 
              height: '100%', 
              overflow: 'auto',
              borderLeft: { md: '1px solid rgba(0,0,0,0.06)' },
              p: { xs: 2.5, md: 3 },
              display: 'flex',
              flexDirection: 'column',
              scrollbarWidth: 'thin',
              scrollbarColor: 'rgba(0,0,0,0.1) transparent',
              '&::-webkit-scrollbar': {
                width: '4px',
              },
              '&::-webkit-scrollbar-track': {
                background: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                background: 'rgba(0,0,0,0.1)',
                borderRadius: '4px',
              }
            }}
          >
            {/* Mobile Close Button with refined styling */}
            <Box 
              sx={{ 
                display: { xs: 'flex', md: 'none' }, 
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 2,
              }}
            >
              <HumanText 
                variant="subtitle2" 
                sx={{ 
                  fontWeight: 600,
                  color: 'primary.main',
                }}
              >
                Details
              </HumanText>
              <IconButton 
                size="small" 
                onClick={() => setShowDetails(false)}
                sx={{ 
                  color: 'text.secondary',
                  width: 32,
                  height: 32,
                  '&:hover': {
                    backgroundColor: alpha('#0066B3', 0.05),
                  }
                }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>
            
            {selectedPoint ? (
              <Box 
                className="animate-fade-in" 
                sx={{ 
                  opacity: 0.97,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                {/* Show document content with enhanced white space and hierarchy */}
                <Box 
                  sx={{ 
                    mb: 3.5,
                    pb: 3,
                    borderBottom: '1px solid',
                    borderColor: 'rgba(0, 0, 0, 0.05)'
                  }}
                >
                  <HumanText 
                    variant="caption" 
                    color="text.secondary"
                    sx={{ 
                      display: 'block',
                      mb: 1,
                      fontSize: '0.7rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      fontWeight: 600,
                    }}
                  >
                    Content
                  </HumanText>
                  <Paper 
                    elevation={0} 
                    sx={{ 
                      p: 2, 
                      backgroundColor: alpha('#f5f8fc', 0.6),
                      borderRadius: 2,
                      border: '1px solid rgba(0, 0, 0, 0.04)',
                    }}
                  >
                    <HumanText 
                      variant="body2"
                      sx={{ 
                        fontSize: '0.8125rem', 
                        lineHeight: 1.6,
                        color: 'text.primary',
                      }}
                    >
                      {selectedPoint.content || 'No content available'}
                    </HumanText>
                  </Paper>
                </Box>
                
                {/* Metadata with improved white space and consistency */}
                <Box sx={{ mb: 'auto' }}>
                  <HumanText 
                    variant="caption" 
                    color="text.secondary"
                    sx={{ 
                      display: 'block',
                      mb: 1,
                      fontSize: '0.7rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      fontWeight: 600,
                    }}
                  >
                    Properties
                  </HumanText>
                  <Box 
                    sx={{ 
                      borderRadius: 2,
                      border: '1px solid rgba(0, 0, 0, 0.04)',
                      overflow: 'hidden',
                      mb: 3,
                    }}
                  >
                    {Object.entries(selectedPoint.metadata).map(([key, value], index) => (
                      <Box 
                        key={key} 
                        sx={{ 
                          p: 1.5,
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          borderBottom: index < Object.entries(selectedPoint.metadata).length - 1 
                            ? '1px solid rgba(0, 0, 0, 0.03)' 
                            : 'none',
                          backgroundColor: index % 2 === 0 
                            ? alpha('#f5f8fc', 0.5)
                            : 'transparent',
                        }}
                      >
                        <HumanText 
                          variant="caption" 
                          sx={{ 
                            fontWeight: 600,
                            color: 'text.secondary',
                            fontSize: '0.75rem'
                          }}
                        >
                          {key}
                        </HumanText>
                        <HumanText 
                          variant="caption" 
                          sx={{ 
                            fontSize: '0.75rem',
                            maxWidth: '60%',
                            textAlign: 'right',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            fontVariantNumeric: 'tabular-nums',
                          }}
                        >
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </HumanText>
                      </Box>
                    ))}
                  </Box>
                </Box>
                
                {/* Actions with enhanced white space and visual hierarchy */}
                <Box sx={{ mt: 3, pt: 2, display: 'flex', gap: 1.5 }}>
                  <Button
                    size="medium"
                    variant="contained"
                    color="primary"
                    fullWidth
                    onClick={() => {
                      // Filter by same category as this point
                      if (colorBy !== 'none' && colorBy in selectedPoint.metadata) {
                        const value = selectedPoint.metadata[colorBy];
                        setFilterOptions(prev => ({
                          ...prev,
                          [colorBy]: value
                        }));
                        setFilterText(`${colorBy}:${value}`);
                        trackInteraction();
                      }
                    }}
                    disabled={colorBy === 'none' || !(colorBy in selectedPoint.metadata)}
                    sx={{ 
                      boxShadow: 'none',
                      textTransform: 'none',
                      background: 'linear-gradient(90deg, #0066B3 0%, #1976d2 100%)',
                      fontWeight: 500,
                      borderRadius: 2,
                      height: 40,
                      '&:hover': {
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                        background: 'linear-gradient(90deg, #0058a3 0%, #1565c0 100%)',
                      }
                    }}
                  >
                    <SearchIcon fontSize="small" sx={{ mr: 1, fontSize: '1rem' }} />
                    <HumanText>Find Similar</HumanText>
                  </Button>
                  
                  <Button
                    size="medium"
                    variant="outlined"
                    onClick={() => setSelectedPoint(null)}
                    sx={{ 
                      minWidth: 40, 
                      width: 40,
                      height: 40,
                      p: 0,
                      borderRadius: 2,
                      borderColor: alpha('#000', 0.15),
                      color: 'text.secondary',
                      '&:hover': {
                        borderColor: alpha('#0066B3', 0.5),
                        backgroundColor: alpha('#0066B3', 0.05),
                      }
                    }}
                  >
                    <CloseIcon fontSize="small" />
                  </Button>
                </Box>
              </Box>
            ) : (
              <Box 
                sx={{ 
                  display: 'flex', 
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                  height: '100%',
                  opacity: 0.7,
                  px: 3,
                  py: 6,
                }}
              >
                <TouchAppIcon 
                  sx={{ 
                    mb: 2, 
                    fontSize: '2.5rem', 
                    color: alpha('#0066B3', 0.4),
                    opacity: 0.8,
                  }} 
                />
                <HumanText 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ 
                    maxWidth: '220px',
                    lineHeight: 1.6,
                  }}
                >
                  Select a point in the visualization to view its details
                </HumanText>
              </Box>
            )}
          </Box>
        </Collapse>
      </Box>
    </Card>
  );
};

export default VectorVisualization;