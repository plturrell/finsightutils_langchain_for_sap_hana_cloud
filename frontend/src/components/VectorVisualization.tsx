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
} from '@mui/material';
import Plot from 'react-plotly.js';
import { developerService, GetVectorsRequest } from '../api/services';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

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
  
  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ padding: 2, flexGrow: 0 }}>
        <Typography variant="h6" gutterBottom>
          Vector Space Visualization
        </Typography>
        
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <ToggleButtonGroup
              value={visualizationMode}
              exclusive
              onChange={(e, newMode) => newMode && setVisualizationMode(newMode)}
              aria-label="visualization mode"
              size="small"
              fullWidth
            >
              <ToggleButton value="2d">2D Plot</ToggleButton>
              <ToggleButton value="3d">3D Explorer</ToggleButton>
            </ToggleButtonGroup>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <FormControl size="small" fullWidth>
              <InputLabel>Color By</InputLabel>
              <Select
                value={colorBy}
                label="Color By"
                onChange={(e) => setColorBy(e.target.value)}
              >
                <MenuItem value="none">None</MenuItem>
                <MenuItem value="cluster">Cluster</MenuItem>
                {metadataFields.map(field => (
                  <MenuItem key={field} value={field}>{field}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <FormControl size="small" fullWidth>
              <InputLabel>Size By</InputLabel>
              <Select
                value={sizeBy}
                label="Size By"
                onChange={(e) => setSizeBy(e.target.value)}
              >
                <MenuItem value="none">Uniform</MenuItem>
                {metadataFields
                  .filter(field => typeof points[0]?.metadata[field] === 'number')
                  .map(field => (
                    <MenuItem key={field} value={field}>{field}</MenuItem>
                  ))
                }
              </Select>
            </FormControl>
          </Grid>
          
          {/* Advanced Visualization Settings */}
          <Grid item xs={12} sm={4}>
            <FormControl size="small" fullWidth>
              <InputLabel>Clustering Algorithm</InputLabel>
              <Select
                value={clusteringAlgorithm}
                label="Clustering Algorithm"
                onChange={(e) => setClusteringAlgorithm(e.target.value as 'kmeans' | 'dbscan' | 'hdbscan')}
              >
                <MenuItem value="kmeans">K-Means</MenuItem>
                <MenuItem value="dbscan">DBSCAN</MenuItem>
                <MenuItem value="hdbscan">HDBSCAN</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <FormControl size="small" fullWidth>
              <InputLabel>Dimensionality Reduction</InputLabel>
              <Select
                value={dimensionalityReduction}
                label="Dimensionality Reduction"
                onChange={(e) => setDimensionalityReduction(e.target.value as 'tsne' | 'umap' | 'pca')}
              >
                <MenuItem value="tsne">t-SNE</MenuItem>
                <MenuItem value="umap">UMAP</MenuItem>
                <MenuItem value="pca">PCA</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <FormControl size="small" fullWidth>
              <InputLabel>Results Per Page</InputLabel>
              <Select
                value={pageSize}
                label="Results Per Page"
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setPage(1); // Reset to first page
                }}
              >
                <MenuItem value={50}>50</MenuItem>
                <MenuItem value={100}>100</MenuItem>
                <MenuItem value={200}>200</MenuItem>
                <MenuItem value={500}>500</MenuItem>
                <MenuItem value={1000}>1000</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          {/* Real-time filtering */}
          <Grid item xs={12}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Filter Vectors
              </Typography>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={8}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Filter Query"
                    value={filterText}
                    onChange={(e) => setFilterText(e.target.value)}
                    placeholder="content:query or field:value or {\"field\":\"value\"}"
                    helperText="Enter content search term, field:value pair, or JSON filter"
                  />
                </Grid>
                <Grid item xs={6} sm={2}>
                  <Button 
                    fullWidth
                    variant="contained" 
                    color="primary"
                    onClick={applyFilter}
                  >
                    Apply
                  </Button>
                </Grid>
                <Grid item xs={6} sm={2}>
                  <Button 
                    fullWidth
                    variant="outlined"
                    onClick={resetFilters}
                  >
                    Reset
                  </Button>
                </Grid>
              </Grid>
              
              {/* Active filters display */}
              {Object.keys(filterOptions).length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    Active Filters:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                    {Object.entries(filterOptions).map(([key, value]) => (
                      <Chip
                        key={key}
                        label={`${key}: ${value}`}
                        size="small"
                        onDelete={() => {
                          const newFilters = { ...filterOptions };
                          delete newFilters[key];
                          setFilterOptions(newFilters);
                        }}
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </Paper>
          </Grid>
          
          {visualizationMode === '2d' && (
            <Grid item xs={12}>
              <Typography id="perplexity-slider" gutterBottom>
                Perplexity: {perplexity}
              </Typography>
              <Slider
                value={perplexity}
                onChange={(e, value) => setPerplexity(value as number)}
                aria-labelledby="perplexity-slider"
                valueLabelDisplay="auto"
                min={5}
                max={50}
                marks={[
                  { value: 5, label: '5' },
                  { value: 30, label: '30' },
                  { value: 50, label: '50' },
                ]}
              />
            </Grid>
          )}
          
          {visualizationMode === '3d' && (
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={useAutoRotate}
                    onChange={(e) => setUseAutoRotate(e.target.checked)}
                  />
                }
                label="Auto-Rotate"
              />
            </Grid>
          )}
        </Grid>
      </CardContent>
      
      <Box sx={{ flexGrow: 1, display: 'flex', overflow: 'hidden' }}>
        <Grid container spacing={0} sx={{ flexGrow: 1 }}>
          <Grid item xs={12} md={9} sx={{ height: '100%' }}>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
              </Box>
            ) : error ? (
              <Box sx={{ p: 3 }}>
                <Alert severity="error">{error}</Alert>
              </Box>
            ) : (
              <Box
                ref={containerRef}
                sx={{
                  width: '100%',
                  height: '100%',
                  minHeight: 400,
                  position: 'relative',
                }}
              >
                {visualizationMode === '2d' && render2DPlot()}
              </Box>
            )}
            
            {/* Pagination controls */}
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              p: 1, 
              borderTop: '1px solid rgba(0,0,0,0.12)'
            }}>
              <Typography variant="body2" color="text.secondary" sx={{ mr: 2 }}>
                {totalCount > 0 ? 
                  `Showing ${((page - 1) * pageSize) + 1}-${Math.min(page * pageSize, totalCount)} of ${totalCount} vectors` : 
                  'No vectors found'}
              </Typography>
              
              <Stack direction="row" spacing={1}>
                <Button
                  size="small"
                  disabled={page === 1}
                  onClick={() => setPage(1)}
                >
                  First
                </Button>
                <Button
                  size="small"
                  disabled={page === 1}
                  onClick={() => setPage(page - 1)}
                >
                  Prev
                </Button>
                <Typography variant="body2" sx={{ alignSelf: 'center', mx: 1 }}>
                  {page} / {totalPages}
                </Typography>
                <Button
                  size="small"
                  disabled={page === totalPages}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </Button>
                <Button
                  size="small"
                  disabled={page === totalPages}
                  onClick={() => setPage(totalPages)}
                >
                  Last
                </Button>
              </Stack>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={3} sx={{ height: '100%', overflow: 'auto', borderLeft: '1px solid rgba(0,0,0,0.12)' }}>
            <Box sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Document Details
              </Typography>
              
              {selectedPoint ? (
                <>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                    Content:
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 1, mb: 2, backgroundColor: '#f9f9f9' }}>
                    <Typography variant="body2">
                      {selectedPoint.content || 'No content available'}
                    </Typography>
                  </Paper>
                  
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                    Metadata:
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 1, backgroundColor: '#f9f9f9' }}>
                    {Object.entries(selectedPoint.metadata).map(([key, value]) => (
                      <Box key={key} sx={{ mb: 0.5 }}>
                        <Typography variant="caption" component="span" sx={{ fontWeight: 'bold' }}>
                          {key}:
                        </Typography>{' '}
                        <Typography variant="caption" component="span">
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </Typography>
                      </Box>
                    ))}
                  </Paper>
                  
                  <Box sx={{ mt: 2 }}>
                    <Button
                      size="small"
                      variant="outlined"
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
                        }
                      }}
                    >
                      Find Similar
                    </Button>
                  </Box>
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Click on a point to view details
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Card>
  );
};

export default VectorVisualization;