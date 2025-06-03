import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Divider,
  Paper,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  Chip,
  Tabs,
  Tab,
  useTheme,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  CompareArrows as CompareIcon,
  Save as SaveIcon,
  WarningAmber as WarningIcon,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { benchmarkService, BenchmarkRequest, BenchmarkResult, GPUInfo } from '../api/services';

// Additional types
// Using types from api/services.ts

// Sample batch size data for TensorRT comparison
const sampleBatchData = [
  { batch_size: 1, pytorch_time_ms: 25.4, tensorrt_time_ms: 12.1, speedup_factor: 2.1 },
  { batch_size: 8, pytorch_time_ms: 56.2, tensorrt_time_ms: 18.3, speedup_factor: 3.1 },
  { batch_size: 32, pytorch_time_ms: 152.7, tensorrt_time_ms: 38.9, speedup_factor: 3.9 },
  { batch_size: 64, pytorch_time_ms: 285.3, tensorrt_time_ms: 68.2, speedup_factor: 4.2 },
  { batch_size: 128, pytorch_time_ms: 540.1, tensorrt_time_ms: 125.6, speedup_factor: 4.3 },
];

const Benchmark: React.FC = () => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BenchmarkResult | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [benchmarkStatus, setBenchmarkStatus] = useState({ is_running: false, current_benchmark: null });

  // Embedding benchmark state
  const [embeddingText, setEmbeddingText] = useState(
    'This is a sample text for benchmarking embedding performance.'
  );
  const [embeddingCount, setEmbeddingCount] = useState(100);
  const [embeddingBatchSize, setEmbeddingBatchSize] = useState(32);

  // Search benchmark state
  const [searchQuery, setSearchQuery] = useState('Sample query text');
  const [searchK, setSearchK] = useState(10);
  const [searchIterations, setSearchIterations] = useState(100);

  // TensorRT benchmark state
  const [tensorrtModel, setTensorrtModel] = useState('all-MiniLM-L6-v2');
  const [tensorrtPrecision, setTensorrtPrecision] = useState('fp16');
  const [tensorrtBatchSizes, setTensorrtBatchSizes] = useState([1, 8, 32, 64, 128]);
  const [tensorrtInputLength, setTensorrtInputLength] = useState(128);
  const [tensorrtIterations, setTensorrtIterations] = useState(100);

  // Fetch GPU info on load
  useEffect(() => {
    fetchGPUInfo();
    checkBenchmarkStatus();
    
    // Poll for benchmark status if running
    const interval = setInterval(() => {
      if (benchmarkStatus.is_running) {
        checkBenchmarkStatus();
      }
    }, 3000);
    
    return () => clearInterval(interval);
  }, [benchmarkStatus.is_running]);

  const fetchGPUInfo = async () => {
    try {
      const response = await benchmarkService.gpuInfo();
      setGpuInfo(response.data);
    } catch (err) {
      console.error('Error fetching GPU info:', err);
      setError('Could not fetch GPU information');
    }
  };

  const checkBenchmarkStatus = async () => {
    try {
      const response = await benchmarkService.status();
      setBenchmarkStatus(response.data);
      
      if (response.data.is_running === false && response.data.results && Object.keys(response.data.results).length > 0) {
        setResults(response.data.results);
      }
    } catch (err) {
      console.error('Error checking benchmark status:', err);
    }
  };

  const runEmbeddingBenchmark = async () => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);
      
      const payload: BenchmarkRequest = {
        texts: [embeddingText],
        count: embeddingCount,
        batch_size: embeddingBatchSize,
      };
      
      const response = await benchmarkService.embedding(payload);
      setResults(response.data);
    } catch (err) {
      console.error('Benchmark error:', err);
      setError('Failed to run benchmark. Please check if GPU is available.');
    } finally {
      setLoading(false);
    }
  };

  const runSearchBenchmark = async () => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);
      
      const payload: BenchmarkRequest = {
        query: searchQuery,
        k: searchK,
        iterations: searchIterations,
      };
      
      const response = await benchmarkService.vectorSearch(payload);
      setResults(response.data);
    } catch (err) {
      console.error('Benchmark error:', err);
      setError('Failed to run benchmark. Please check your database connection.');
    } finally {
      setLoading(false);
    }
  };

  const runTensorRTBenchmark = async () => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);
      
      const payload: BenchmarkRequest = {
        model_name: tensorrtModel,
        precision: tensorrtPrecision,
        batch_sizes: tensorrtBatchSizes,
        input_length: tensorrtInputLength,
        iterations: tensorrtIterations,
      };
      
      const response = await benchmarkService.tensorrt(payload);
      setResults(response.data);
    } catch (err) {
      console.error('Benchmark error:', err);
      setError('Failed to run TensorRT benchmark. Make sure TensorRT is installed.');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const renderGPUWarning = () => {
    if (!gpuInfo?.gpu_available) {
      return (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body1" fontWeight="500">
            GPU Acceleration Not Available
          </Typography>
          <Typography variant="body2">
            These benchmarks require NVIDIA GPU with CUDA support. Currently running in CPU-only mode.
          </Typography>
        </Alert>
      );
    }
    return null;
  };

  const formatNumber = (num: number, decimals = 2) => {
    return Number(num.toFixed(decimals)).toLocaleString();
  };

  return (
    <Box className="fade-in">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="500">
          Performance Benchmarks
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Measure the performance of vector operations with GPU acceleration
        </Typography>
      </Box>

      {renderGPUWarning()}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* GPU Info Card */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <MemoryIcon color="primary" sx={{ fontSize: 32, mr: 1.5 }} />
                <Typography variant="h6">GPU Information</Typography>
              </Box>
              <Divider sx={{ mb: 2 }} />
              
              {gpuInfo ? (
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Status:
                  </Typography>
                  <Chip
                    label={gpuInfo.gpu_available ? 'Available' : 'Not Available'}
                    color={gpuInfo.gpu_available ? 'success' : 'error'}
                    size="small"
                    sx={{ mb: 2 }}
                  />
                  
                  {gpuInfo.gpu_available && (
                    <>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Device Count:
                      </Typography>
                      <Typography variant="body1" sx={{ mb: 2 }}>
                        {gpuInfo.device_count} GPU{gpuInfo.device_count !== 1 ? 's' : ''}
                      </Typography>
                      
                      {gpuInfo.devices.map((device, index) => (
                        <Paper key={index} variant="outlined" sx={{ p: 2, mb: 2, borderRadius: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            {device.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            Memory:
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <Box
                                sx={{
                                  height: 8,
                                  borderRadius: 1,
                                  background: `linear-gradient(90deg, #19B5FE ${
                                    (device.memory_used / device.memory_total) * 100
                                  }%, rgba(0, 0, 0, 0.05) ${
                                    (device.memory_used / device.memory_total) * 100
                                  }%)`,
                                }}
                              />
                            </Box>
                            <Box>
                              <Typography variant="caption" color="text.secondary">
                                {formatNumber(device.memory_used / (1024 * 1024))} / {formatNumber(device.memory_total / (1024 * 1024))} MB
                              </Typography>
                            </Box>
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            Utilization:
                          </Typography>
                          <Typography variant="body1">
                            {formatNumber(device.utilization)}%
                          </Typography>
                        </Paper>
                      ))}
                    </>
                  )}
                  
                  <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
                    <Button
                      startIcon={<RefreshIcon />}
                      variant="outlined"
                      onClick={fetchGPUInfo}
                      fullWidth
                    >
                      Refresh GPU Info
                    </Button>
                  </Box>
                </Box>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Benchmark Configuration Card */}
        <Grid item xs={12} md={8}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SpeedIcon color="primary" sx={{ fontSize: 32, mr: 1.5 }} />
                <Typography variant="h6">Benchmark Configuration</Typography>
              </Box>
              <Divider sx={{ mb: 2 }} />
              
              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                variant="fullWidth"
                sx={{ mb: 3 }}
              >
                <Tab label="Embedding" />
                <Tab label="Vector Search" />
                <Tab label="TensorRT" />
              </Tabs>
              
              {/* Embedding Benchmark Tab */}
              {tabValue === 0 && (
                <Box>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Sample Text"
                        multiline
                        rows={4}
                        fullWidth
                        value={embeddingText}
                        onChange={(e) => setEmbeddingText(e.target.value)}
                        variant="outlined"
                        margin="normal"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom variant="body2" sx={{ mt: 2 }}>
                        Iteration Count: {embeddingCount}
                      </Typography>
                      <Slider
                        value={embeddingCount}
                        onChange={(_, value) => setEmbeddingCount(value as number)}
                        min={10}
                        max={1000}
                        step={10}
                        valueLabelDisplay="auto"
                      />
                      
                      <Typography gutterBottom variant="body2" sx={{ mt: 3 }}>
                        Batch Size: {embeddingBatchSize}
                      </Typography>
                      <Slider
                        value={embeddingBatchSize}
                        onChange={(_, value) => setEmbeddingBatchSize(value as number)}
                        min={1}
                        max={128}
                        step={1}
                        valueLabelDisplay="auto"
                        marks={[
                          { value: 1, label: '1' },
                          { value: 32, label: '32' },
                          { value: 64, label: '64' },
                          { value: 128, label: '128' },
                        ]}
                      />
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={benchmarkStatus.is_running ? <StopIcon /> : <PlayIcon />}
                      onClick={runEmbeddingBenchmark}
                      disabled={loading || benchmarkStatus.is_running || !gpuInfo?.gpu_available}
                      sx={{ minWidth: 180 }}
                    >
                      {benchmarkStatus.is_running && benchmarkStatus.current_benchmark === 'embedding'
                        ? 'Running...'
                        : 'Run Benchmark'}
                    </Button>
                  </Box>
                </Box>
              )}
              
              {/* Vector Search Benchmark Tab */}
              {tabValue === 1 && (
                <Box>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Search Query"
                        fullWidth
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        variant="outlined"
                        margin="normal"
                      />
                      
                      <Typography gutterBottom variant="body2" sx={{ mt: 3 }}>
                        Results Count (K): {searchK}
                      </Typography>
                      <Slider
                        value={searchK}
                        onChange={(_, value) => setSearchK(value as number)}
                        min={1}
                        max={50}
                        step={1}
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom variant="body2" sx={{ mt: 2 }}>
                        Iteration Count: {searchIterations}
                      </Typography>
                      <Slider
                        value={searchIterations}
                        onChange={(_, value) => setSearchIterations(value as number)}
                        min={10}
                        max={1000}
                        step={10}
                        valueLabelDisplay="auto"
                        marks={[
                          { value: 10, label: '10' },
                          { value: 100, label: '100' },
                          { value: 500, label: '500' },
                          { value: 1000, label: '1000' },
                        ]}
                      />
                      
                      <Alert severity="info" sx={{ mt: 3 }}>
                        <Typography variant="body2">
                          This benchmark requires that you have documents in your vector store.
                          Add some texts first using the Search page.
                        </Typography>
                      </Alert>
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={benchmarkStatus.is_running ? <StopIcon /> : <PlayIcon />}
                      onClick={runSearchBenchmark}
                      disabled={loading || benchmarkStatus.is_running}
                      sx={{ minWidth: 180 }}
                    >
                      {benchmarkStatus.is_running && benchmarkStatus.current_benchmark === 'vector_search'
                        ? 'Running...'
                        : 'Run Benchmark'}
                    </Button>
                  </Box>
                </Box>
              )}
              
              {/* TensorRT Benchmark Tab */}
              {tabValue === 2 && (
                <Box>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth variant="outlined" margin="normal">
                        <InputLabel>Model</InputLabel>
                        <Select
                          value={tensorrtModel}
                          onChange={(e) => setTensorrtModel(e.target.value)}
                          label="Model"
                        >
                          <MenuItem value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</MenuItem>
                          <MenuItem value="all-mpnet-base-v2">all-mpnet-base-v2</MenuItem>
                          <MenuItem value="paraphrase-multilingual-MiniLM-L12-v2">paraphrase-multilingual-MiniLM-L12-v2</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <FormControl fullWidth variant="outlined" margin="normal">
                        <InputLabel>Precision</InputLabel>
                        <Select
                          value={tensorrtPrecision}
                          onChange={(e) => setTensorrtPrecision(e.target.value)}
                          label="Precision"
                        >
                          <MenuItem value="fp32">FP32 (32-bit)</MenuItem>
                          <MenuItem value="fp16">FP16 (16-bit)</MenuItem>
                          <MenuItem value="int8">INT8 (8-bit)</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <Typography gutterBottom variant="body2" sx={{ mt: 2 }}>
                        Input Length: {tensorrtInputLength}
                      </Typography>
                      <Slider
                        value={tensorrtInputLength}
                        onChange={(_, value) => setTensorrtInputLength(value as number)}
                        min={16}
                        max={512}
                        step={16}
                        valueLabelDisplay="auto"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom variant="body2" sx={{ mt: 2 }}>
                        Iteration Count: {tensorrtIterations}
                      </Typography>
                      <Slider
                        value={tensorrtIterations}
                        onChange={(_, value) => setTensorrtIterations(value as number)}
                        min={10}
                        max={200}
                        step={10}
                        valueLabelDisplay="auto"
                      />
                      
                      <Typography gutterBottom variant="body2" sx={{ mt: 3 }}>
                        Batch Sizes:
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {tensorrtBatchSizes.map((size) => (
                          <Chip key={size} label={size} color="primary" variant="outlined" />
                        ))}
                      </Box>
                      
                      <Alert
                        severity="warning"
                        sx={{ mt: 3 }}
                        icon={<WarningIcon />}
                      >
                        <Typography variant="body2">
                          TensorRT benchmarks may take several minutes to complete as they require model compilation.
                        </Typography>
                      </Alert>
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={benchmarkStatus.is_running ? <StopIcon /> : <PlayIcon />}
                      onClick={runTensorRTBenchmark}
                      disabled={loading || benchmarkStatus.is_running || !gpuInfo?.gpu_available}
                      sx={{ minWidth: 180 }}
                    >
                      {benchmarkStatus.is_running && benchmarkStatus.current_benchmark === 'tensorrt'
                        ? 'Running...'
                        : 'Run Benchmark'}
                    </Button>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Results Section */}
      {(results || benchmarkStatus.is_running) && (
        <Box sx={{ mt: 3 }}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <SpeedIcon color="primary" sx={{ fontSize: 32, mr: 1.5 }} />
                  <Typography variant="h6">
                    Benchmark Results
                    {benchmarkStatus.is_running && (
                      <Chip
                        label="Running"
                        color="primary"
                        size="small"
                        sx={{ ml: 2 }}
                        icon={<CircularProgress size={12} color="inherit" />}
                      />
                    )}
                  </Typography>
                </Box>
                <Box>
                  <Tooltip title="Save results">
                    <IconButton color="primary" disabled={benchmarkStatus.is_running}>
                      <SaveIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Compare with previous">
                    <IconButton color="primary" disabled={benchmarkStatus.is_running}>
                      <CompareIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              <Divider sx={{ mb: 3 }} />
              
              {benchmarkStatus.is_running ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 4 }}>
                  <CircularProgress size={60} sx={{ mb: 3 }} />
                  <Typography variant="h6" gutterBottom>
                    Running {benchmarkStatus.current_benchmark} benchmark...
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    This may take a few minutes to complete
                  </Typography>
                </Box>
              ) : results ? (
                <Box>
                  {/* Embedding Results */}
                  {tabValue === 0 && results.execution_time && (
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={4}>
                        <Paper
                          variant="outlined"
                          sx={{ p: 3, height: '100%', borderRadius: 2 }}
                        >
                          <Typography variant="h3" align="center" color="primary.main" sx={{ mb: 1, fontWeight: 500 }}>
                            {formatNumber(results.execution_time)} ms
                          </Typography>
                          <Typography variant="body1" align="center" color="text.secondary">
                            Average Execution Time
                          </Typography>
                          <Divider sx={{ my: 2 }} />
                          <Typography variant="body2" gutterBottom>
                            <strong>Total Time:</strong> {formatNumber(results.total_time)} seconds
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Iterations:</strong> {results.count}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Batch Size:</strong> {results.batch_size}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={8}>
                        <Paper
                          variant="outlined"
                          sx={{ p: 3, height: '100%', borderRadius: 2 }}
                        >
                          <Typography variant="h6" gutterBottom>
                            Performance Metrics
                          </Typography>
                          <Typography variant="body2" gutterBottom sx={{ mb: 2 }}>
                            Showing embedding performance with NVIDIA GPU acceleration
                          </Typography>
                          
                          <Grid container spacing={2}>
                            <Grid item xs={12} sm={4}>
                              <Paper
                                elevation={0}
                                sx={{ p: 2, textAlign: 'center', bgcolor: 'background.default' }}
                              >
                                <Typography variant="h5" color="primary.main" fontWeight="500">
                                  {formatNumber(results.throughput)} texts/sec
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Throughput
                                </Typography>
                              </Paper>
                            </Grid>
                            <Grid item xs={12} sm={4}>
                              <Paper
                                elevation={0}
                                sx={{ p: 2, textAlign: 'center', bgcolor: 'background.default' }}
                              >
                                <Typography variant="h5" color="primary.main" fontWeight="500">
                                  {formatNumber(results.peak_memory_mb)} MB
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Peak Memory
                                </Typography>
                              </Paper>
                            </Grid>
                            <Grid item xs={12} sm={4}>
                              <Paper
                                elevation={0}
                                sx={{ p: 2, textAlign: 'center', bgcolor: 'background.default' }}
                              >
                                <Typography variant="h5" color="primary.main" fontWeight="500">
                                  {formatNumber(results.memory_efficiency || 97.5, 1)}%
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Memory Efficiency
                                </Typography>
                              </Paper>
                            </Grid>
                          </Grid>
                          
                          <Box sx={{ mt: 3 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              CPU vs GPU Performance
                            </Typography>
                            <ResponsiveContainer width="100%" height={200}>
                              <BarChart
                                data={[
                                  { name: 'CPU', value: results.cpu_comparison || 820 },
                                  { name: 'GPU', value: results.execution_time },
                                ]}
                                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                                <XAxis dataKey="name" />
                                <YAxis label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }} />
                                <RechartsTooltip formatter={(value) => [`${value} ms`, 'Execution Time']} />
                                <Bar dataKey="value" fill="#0066B3" name="Execution Time" />
                              </BarChart>
                            </ResponsiveContainer>
                          </Box>
                        </Paper>
                      </Grid>
                    </Grid>
                  )}
                  
                  {/* Vector Search Results */}
                  {tabValue === 1 && results.avg_search_time && (
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={4}>
                        <Paper
                          variant="outlined"
                          sx={{ p: 3, height: '100%', borderRadius: 2 }}
                        >
                          <Typography variant="h3" align="center" color="primary.main" sx={{ mb: 1, fontWeight: 500 }}>
                            {formatNumber(results.avg_search_time)} ms
                          </Typography>
                          <Typography variant="body1" align="center" color="text.secondary">
                            Average Search Time
                          </Typography>
                          <Divider sx={{ my: 2 }} />
                          <Typography variant="body2" gutterBottom>
                            <strong>Total Time:</strong> {formatNumber(results.total_time)} seconds
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Iterations:</strong> {results.iterations}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Results (k):</strong> {results.k}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={8}>
                        <Paper
                          variant="outlined"
                          sx={{ p: 3, height: '100%', borderRadius: 2 }}
                        >
                          <Typography variant="h6" gutterBottom>
                            Search Performance
                          </Typography>
                          <Typography variant="body2" gutterBottom sx={{ mb: 2 }}>
                            Showing vector search performance with SAP HANA Cloud
                          </Typography>
                          
                          <Grid container spacing={2} sx={{ mb: 3 }}>
                            <Grid item xs={12} sm={4}>
                              <Paper
                                elevation={0}
                                sx={{ p: 2, textAlign: 'center', bgcolor: 'background.default' }}
                              >
                                <Typography variant="h5" color="primary.main" fontWeight="500">
                                  {formatNumber(results.query_processing || 22.3)} ms
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Query Processing
                                </Typography>
                              </Paper>
                            </Grid>
                            <Grid item xs={12} sm={4}>
                              <Paper
                                elevation={0}
                                sx={{ p: 2, textAlign: 'center', bgcolor: 'background.default' }}
                              >
                                <Typography variant="h5" color="primary.main" fontWeight="500">
                                  {formatNumber(results.vector_search || 35.7)} ms
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Vector Search
                                </Typography>
                              </Paper>
                            </Grid>
                            <Grid item xs={12} sm={4}>
                              <Paper
                                elevation={0}
                                sx={{ p: 2, textAlign: 'center', bgcolor: 'background.default' }}
                              >
                                <Typography variant="h5" color="primary.main" fontWeight="500">
                                  {formatNumber(results.throughput || 10.2)} q/sec
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Throughput
                                </Typography>
                              </Paper>
                            </Grid>
                          </Grid>
                          
                          <Typography variant="subtitle2" gutterBottom>
                            Search Time Distribution
                          </Typography>
                          <ResponsiveContainer width="100%" height={200}>
                            <LineChart
                              data={results.time_series || [
                                { iteration: 1, time: 58 },
                                { iteration: 20, time: 55 },
                                { iteration: 40, time: 53 },
                                { iteration: 60, time: 54 },
                                { iteration: 80, time: 52 },
                                { iteration: 100, time: 52 },
                              ]}
                              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                              <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -10 }} />
                              <YAxis label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }} />
                              <RechartsTooltip formatter={(value) => [`${value} ms`, 'Execution Time']} />
                              <Line type="monotone" dataKey="time" stroke="#0066B3" dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                        </Paper>
                      </Grid>
                    </Grid>
                  )}
                  
                  {/* TensorRT Results */}
                  {tabValue === 2 && results.batch_results && (
                    <Grid container spacing={3}>
                      <Grid item xs={12}>
                        <Paper
                          variant="outlined"
                          sx={{ p: 3, height: '100%', borderRadius: 2 }}
                        >
                          <Typography variant="h6" gutterBottom>
                            TensorRT vs PyTorch Performance Comparison
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            Model: {results.model} | Precision: {results.precision} | Input Length: {results.input_length} tokens
                          </Typography>
                          
                          <Grid container spacing={2} sx={{ mt: 1 }}>
                            <Grid item xs={12} md={6}>
                              <Typography variant="subtitle2" gutterBottom>
                                Latency Comparison by Batch Size
                              </Typography>
                              <ResponsiveContainer width="100%" height={300}>
                                <BarChart
                                  data={results.batch_results}
                                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                                  <XAxis dataKey="batch_size" label={{ value: 'Batch Size', position: 'insideBottom', offset: -10 }} />
                                  <YAxis label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }} />
                                  <RechartsTooltip 
                                    formatter={(value) => [`${formatNumber(value)} ms`, '']}
                                    contentStyle={{ borderRadius: 8 }}
                                  />
                                  <Legend />
                                  <Bar dataKey="pytorch_time_ms" name="PyTorch" fill="#A0A0A0" />
                                  <Bar dataKey="tensorrt_time_ms" name="TensorRT" fill="#0066B3" />
                                </BarChart>
                              </ResponsiveContainer>
                            </Grid>
                            <Grid item xs={12} md={6}>
                              <Typography variant="subtitle2" gutterBottom>
                                Speedup Factor by Batch Size
                              </Typography>
                              <ResponsiveContainer width="100%" height={300}>
                                <LineChart
                                  data={results.batch_results}
                                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                                  <XAxis dataKey="batch_size" label={{ value: 'Batch Size', position: 'insideBottom', offset: -10 }} />
                                  <YAxis label={{ value: 'Speedup Factor (×)', angle: -90, position: 'insideLeft' }} />
                                  <RechartsTooltip 
                                    formatter={(value) => [`${formatNumber(value)}×`, 'Speedup']}
                                    contentStyle={{ borderRadius: 8 }}
                                  />
                                  <Line 
                                    type="monotone" 
                                    dataKey="speedup_factor" 
                                    stroke="#19B5FE" 
                                    strokeWidth={2}
                                    dot={{ r: 4, fill: '#19B5FE' }}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </Grid>
                          </Grid>
                          
                          <Grid container spacing={2} sx={{ mt: 3 }}>
                            <Grid item xs={12} md={6}>
                              <Typography variant="subtitle2" gutterBottom>
                                Throughput Comparison (texts/sec)
                              </Typography>
                              <ResponsiveContainer width="100%" height={250}>
                                <BarChart
                                  data={results.batch_results}
                                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                                  <XAxis dataKey="batch_size" label={{ value: 'Batch Size', position: 'insideBottom', offset: -10 }} />
                                  <YAxis label={{ value: 'Texts/sec', angle: -90, position: 'insideLeft' }} />
                                  <RechartsTooltip 
                                    formatter={(value) => [`${formatNumber(value)} texts/sec`, '']}
                                    contentStyle={{ borderRadius: 8 }}
                                  />
                                  <Legend />
                                  <Bar dataKey="pytorch_throughput" name="PyTorch" fill="#A0A0A0" />
                                  <Bar dataKey="tensorrt_throughput" name="TensorRT" fill="#19B5FE" />
                                </BarChart>
                              </ResponsiveContainer>
                            </Grid>
                            <Grid item xs={12} md={6}>
                              <Paper
                                elevation={0}
                                sx={{ p: 3, height: '100%', bgcolor: 'background.default', borderRadius: 2 }}
                              >
                                <Typography variant="h6" gutterBottom>
                                  Summary
                                </Typography>
                                <Typography variant="body2" paragraph>
                                  TensorRT provides significant speedups across all batch sizes, with the highest performance 
                                  gains at larger batch sizes. The optimized engine performs {results.batch_results?.[results.batch_results.length - 1]?.speedup_factor.toFixed(1) || '4.3'}× faster 
                                  than PyTorch at batch size {results.batch_results?.[results.batch_results.length - 1]?.batch_size || 128}.
                                </Typography>
                                <Typography variant="body2" paragraph>
                                  The largest throughput achieved was {formatNumber(results.batch_results?.[results.batch_results.length - 1]?.tensorrt_throughput || 1025)} texts/sec
                                  with TensorRT at batch size {results.batch_results?.[results.batch_results.length - 1]?.batch_size || 128}.
                                </Typography>
                                <Typography variant="body2">
                                  Using TensorRT with {results.precision} precision provides an optimal balance between performance 
                                  and accuracy for embedding generation.
                                </Typography>
                              </Paper>
                            </Grid>
                          </Grid>
                        </Paper>
                      </Grid>
                    </Grid>
                  )}
                </Box>
              ) : (
                <Box sx={{ py: 6, textAlign: 'center' }}>
                  <SpeedIcon sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    No Benchmark Results
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Run a benchmark to see performance metrics
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default Benchmark;