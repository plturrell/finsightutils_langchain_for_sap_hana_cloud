import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  CircularProgress,
  Chip,
  Button,
  IconButton,
  Tooltip,
  Alert,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  QueryStats as QueryStatsIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { healthService, gpuService, analyticsService } from '../api/services';

// Types
interface GPUDevice {
  name: string;
  memory_total: number;
  memory_used: number;
  memory_free: number;
  utilization: number;
}

interface GPUInfo {
  device_count: number;
  devices: GPUDevice[];
}

interface HealthStatus {
  status: string;
  database: string;
  gpu_acceleration: string;
  gpu_count: number;
  gpu_info: GPUDevice[];
}

interface RecentQuery {
  id: number;
  query: string;
  timestamp: string;
  results_count: number;
  execution_time: number;
}

// Initialize empty arrays for data that will be loaded from the API
const initialRecentQueries: RecentQuery[] = [];
const initialPerformanceData: { name: string; queries: number; avgTime: number }[] = [];

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recentQueries, setRecentQueries] = useState<RecentQuery[]>(initialRecentQueries);
  const [performanceStats, setPerformanceStats] = useState(initialPerformanceData);

  const fetchHealthStatus = async () => {
    try {
      setLoading(true);
      const response = await healthService.check();
      setHealthStatus(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch health status');
      console.error('Error fetching health status:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchGPUInfo = async () => {
    try {
      const response = await gpuService.info();
      setGpuInfo(response.data);
    } catch (err) {
      console.error('Error fetching GPU info:', err);
    }
  };

  const fetchRecentQueries = async () => {
    try {
      const response = await analyticsService.getRecentQueries();
      setRecentQueries(response.data);
    } catch (err) {
      console.error('Error fetching recent queries:', err);
    }
  };

  const fetchPerformanceStats = async () => {
    try {
      const response = await analyticsService.getPerformanceStats();
      setPerformanceStats(response.data);
    } catch (err) {
      console.error('Error fetching performance stats:', err);
    }
  };

  useEffect(() => {
    fetchHealthStatus();
    fetchGPUInfo();
    fetchRecentQueries();
    fetchPerformanceStats();
    
    // Set up polling for health status every 30 seconds
    const interval = setInterval(() => {
      fetchHealthStatus();
      fetchGPUInfo();
      fetchRecentQueries();
      fetchPerformanceStats();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchHealthStatus();
    fetchGPUInfo();
    fetchRecentQueries();
    fetchPerformanceStats();
  };

  return (
    <Box className="fade-in">
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Typography variant="h4" fontWeight="500">
          Dashboard
        </Typography>
        <Tooltip title="Refresh data">
          <IconButton onClick={handleRefresh} color="primary">
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* System Status Card */}
        <Grid item xs={12} md={6} lg={3}>
          <Card
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <CardHeader
              title="System Status"
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              {loading ? (
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: '100%',
                  }}
                >
                  <CircularProgress size={40} />
                </Box>
              ) : (
                <Box>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      mb: 2,
                    }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: 60,
                        height: 60,
                        borderRadius: '50%',
                        bgcolor: healthStatus?.status === 'healthy' ? 'success.light' : 'error.light',
                        color: healthStatus?.status === 'healthy' ? 'success.dark' : 'error.dark',
                        mr: 2,
                      }}
                    >
                      {healthStatus?.status === 'healthy' ? (
                        <CheckCircleIcon fontSize="large" />
                      ) : (
                        <ErrorIcon fontSize="large" />
                      )}
                    </Box>
                    <Box>
                      <Typography variant="h6" fontWeight="500">
                        {healthStatus?.status === 'healthy' ? 'All Systems Operational' : 'System Issue Detected'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Last updated: {new Date().toLocaleTimeString()}
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Service Status
                    </Typography>
                    <Box
                      sx={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: 1,
                        mt: 1,
                      }}
                    >
                      <Chip
                        label="Database"
                        color={healthStatus?.database === 'connected' ? 'success' : 'error'}
                        size="small"
                        icon={<StorageIcon />}
                      />
                      <Chip
                        label={`GPU ${healthStatus?.gpu_acceleration === 'available' ? 'Active' : 'Inactive'}`}
                        color={healthStatus?.gpu_acceleration === 'available' ? 'success' : 'default'}
                        size="small"
                        icon={<MemoryIcon />}
                      />
                      <Chip
                        label="API"
                        color="success"
                        size="small"
                        icon={<InfoIcon />}
                      />
                    </Box>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* GPU Stats Card */}
        <Grid item xs={12} md={6} lg={3}>
          <Card
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <CardHeader
              title="GPU Resources"
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              {loading ? (
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: '100%',
                  }}
                >
                  <CircularProgress size={40} />
                </Box>
              ) : (
                <Box>
                  {gpuInfo?.device_count && gpuInfo.device_count > 0 ? (
                    <>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          mb: 2,
                        }}
                      >
                        <MemoryIcon
                          color="primary"
                          sx={{ fontSize: 48, mr: 2 }}
                        />
                        <Box>
                          <Typography variant="h6" fontWeight="500">
                            {gpuInfo.device_count} GPU{gpuInfo.device_count > 1 ? 's' : ''} Available
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            NVIDIA CUDA Acceleration
                          </Typography>
                        </Box>
                      </Box>

                      <Box sx={{ mt: 2 }}>
                        {gpuInfo.devices.map((device, index) => (
                          <Box key={index} sx={{ mb: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                              <Typography variant="subtitle2">{device.name}</Typography>
                              <Typography variant="subtitle2" color="text.secondary">
                                {Math.round(device.utilization)}% Load
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box sx={{ width: '100%', mr: 1 }}>
                                <Box
                                  sx={{
                                    height: 8,
                                    borderRadius: 1,
                                    background: `linear-gradient(90deg, #19B5FE ${device.memory_used / device.memory_total * 100}%, rgba(0, 0, 0, 0.05) ${device.memory_used / device.memory_total * 100}%)`,
                                  }}
                                />
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">
                                  {Math.round(device.memory_used / (1024 * 1024))} / {Math.round(device.memory_total / (1024 * 1024))} MB
                                </Typography>
                              </Box>
                            </Box>
                          </Box>
                        ))}
                      </Box>
                    </>
                  ) : (
                    <Box sx={{ textAlign: 'center', py: 3 }}>
                      <MemoryIcon color="disabled" sx={{ fontSize: 60, mb: 2, opacity: 0.6 }} />
                      <Typography variant="body1" gutterBottom>
                        No GPU Acceleration
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Running in CPU-only mode
                      </Typography>
                    </Box>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Card */}
        <Grid item xs={12} md={6} lg={6}>
          <Card
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <CardHeader
              title="Query Performance"
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart
                  data={performanceStats.length > 0 ? performanceStats : []}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                  <XAxis dataKey="name" stroke="#666666" />
                  <YAxis yAxisId="left" stroke="#0066B3" />
                  <YAxis yAxisId="right" orientation="right" stroke="#19B5FE" />
                  <RechartsTooltip 
                    contentStyle={{ borderRadius: 8, boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)' }} 
                    formatter={(value, name) => {
                      if (name === 'avgTime') {
                        return [`${value} ms`, 'Avg Response Time'];
                      }
                      return [value.toLocaleString(), 'Queries'];
                    }}
                  />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="queries"
                    name="Queries"
                    stroke="#0066B3"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="avgTime"
                    name="avgTime"
                    stroke="#19B5FE"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 4, mt: 2 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" fontWeight="500" color="primary.main">
                    {performanceStats.length > 0 
                      ? performanceStats[performanceStats.length - 1].queries.toLocaleString() 
                      : "0"}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Queries Last Month
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" fontWeight="500" color="secondary.main">
                    {performanceStats.length > 0 
                      ? `${performanceStats[performanceStats.length - 1].avgTime} ms` 
                      : "0 ms"}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Average Response Time
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Queries Card */}
        <Grid item xs={12} md={6} lg={6}>
          <Card
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <CardHeader
              title="Recent Queries"
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 1 }}
            />
            <CardContent sx={{ pt: 1, pb: 1, flexGrow: 1 }}>
              <List disablePadding>
                {recentQueries.map((query) => (
                  <React.Fragment key={query.id}>
                    <ListItem disablePadding sx={{ py: 1 }}>
                      <ListItemText
                        primary={
                          <Typography
                            variant="body1"
                            sx={{
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            }}
                          >
                            {query.query}
                          </Typography>
                        }
                        secondary={
                          <Box
                            sx={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              mt: 0.5,
                            }}
                          >
                            <Typography variant="caption" color="text.secondary">
                              {new Date(query.timestamp).toLocaleString()}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Chip
                                label={`${query.results_count} results`}
                                size="small"
                                variant="outlined"
                              />
                              <Chip
                                label={`${query.execution_time} ms`}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                            </Box>
                          </Box>
                        }
                      />
                    </ListItem>
                    <Divider />
                  </React.Fragment>
                ))}
              </List>
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                <Button
                  variant="outlined"
                  color="primary"
                  size="small"
                  startIcon={<QueryStatsIcon />}
                  sx={{ borderRadius: 4 }}
                >
                  View All Queries
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* GPU vs CPU Performance Card */}
        <Grid item xs={12} md={6} lg={6}>
          <Card
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <CardHeader
              title="GPU vs CPU Performance"
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              <ResponsiveContainer width="100%" height={270}>
                <BarChart
                  data={[
                    { name: 'Embedding', CPU: 250, GPU: 25, TensorRT: 12 },
                    { name: 'Vector Search', CPU: 180, GPU: 30, TensorRT: 20 },
                    { name: 'Batch Processing', CPU: 420, GPU: 60, TensorRT: 28 },
                  ]}
                  margin={{
                    top: 20,
                    right: 30,
                    left: 20,
                    bottom: 20,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                  <XAxis dataKey="name" stroke="#666666" />
                  <YAxis stroke="#666666" label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft', offset: -5 }} />
                  <RechartsTooltip contentStyle={{ borderRadius: 8, boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)' }} />
                  <Legend verticalAlign="top" height={40} />
                  <Bar dataKey="CPU" name="CPU" fill="#A0A0A0" radius={[4, 4, 0, 0]} barSize={25} />
                  <Bar dataKey="GPU" name="CUDA GPU" fill="#19B5FE" radius={[4, 4, 0, 0]} barSize={25} />
                  <Bar dataKey="TensorRT" name="TensorRT" fill="#0066B3" radius={[4, 4, 0, 0]} barSize={25} />
                </BarChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                <Button
                  variant="outlined"
                  color="primary"
                  size="small"
                  startIcon={<SpeedIcon />}
                  sx={{ borderRadius: 4 }}
                >
                  Run Benchmark
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* System Info Card */}
        <Grid item xs={12} md={6} lg={6}>
          <Card
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <CardHeader
              title="System Information"
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2,
                      bgcolor: 'background.default',
                      height: '100%',
                    }}
                  >
                    <Typography variant="subtitle2" gutterBottom>
                      Database
                    </Typography>
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Connection:
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        SAP HANA Cloud
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Version:
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        2023.25.0
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Status:
                      </Typography>
                      <Typography variant="body1" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <CheckCircleIcon color="success" fontSize="small" />
                        Connected
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2,
                      bgcolor: 'background.default',
                      height: '100%',
                    }}
                  >
                    <Typography variant="subtitle2" gutterBottom>
                      API
                    </Typography>
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Version:
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        1.0.0
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Embedding Model:
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        {gpuInfo?.device_count && gpuInfo.device_count > 0
                          ? "all-MiniLM-L6-v2 (GPU)"
                          : "SAP_NEB.20240715 (CPU)"}
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Vector Dimension:
                      </Typography>
                      <Typography variant="body1">
                        384
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;