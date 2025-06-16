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
  alpha,
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
import { useSpring, animated, useTrail, config, useChain, useSpringRef } from '@react-spring/web';
import useErrorHandler from '../hooks/useErrorHandler';
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

// Animated MUI components
const AnimatedCard = animated(Card);
const AnimatedGrid = animated(Grid);
const AnimatedTypography = animated(Typography);
const AnimatedChip = animated(Chip);
const AnimatedButton = animated(Button);
const AnimatedIconButton = animated(IconButton);
const AnimatedBox = animated(Box);

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
  const [performanceComparison, setPerformanceComparison] = useState<any[]>([]);
  const [animationsVisible, setAnimationsVisible] = useState(false);

  // Spring refs for sequencing animations
  const headerSpringRef = useSpringRef();
  const metricsSpringRef = useSpringRef();
  const chartsSpringRef = useSpringRef();
  const tablesSpringRef = useSpringRef();

  // Header animation
  const headerAnimation = useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });

  // Metrics card animations with trail effect
  const cardTrail = useTrail(4, {
    ref: metricsSpringRef,
    from: { opacity: 0, transform: 'translateY(40px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(40px)' },
    config: { mass: 1, tension: 280, friction: 60 }
  });

  // Charts animation
  const chartsAnimation = useSpring({
    ref: chartsSpringRef,
    from: { opacity: 0, scale: 0.9 },
    to: { opacity: animationsVisible ? 1 : 0, scale: animationsVisible ? 1 : 0.9 },
    config: { tension: 300, friction: 30 }
  });

  // Tables and additional content animations
  const tablesAnimation = useSpring({
    ref: tablesSpringRef,
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
    config: { tension: 280, friction: 60 }
  });

  // Refresh button animation
  const refreshButtonAnimation = useSpring({
    from: { transform: 'rotate(0deg)' },
    to: { transform: loading ? 'rotate(360deg)' : 'rotate(0deg)' },
    config: { duration: loading ? 1000 : 0 },
    loop: loading,
  });

  // Metrics value animations
  const metricsAnimation = {
    utilization: useSpring({
      from: { value: 0 },
      to: { value: healthStatus && gpuInfo ? gpuInfo.devices[0]?.utilization || 0 : 0 },
      config: { tension: 140, friction: 20, duration: 1500 }
    }),
    memory: useSpring({
      from: { value: 0 },
      to: { value: healthStatus && gpuInfo ? gpuInfo.devices[0]?.memory_used || 0 : 0 },
      config: { tension: 140, friction: 20, duration: 1500 }
    }),
    temperature: useSpring({
      from: { value: 0 },
      to: { value: 68 }, // Mock temperature value for animation
      config: { tension: 140, friction: 20, duration: 1500 }
    }),
    queries: useSpring({
      from: { value: 0 },
      to: { value: performanceStats.length > 0 ? performanceStats[performanceStats.length - 1].queries : 0 },
      config: { tension: 140, friction: 20, duration: 2000 }
    })
  };

  // Chain animations in sequence
  useChain(
    animationsVisible 
      ? [headerSpringRef, metricsSpringRef, chartsSpringRef, tablesSpringRef] 
      : [tablesSpringRef, chartsSpringRef, metricsSpringRef, headerSpringRef],
    animationsVisible 
      ? [0, 0.3, 0.6, 0.8] 
      : [0, 0.2, 0.4, 0.6]
  );

  // Make animations visible after initial data load
  useEffect(() => {
    if (!loading && (healthStatus || gpuInfo)) {
      setTimeout(() => setAnimationsVisible(true), 300);
    }
  }, [loading, healthStatus, gpuInfo]);

  // Import error handling hook
  const { safeFetch } = useErrorHandler();

  const fetchHealthStatus = async () => {
    setLoading(true);
    const response = await safeFetch(
      () => healthService.check(),
      { 
        fallback: { data: null },
        customErrorHandler: () => setError('Failed to fetch health status')
      }
    );
    if (response) {
      setHealthStatus(response.data);
      setError(null);
    }
    setLoading(false);
  };

  const fetchGPUInfo = async () => {
    const response = await safeFetch(
      () => gpuService.info(),
      { 
        fallback: { data: null },
        silentError: true  // Don't show error to user for this non-critical data
      }
    );
    if (response) {
      setGpuInfo(response.data);
    }
  };

  const fetchRecentQueries = async () => {
    const response = await safeFetch(
      () => analyticsService.getRecentQueries(),
      { 
        fallback: { data: [] },
        silentError: true  // Don't show error for non-critical data
      }
    );
    if (response) {
      setRecentQueries(response.data);
    }
  };

  const fetchPerformanceStats = async () => {
    const response = await safeFetch(
      () => analyticsService.getPerformanceStats(),
      { 
        fallback: { data: [] },
        silentError: true  // Don't show error for non-critical data
      }
    );
    if (response) {
      setPerformanceStats(response.data);
    }
  };
  
  const fetchPerformanceComparison = async () => {
    const response = await safeFetch(
      () => analyticsService.getPerformanceComparison(),
      { 
        fallback: { 
          data: [
            { name: 'Embedding', CPU: 250, GPU: 25, TensorRT: 12 },
            { name: 'Vector Search', CPU: 180, GPU: 30, TensorRT: 20 },
            { name: 'Batch Processing', CPU: 420, GPU: 60, TensorRT: 28 }
          ]
        },
        silentError: true  // Use fallback data if the API fails
      }
    );
    if (response) {
      setPerformanceComparison(response.data);
    }
  };

  useEffect(() => {
    fetchHealthStatus();
    fetchGPUInfo();
    fetchRecentQueries();
    fetchPerformanceStats();
    fetchPerformanceComparison();
    
    // Set up polling for health status every 30 seconds
    const interval = setInterval(() => {
      fetchHealthStatus();
      fetchGPUInfo();
      fetchRecentQueries();
      fetchPerformanceStats();
      fetchPerformanceComparison();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchHealthStatus();
    fetchGPUInfo();
    fetchRecentQueries();
    fetchPerformanceStats();
    fetchPerformanceComparison();
  };

  return (
    <AnimatedBox 
      className="fade-in"
      style={useSpring({
        from: { opacity: 0 },
        to: { opacity: 1 },
        config: { tension: 280, friction: 60 }
      })}
    >
      <AnimatedBox
        style={headerAnimation}
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <AnimatedTypography 
          variant="h4" 
          fontWeight="600"
          style={useSpring({
            from: { 
              opacity: 0, 
              transform: 'translateY(-10px)',
              background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
              backgroundSize: '200% 100%',
              backgroundPosition: '0% 50%'
            },
            to: { 
              opacity: animationsVisible ? 1 : 0, 
              transform: animationsVisible ? 'translateY(0px)' : 'translateY(-10px)',
              backgroundPosition: '100% 50%'
            },
            config: { tension: 280, friction: 60 },
          })}
          sx={{ 
            background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            transition: 'all 0.3s ease',
          }}
        >
          Dashboard
        </AnimatedTypography>
        <Tooltip title="Refresh data">
          <AnimatedIconButton 
            onClick={handleRefresh} 
            color="primary"
            style={refreshButtonAnimation}
            sx={{
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'rotate(30deg) scale(1.1)',
                backgroundColor: alpha('#0066B3', 0.1),
              }
            }}
          >
            <RefreshIcon />
          </AnimatedIconButton>
        </Tooltip>
      </AnimatedBox>

      {error && (
        <Alert 
          severity="error" 
          sx={{ 
            mb: 3,
            animation: 'slideIn 0.3s ease-out',
            '@keyframes slideIn': {
              from: { opacity: 0, transform: 'translateY(-10px)' },
              to: { opacity: 1, transform: 'translateY(0)' }
            }
          }}
        >
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* System Status Card */}
        <AnimatedGrid 
          item 
          xs={12} 
          md={6} 
          lg={3}
          style={cardTrail[0]}
        >
          <AnimatedCard
            style={useSpring({
              transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'all 0.3s ease',
              overflow: 'hidden',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme => `0 8px 24px ${alpha(theme.palette.primary.main, 0.15)}`,
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '4px',
                background: theme => `linear-gradient(90deg, ${theme.palette.success.main}, ${theme.palette.success.light})`,
                opacity: 0,
                transition: 'opacity 0.3s ease',
              },
              '&:hover::after': {
                opacity: 1,
              },
            }}
          >
            <CardHeader
              title={
                <AnimatedTypography 
                  variant="h6" 
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
                    delay: 100,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  System Status
                </AnimatedTypography>
              }
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
                  <CircularProgress 
                    size={40} 
                    sx={{
                      color: theme => theme.palette.primary.main,
                      animation: 'pulse 1.5s ease-in-out infinite',
                      '@keyframes pulse': {
                        '0%': { opacity: 0.6 },
                        '50%': { opacity: 1 },
                        '100%': { opacity: 0.6 }
                      }
                    }}
                  />
                </Box>
              ) : (
                <AnimatedBox
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
                    delay: 150,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      mb: 2,
                    }}
                  >
                    <AnimatedBox
                      style={useSpring({
                        transform: animationsVisible ? 'scale(1)' : 'scale(0.8)',
                        rotateZ: animationsVisible ? '0deg' : '-45deg',
                        config: { tension: 200, friction: 12 },
                        delay: 200
                      })}
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
                        boxShadow: '0 4px 14px rgba(0, 0, 0, 0.1)',
                        transition: 'all 0.3s ease',
                        animation: healthStatus?.status === 'healthy' 
                          ? 'pulse 3s ease-in-out infinite'
                          : 'none',
                        '@keyframes pulse': {
                          '0%': { boxShadow: '0 4px 14px rgba(0, 0, 0, 0.1)' },
                          '50%': { boxShadow: '0 4px 20px rgba(52, 199, 89, 0.4)' },
                          '100%': { boxShadow: '0 4px 14px rgba(0, 0, 0, 0.1)' }
                        }
                      }}
                    >
                      {healthStatus?.status === 'healthy' ? (
                        <CheckCircleIcon fontSize="large" />
                      ) : (
                        <ErrorIcon fontSize="large" />
                      )}
                    </AnimatedBox>
                    <Box>
                      <AnimatedTypography 
                        variant="h6" 
                        fontWeight="500"
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(-5px)',
                          delay: 250,
                          config: { tension: 280, friction: 60 }
                        })}
                      >
                        {healthStatus?.status === 'healthy' ? 'All Systems Operational' : 'System Issue Detected'}
                      </AnimatedTypography>
                      <AnimatedTypography 
                        variant="body2" 
                        color="text.secondary"
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(-5px)',
                          delay: 300,
                          config: { tension: 280, friction: 60 }
                        })}
                      >
                        Last updated: {new Date().toLocaleTimeString()}
                      </AnimatedTypography>
                    </Box>
                  </Box>

                  <AnimatedBox 
                    style={useSpring({
                      opacity: animationsVisible ? 1 : 0,
                      transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                      delay: 350,
                      config: { tension: 280, friction: 60 }
                    })}
                    sx={{ mt: 3 }}
                  >
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
                      <AnimatedChip
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                          delay: 400,
                          config: { tension: 280, friction: 60 }
                        })}
                        label="Database"
                        color={healthStatus?.database === 'connected' ? 'success' : 'error'}
                        size="small"
                        icon={<StorageIcon />}
                        sx={{
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                          }
                        }}
                      />
                      <AnimatedChip
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                          delay: 450,
                          config: { tension: 280, friction: 60 }
                        })}
                        label={`GPU ${healthStatus?.gpu_acceleration === 'available' ? 'Active' : 'Inactive'}`}
                        color={healthStatus?.gpu_acceleration === 'available' ? 'success' : 'default'}
                        size="small"
                        icon={<MemoryIcon />}
                        sx={{
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                          }
                        }}
                      />
                      <AnimatedChip
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                          delay: 500,
                          config: { tension: 280, friction: 60 }
                        })}
                        label="API"
                        color="success"
                        size="small"
                        icon={<InfoIcon />}
                        sx={{
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                          }
                        }}
                      />
                    </Box>
                  </AnimatedBox>
                </AnimatedBox>
              )}
            </CardContent>
          </AnimatedCard>
        </AnimatedGrid>

        {/* GPU Stats Card */}
        <AnimatedGrid 
          item 
          xs={12} 
          md={6} 
          lg={3} 
          style={cardTrail[1]}
        >
          <AnimatedCard
            style={useSpring({
              transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'all 0.3s ease',
              overflow: 'hidden',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme => `0 8px 24px ${alpha(theme.palette.primary.main, 0.15)}`,
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '4px',
                background: theme => `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
                opacity: 0,
                transition: 'opacity 0.3s ease',
              },
              '&:hover::after': {
                opacity: 1,
              },
            }}
          >
            <CardHeader
              title={
                <AnimatedTypography 
                  variant="h6"
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
                    delay: 100,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  GPU Resources
                </AnimatedTypography>
              }
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
                  <CircularProgress 
                    size={40}
                    sx={{
                      color: theme => theme.palette.primary.main,
                      animation: 'pulse 1.5s ease-in-out infinite',
                      '@keyframes pulse': {
                        '0%': { opacity: 0.6 },
                        '50%': { opacity: 1 },
                        '100%': { opacity: 0.6 }
                      }
                    }}
                  />
                </Box>
              ) : (
                <AnimatedBox
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
                    delay: 150,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  {gpuInfo?.device_count && gpuInfo.device_count > 0 ? (
                    <>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          mb: 2,
                        }}
                      >
                        <AnimatedBox
                          style={useSpring({
                            transform: animationsVisible ? 'scale(1) rotate(0deg)' : 'scale(0.8) rotate(-45deg)',
                            config: { tension: 200, friction: 12 },
                            delay: 200
                          })}
                        >
                          <MemoryIcon
                            color="primary"
                            sx={{ 
                              fontSize: 48, 
                              mr: 2,
                              filter: 'drop-shadow(0 2px 4px rgba(0, 102, 179, 0.3))',
                              transition: 'all 0.3s ease',
                              animation: 'pulse 3s ease-in-out infinite',
                              '@keyframes pulse': {
                                '0%': { filter: 'drop-shadow(0 2px 4px rgba(0, 102, 179, 0.3))' },
                                '50%': { filter: 'drop-shadow(0 2px 8px rgba(0, 102, 179, 0.5))' },
                                '100%': { filter: 'drop-shadow(0 2px 4px rgba(0, 102, 179, 0.3))' }
                              }
                            }}
                          />
                        </AnimatedBox>
                        <Box>
                          <AnimatedTypography 
                            variant="h6" 
                            fontWeight="500"
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateY(0)' : 'translateY(-5px)',
                              delay: 250,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            {gpuInfo.device_count} GPU{gpuInfo.device_count > 1 ? 's' : ''} Available
                          </AnimatedTypography>
                          <AnimatedTypography 
                            variant="body2" 
                            color="text.secondary"
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateY(0)' : 'translateY(-5px)',
                              delay: 300,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            NVIDIA CUDA Acceleration
                          </AnimatedTypography>
                        </Box>
                      </Box>

                      <AnimatedBox 
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                          delay: 350,
                          config: { tension: 280, friction: 60 }
                        })}
                        sx={{ mt: 2 }}
                      >
                        {gpuInfo.devices.map((device, index) => (
                          <AnimatedBox 
                            key={index} 
                            sx={{ mb: 2 }}
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                              delay: 400 + (index * 100),
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                              <Typography variant="subtitle2">{device.name}</Typography>
                              <Typography variant="subtitle2" color="text.secondary">
                                {metricsAnimation.utilization.value.to(val => 
                                  `${Math.round(val)}% Load`
                                )}
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box sx={{ width: '100%', mr: 1 }}>
                                <Box
                                  sx={{
                                    height: 8,
                                    borderRadius: 1,
                                    background: theme => `linear-gradient(90deg, #19B5FE ${device.memory_used / device.memory_total * 100}%, rgba(0, 0, 0, 0.05) ${device.memory_used / device.memory_total * 100}%)`,
                                    transition: 'all 1.5s ease-out',
                                    position: 'relative',
                                    overflow: 'hidden',
                                    '&::after': {
                                      content: '""',
                                      position: 'absolute',
                                      top: 0,
                                      left: 0,
                                      right: 0,
                                      bottom: 0,
                                      background: 'linear-gradient(90deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0.15) 100%)',
                                      transform: 'translateX(-100%)',
                                      animation: animationsVisible ? 'shimmer 2s infinite' : 'none',
                                      '@keyframes shimmer': {
                                        '100%': { transform: 'translateX(100%)' }
                                      }
                                    }
                                  }}
                                />
                              </Box>
                              <Box>
                                <Typography variant="caption" color="text.secondary">
                                  {metricsAnimation.memory.value.to(val => 
                                    `${Math.round(val / (1024 * 1024))} / ${Math.round(device.memory_total / (1024 * 1024))} MB`
                                  )}
                                </Typography>
                              </Box>
                            </Box>
                          </AnimatedBox>
                        ))}
                      </AnimatedBox>
                    </>
                  ) : (
                    <AnimatedBox 
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        scale: animationsVisible ? 1 : 0.9,
                        delay: 200,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ textAlign: 'center', py: 3 }}
                    >
                      <MemoryIcon 
                        color="disabled" 
                        sx={{ 
                          fontSize: 60, 
                          mb: 2, 
                          opacity: 0.6,
                          transition: 'all 0.3s ease',
                          animation: 'pulse 3s ease-in-out infinite',
                          '@keyframes pulse': {
                            '0%': { opacity: 0.5 },
                            '50%': { opacity: 0.7 },
                            '100%': { opacity: 0.5 }
                          }
                        }}
                      />
                      <Typography variant="body1" gutterBottom>
                        No GPU Acceleration
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Running in CPU-only mode
                      </Typography>
                    </AnimatedBox>
                  )}
                </AnimatedBox>
              )}
            </CardContent>
          </AnimatedCard>
        </AnimatedGrid>

        {/* Performance Card */}
        <AnimatedGrid 
          item 
          xs={12} 
          md={6} 
          lg={6}
          style={cardTrail[2]}
        >
          <AnimatedCard
            style={useSpring({
              transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'all 0.3s ease',
              overflow: 'hidden',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme => `0 8px 24px ${alpha(theme.palette.primary.main, 0.15)}`,
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '4px',
                background: theme => `linear-gradient(90deg, ${theme.palette.info.main}, ${theme.palette.info.light})`,
                opacity: 0,
                transition: 'opacity 0.3s ease',
              },
              '&:hover::after': {
                opacity: 1,
              },
            }}
          >
            <CardHeader
              title={
                <AnimatedTypography 
                  variant="h6"
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
                    delay: 100,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  Query Performance
                </AnimatedTypography>
              }
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              <AnimatedBox
                style={chartsAnimation}
                sx={{ height: 220 }}
              >
                <ResponsiveContainer width="100%" height="100%">
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
                      contentStyle={{ 
                        borderRadius: 8, 
                        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
                        border: 'none',
                        animation: 'fadeIn 0.2s ease-out',
                        '@keyframes fadeIn': {
                          from: { opacity: 0, transform: 'translateY(5px)' },
                          to: { opacity: 1, transform: 'translateY(0)' }
                        }
                      }} 
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
                      animationDuration={1500}
                      animationEasing="ease-out"
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
                      animationDuration={1500}
                      animationEasing="ease-out"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </AnimatedBox>
              <AnimatedBox 
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                  delay: 600,
                  config: { tension: 280, friction: 60 }
                })}
                sx={{ display: 'flex', justifyContent: 'center', gap: 4, mt: 2 }}
              >
                <Box sx={{ textAlign: 'center' }}>
                  <AnimatedTypography 
                    variant="h4" 
                    fontWeight="600" 
                    color="primary.main"
                    style={useSpring({
                      opacity: animationsVisible ? 1 : 0,
                      transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                      delay: 650,
                      config: { tension: 280, friction: 60 }
                    })}
                  >
                    {metricsAnimation.queries.value.to(val => 
                      Math.floor(val).toLocaleString()
                    )}
                  </AnimatedTypography>
                  <Typography variant="body2" color="text.secondary">
                    Queries Last Month
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <AnimatedTypography 
                    variant="h4" 
                    fontWeight="600" 
                    color="secondary.main"
                    style={useSpring({
                      opacity: animationsVisible ? 1 : 0,
                      transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                      delay: 700,
                      config: { tension: 280, friction: 60 }
                    })}
                  >
                    {performanceStats.length > 0 
                      ? `${performanceStats[performanceStats.length - 1].avgTime} ms` 
                      : "0 ms"}
                  </AnimatedTypography>
                  <Typography variant="body2" color="text.secondary">
                    Average Response Time
                  </Typography>
                </Box>
              </AnimatedBox>
            </CardContent>
          </AnimatedCard>
        </AnimatedGrid>

        {/* Recent Queries Card */}
        <AnimatedGrid 
          item 
          xs={12} 
          md={6} 
          lg={6}
          style={cardTrail[3]}
        >
          <AnimatedCard
            style={useSpring({
              transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'all 0.3s ease',
              overflow: 'hidden',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme => `0 8px 24px ${alpha(theme.palette.primary.main, 0.15)}`,
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '4px',
                background: theme => `linear-gradient(90deg, ${theme.palette.secondary.main}, ${theme.palette.secondary.light})`,
                opacity: 0,
                transition: 'opacity 0.3s ease',
              },
              '&:hover::after': {
                opacity: 1,
              },
            }}
          >
            <CardHeader
              title={
                <AnimatedTypography 
                  variant="h6"
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
                    delay: 100,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  Recent Queries
                </AnimatedTypography>
              }
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 1 }}
            />
            <CardContent sx={{ pt: 1, pb: 1, flexGrow: 1 }}>
              <AnimatedBox
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                  delay: 200,
                  config: { tension: 280, friction: 60 }
                })}
              >
                <List disablePadding>
                  {recentQueries.map((query, index) => (
                    <React.Fragment key={query.id}>
                      <AnimatedBox
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                          delay: 300 + (index * 100),
                          config: { tension: 280, friction: 60 }
                        })}
                      >
                        <ListItem 
                          disablePadding 
                          sx={{ 
                            py: 1,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                              backgroundColor: 'rgba(0, 102, 179, 0.05)',
                              transform: 'translateX(2px)',
                            }
                          }}
                        >
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
                                  <AnimatedChip
                                    style={useSpring({
                                      opacity: animationsVisible ? 1 : 0,
                                      transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                                      delay: 350 + (index * 100),
                                      config: { tension: 280, friction: 60 }
                                    })}
                                    label={`${query.results_count} results`}
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      transition: 'all 0.3s ease',
                                      '&:hover': {
                                        transform: 'translateY(-1px)',
                                        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.1)',
                                      }
                                    }}
                                  />
                                  <AnimatedChip
                                    style={useSpring({
                                      opacity: animationsVisible ? 1 : 0,
                                      transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                                      delay: 400 + (index * 100),
                                      config: { tension: 280, friction: 60 }
                                    })}
                                    label={`${query.execution_time} ms`}
                                    size="small"
                                    color="primary"
                                    variant="outlined"
                                    sx={{
                                      transition: 'all 0.3s ease',
                                      '&:hover': {
                                        transform: 'translateY(-1px)',
                                        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.1)',
                                      }
                                    }}
                                  />
                                </Box>
                              </Box>
                            }
                          />
                        </ListItem>
                        <Divider />
                      </AnimatedBox>
                    </React.Fragment>
                  ))}
                </List>
                <AnimatedBox 
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(15px)',
                    delay: 500,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}
                >
                  <AnimatedButton
                    style={useSpring({
                      opacity: animationsVisible ? 1 : 0,
                      transform: animationsVisible ? 'scale(1)' : 'scale(0.9)',
                      delay: 550,
                      config: { tension: 280, friction: 60 }
                    })}
                    variant="outlined"
                    color="primary"
                    size="small"
                    startIcon={<QueryStatsIcon />}
                    sx={{ 
                      borderRadius: 4,
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: '0 4px 8px rgba(0, 102, 179, 0.15)',
                      }
                    }}
                  >
                    View All Queries
                  </AnimatedButton>
                </AnimatedBox>
              </AnimatedBox>
            </CardContent>
          </AnimatedCard>
        </AnimatedGrid>

        {/* GPU vs CPU Performance Card */}
        <AnimatedGrid 
          item 
          xs={12} 
          md={6} 
          lg={6}
          style={tablesAnimation}
        >
          <AnimatedCard
            style={useSpring({
              transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'all 0.3s ease',
              overflow: 'hidden',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme => `0 8px 24px ${alpha(theme.palette.primary.main, 0.15)}`,
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '4px',
                background: theme => `linear-gradient(90deg, ${theme.palette.info.dark}, ${theme.palette.info.light})`,
                opacity: 0,
                transition: 'opacity 0.3s ease',
              },
              '&:hover::after': {
                opacity: 1,
              },
            }}
          >
            <CardHeader
              title={
                <AnimatedTypography 
                  variant="h6"
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
                    delay: 100,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  GPU vs CPU Performance
                </AnimatedTypography>
              }
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              <AnimatedBox
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'scale(1)' : 'scale(0.9)',
                  delay: 200,
                  config: { tension: 280, friction: 60 }
                })}
              >
                <ResponsiveContainer width="100%" height={270}>
                  <BarChart
                    data={performanceComparison}
                    margin={{
                      top: 20,
                      right: 30,
                      left: 20,
                      bottom: 20,
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.05)" />
                    <XAxis 
                      dataKey="name" 
                      stroke="#666666" 
                      tick={{ 
                        opacity: animationsVisible ? 1 : 0,
                        animation: 'fadeIn 0.5s ease-out',
                        '@keyframes fadeIn': {
                          from: { opacity: 0 },
                          to: { opacity: 1 }
                        }
                      }}
                    />
                    <YAxis 
                      stroke="#666666" 
                      label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft', offset: -5 }}
                      tick={{ 
                        opacity: animationsVisible ? 1 : 0,
                        animation: 'fadeIn 0.5s ease-out',
                        '@keyframes fadeIn': {
                          from: { opacity: 0 },
                          to: { opacity: 1 }
                        }
                      }}
                    />
                    <RechartsTooltip 
                      contentStyle={{ 
                        borderRadius: 8, 
                        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
                        border: 'none',
                        animation: 'fadeIn 0.2s ease-out',
                        '@keyframes fadeIn': {
                          from: { opacity: 0, transform: 'translateY(5px)' },
                          to: { opacity: 1, transform: 'translateY(0)' }
                        }
                      }} 
                    />
                    <Legend 
                      verticalAlign="top" 
                      height={40} 
                      wrapperStyle={{
                        opacity: animationsVisible ? 1 : 0,
                        transition: 'opacity 0.5s ease-out 0.3s',
                      }}
                    />
                    <Bar 
                      dataKey="CPU" 
                      name="CPU" 
                      fill="#A0A0A0" 
                      radius={[4, 4, 0, 0]} 
                      barSize={25}
                      animationDuration={1500}
                      animationBegin={animationsVisible ? 200 : 0}
                      animationEasing="ease-out"
                    />
                    <Bar 
                      dataKey="GPU" 
                      name="CUDA GPU" 
                      fill="#19B5FE" 
                      radius={[4, 4, 0, 0]} 
                      barSize={25}
                      animationDuration={1500}
                      animationBegin={animationsVisible ? 400 : 0}
                      animationEasing="ease-out"
                    />
                    <Bar 
                      dataKey="TensorRT" 
                      name="TensorRT" 
                      fill="#0066B3" 
                      radius={[4, 4, 0, 0]} 
                      barSize={25}
                      animationDuration={1500}
                      animationBegin={animationsVisible ? 600 : 0}
                      animationEasing="ease-out"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </AnimatedBox>
              <AnimatedBox 
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(15px)',
                  delay: 800,
                  config: { tension: 280, friction: 60 }
                })}
                sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}
              >
                <AnimatedButton
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'scale(1)' : 'scale(0.9)',
                    delay: 850,
                    config: { tension: 280, friction: 60 }
                  })}
                  variant="outlined"
                  color="primary"
                  size="small"
                  startIcon={<SpeedIcon />}
                  sx={{ 
                    borderRadius: 4,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 4px 8px rgba(0, 102, 179, 0.15)',
                    }
                  }}
                >
                  Run Benchmark
                </AnimatedButton>
              </AnimatedBox>
            </CardContent>
          </AnimatedCard>
        </AnimatedGrid>

        {/* System Info Card */}
        <AnimatedGrid 
          item 
          xs={12} 
          md={6} 
          lg={6}
          style={tablesAnimation}
        >
          <AnimatedCard
            style={useSpring({
              transform: animationsVisible ? 'scale(1)' : 'scale(0.95)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'all 0.3s ease',
              overflow: 'hidden',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme => `0 8px 24px ${alpha(theme.palette.primary.main, 0.15)}`,
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '4px',
                background: theme => `linear-gradient(90deg, ${theme.palette.success.main}, ${theme.palette.success.light})`,
                opacity: 0,
                transition: 'opacity 0.3s ease',
              },
              '&:hover::after': {
                opacity: 1,
              },
            }}
          >
            <CardHeader
              title={
                <AnimatedTypography 
                  variant="h6"
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
                    delay: 100,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  System Information
                </AnimatedTypography>
              }
              titleTypographyProps={{ variant: 'h6' }}
              sx={{ pb: 0 }}
            />
            <CardContent sx={{ pt: 2, flexGrow: 1 }}>
              <AnimatedBox
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                  delay: 200,
                  config: { tension: 280, friction: 60 }
                })}
              >
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <AnimatedBox
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(15px)',
                        delay: 300,
                        config: { tension: 280, friction: 60 }
                      })}
                    >
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          bgcolor: 'background.default',
                          height: '100%',
                          borderRadius: 2,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
                            bgcolor: 'background.paper',
                          }
                        }}
                      >
                        <AnimatedTypography 
                          variant="subtitle2" 
                          gutterBottom
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                            delay: 350,
                            config: { tension: 280, friction: 60 }
                          })}
                          sx={{
                            fontWeight: 600,
                            color: 'primary.main',
                          }}
                        >
                          Database
                        </AnimatedTypography>
                        <AnimatedBox 
                          sx={{ mt: 1 }}
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            delay: 400,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          <Typography variant="body2" color="text.secondary">
                            Connection:
                          </Typography>
                          <AnimatedTypography 
                            variant="body1" 
                            gutterBottom
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                              delay: 450,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            SAP HANA Cloud
                          </AnimatedTypography>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            Version:
                          </Typography>
                          <AnimatedTypography 
                            variant="body1" 
                            gutterBottom
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                              delay: 500,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            2023.25.0
                          </AnimatedTypography>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            Status:
                          </Typography>
                          <AnimatedBox 
                            sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                              delay: 550,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            <AnimatedBox
                              style={useSpring({
                                transform: animationsVisible ? 'scale(1) rotate(0deg)' : 'scale(0.5) rotate(-90deg)',
                                config: { tension: 200, friction: 12 },
                                delay: 600
                              })}
                            >
                              <CheckCircleIcon 
                                color="success" 
                                fontSize="small" 
                                sx={{
                                  animation: 'pulse 3s ease-in-out infinite',
                                  '@keyframes pulse': {
                                    '0%': { opacity: 0.8 },
                                    '50%': { opacity: 1 },
                                    '100%': { opacity: 0.8 }
                                  }
                                }}
                              />
                            </AnimatedBox>
                            <Typography variant="body1">
                              Connected
                            </Typography>
                          </AnimatedBox>
                        </AnimatedBox>
                      </Paper>
                    </AnimatedBox>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <AnimatedBox
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(15px)',
                        delay: 400,
                        config: { tension: 280, friction: 60 }
                      })}
                    >
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          bgcolor: 'background.default',
                          height: '100%',
                          borderRadius: 2,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
                            bgcolor: 'background.paper',
                          }
                        }}
                      >
                        <AnimatedTypography 
                          variant="subtitle2" 
                          gutterBottom
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                            delay: 450,
                            config: { tension: 280, friction: 60 }
                          })}
                          sx={{
                            fontWeight: 600,
                            color: 'primary.main',
                          }}
                        >
                          API
                        </AnimatedTypography>
                        <AnimatedBox 
                          sx={{ mt: 1 }}
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            delay: 500,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          <Typography variant="body2" color="text.secondary">
                            Version:
                          </Typography>
                          <AnimatedTypography 
                            variant="body1" 
                            gutterBottom
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                              delay: 550,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            1.0.0
                          </AnimatedTypography>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            Embedding Model:
                          </Typography>
                          <AnimatedTypography 
                            variant="body1" 
                            gutterBottom
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                              delay: 600,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            {gpuInfo?.device_count && gpuInfo.device_count > 0
                              ? "all-MiniLM-L6-v2 (GPU)"
                              : "SAP_NEB.20240715 (CPU)"}
                          </AnimatedTypography>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            Vector Dimension:
                          </Typography>
                          <AnimatedTypography 
                            variant="body1"
                            style={useSpring({
                              opacity: animationsVisible ? 1 : 0,
                              transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                              delay: 650,
                              config: { tension: 280, friction: 60 }
                            })}
                          >
                            384
                          </AnimatedTypography>
                        </AnimatedBox>
                      </Paper>
                    </AnimatedBox>
                  </Grid>
                </Grid>
              </AnimatedBox>
            </CardContent>
          </AnimatedCard>
        </AnimatedGrid>
      </Grid>
    </Box>
  );
};

export default Dashboard;