import React from 'react';
import { Grid, Box, Typography, CircularProgress } from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Info as InfoIcon,
  QueryStats as QueryStatsIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { useSpring, animated } from '@react-spring/web';
import { 
  EnhancedDashboardCard, 
  EnhancedDashboardCardGrid, 
  EnhancedBox, 
  EnhancedButton, 
  EnhancedIconButton, 
  EnhancedTypography, 
  EnhancedChip,
  EnhancedAlert
} from '../enhanced';
import { soundEffects } from '../../utils/soundEffects';
import { useAnimationContext } from '../../context/AnimationContext';

// Animated MUI components
const AnimatedTypography = animated(Typography);

/**
 * Example of Dashboard with Enhanced Dashboard Cards using batch animations
 * This is an optimized version that applies the new batch animation system
 */
const EnhancedDashboard: React.FC = () => {
  const { animationsEnabled } = useAnimationContext();
  const [animationsVisible, setAnimationsVisible] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  
  // System status data (mockup)
  const healthStatus = {
    status: 'healthy',
    database: 'connected',
    gpu_acceleration: 'available',
    gpu_count: 2
  };
  
  // GPU data (mockup)
  const gpuInfo = {
    device_count: 2,
    devices: [
      {
        name: 'NVIDIA T4',
        memory_total: 16 * 1024 * 1024 * 1024,
        memory_used: 4 * 1024 * 1024 * 1024,
        memory_free: 12 * 1024 * 1024 * 1024,
        utilization: 25
      },
      {
        name: 'NVIDIA T4',
        memory_total: 16 * 1024 * 1024 * 1024,
        memory_used: 6 * 1024 * 1024 * 1024,
        memory_free: 10 * 1024 * 1024 * 1024,
        utilization: 38
      }
    ]
  };
  
  // Recent queries data (mockup)
  const recentQueries = [
    {
      id: 1,
      query: "What is the revenue forecast for Q3 2023?",
      timestamp: "2023-08-15T14:32:45",
      results_count: 5,
      execution_time: 225
    },
    {
      id: 2,
      query: "Show me the sales performance by region",
      timestamp: "2023-08-15T10:15:20",
      results_count: 12,
      execution_time: 310
    },
    {
      id: 3,
      query: "Compare expenses between departments",
      timestamp: "2023-08-14T16:48:33",
      results_count: 8,
      execution_time: 256
    }
  ];
  
  // Header animation
  const headerAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { 
      opacity: animationsVisible ? 1 : 0, 
      transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' 
    },
    config: { tension: 280, friction: 60 }
  });
  
  // Refresh button animation
  const refreshButtonAnimation = useSpring({
    from: { transform: 'rotate(0deg)' },
    to: { transform: loading ? 'rotate(360deg)' : 'rotate(0deg)' },
    config: { duration: loading ? 1000 : 0 },
    loop: loading,
  });
  
  // Handle refresh
  const handleRefresh = () => {
    setLoading(true);
    
    // Simulate data loading
    setTimeout(() => {
      setLoading(false);
      if (animationsEnabled) {
        soundEffects.success();
      }
    }, 1500);
  };
  
  // Make animations visible after initial render
  React.useEffect(() => {
    setTimeout(() => setAnimationsVisible(true), 300);
  }, []);
  
  // Function to render a card (to demonstrate the usage of EnhancedDashboardCard)
  const renderSystemStatusCard = () => {
    return (
      <Grid item xs={12} md={6} lg={3}>
        <EnhancedDashboardCard
          title="System Status"
          gradientColors={['success.main', 'success.light']}
        >
          {loading ? (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                py: 4
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
                  <EnhancedChip
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
                  <EnhancedChip
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
                  <EnhancedChip
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
              </Box>
            </Box>
          )}
        </EnhancedDashboardCard>
      </Grid>
    );
  };
  
  const renderGPUCard = () => {
    return (
      <Grid item xs={12} md={6} lg={3}>
        <EnhancedDashboardCard
          title="GPU Resources"
          gradientColors={['primary.main', 'primary.light']}
        >
          {loading ? (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                py: 4
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
                              animation: 'shimmer 2s infinite',
                              '@keyframes shimmer': {
                                '100%': { transform: 'translateX(100%)' }
                              }
                            }
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
            </Box>
          )}
        </EnhancedDashboardCard>
      </Grid>
    );
  };
  
  const renderPerformanceCard = () => {
    return (
      <Grid item xs={12} md={6} lg={6}>
        <EnhancedDashboardCard
          title="Performance"
          gradientColors={['info.main', 'info.light']}
        >
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 220 }}>
            <Box sx={{ textAlign: 'center' }}>
              <SpeedIcon sx={{ fontSize: 60, color: 'info.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Performance data visualization would appear here
              </Typography>
              <EnhancedButton
                variant="outlined"
                size="small"
                color="primary"
                startIcon={<QueryStatsIcon />}
              >
                View Performance Details
              </EnhancedButton>
            </Box>
          </Box>
        </EnhancedDashboardCard>
      </Grid>
    );
  };
  
  const renderRecentQueriesCard = () => {
    return (
      <Grid item xs={12} md={6} lg={6}>
        <EnhancedDashboardCard
          title="Recent Queries"
          gradientColors={['secondary.main', 'secondary.light']}
        >
          <Box sx={{ height: 220, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {recentQueries.length > 0 ? (
              <Box>
                {recentQueries.map((query, index) => (
                  <Box 
                    key={query.id}
                    sx={{ 
                      mb: 2, 
                      p: 1.5, 
                      borderRadius: 1,
                      bgcolor: 'background.paper',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateX(5px)',
                        bgcolor: 'action.hover'
                      }
                    }}
                  >
                    <Typography variant="body2" noWrap>{query.query}</Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(query.timestamp).toLocaleString()}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <EnhancedChip
                          label={`${query.results_count} results`}
                          size="small"
                          variant="outlined"
                          sx={{ height: 20, fontSize: '0.7rem' }}
                        />
                        <EnhancedChip
                          label={`${query.execution_time} ms`}
                          size="small"
                          color="primary"
                          variant="outlined"
                          sx={{ height: 20, fontSize: '0.7rem' }}
                        />
                      </Box>
                    </Box>
                  </Box>
                ))}
              </Box>
            ) : (
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="body1" gutterBottom>
                  No recent queries
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Recent query history will appear here
                </Typography>
              </Box>
            )}
          </Box>
        </EnhancedDashboardCard>
      </Grid>
    );
  };
  
  return (
    <EnhancedBox sx={{ p: 3 }}>
      <EnhancedBox
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
          Enhanced Dashboard
        </AnimatedTypography>
        
        <EnhancedIconButton 
          onClick={handleRefresh} 
          color="primary"
          style={refreshButtonAnimation}
        >
          <RefreshIcon />
        </EnhancedIconButton>
      </EnhancedBox>
      
      {/* Error message example */}
      <EnhancedAlert 
        severity="info" 
        sx={{ mb: 3 }}
      >
        This is an example of enhanced dashboard cards with batch animations
      </EnhancedAlert>
      
      {/* Cards Grid */}
      <Grid container spacing={3}>
        {/* Use the EnhancedDashboardCardGrid for optimal batch animations */}
        <EnhancedDashboardCardGrid
          animationsVisible={animationsVisible}
          baseDelay={0}
        >
          {renderSystemStatusCard()}
          {renderGPUCard()}
          {renderPerformanceCard()}
          {renderRecentQueriesCard()}
        </EnhancedDashboardCardGrid>
      </Grid>
    </EnhancedBox>
  );
};

export default EnhancedDashboard;