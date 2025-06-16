import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Grid,
  Divider,
  Alert,
  Tabs,
  Tab,
  Paper,
  Slider,
  InputAdornment,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  IconButton,
  Tooltip,
  Snackbar,
  CircularProgress,
  Container,
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Security as SecurityIcon,
  Info as InfoIcon,
  DataObject as DataObjectIcon,
  Check as CheckIcon,
} from '@mui/icons-material';
import { useSpring, useTrail, useChain, animated, useSpringRef, config } from 'react-spring';
import axios from 'axios';
import ExperienceManager from '../components/ExperienceManager';

// Animated components
const AnimatedBox = animated(Box);
const AnimatedCard = animated(Card);
const AnimatedPaper = animated(Paper);
const AnimatedTypography = animated(Typography);
const AnimatedGrid = animated(Grid);
const AnimatedContainer = animated(Container);
const AnimatedButton = animated(Button);
const AnimatedAlert = animated(Alert);
const AnimatedDivider = animated(Divider);

interface Config {
  database: {
    host: string;
    port: number;
    user: string;
    password: string;
    schema: string;
    table_name: string;
  };
  embeddings: {
    model: string;
    internal_model_id: string;
    use_internal_embeddings: boolean;
    dimension: number;
  };
  gpu: {
    enabled: boolean;
    device: string;
    batch_size: number;
    use_tensorrt: boolean;
    tensorrt_precision: string;
    use_multi_gpu: boolean;
  };
  api: {
    host: string;
    port: number;
    log_level: string;
    enable_cache: boolean;
    cache_ttl: number;
  };
}

const defaultConfig: Config = {
  database: {
    host: 'your-hana-host.example.com',
    port: 443,
    user: 'DBADMIN',
    password: '********',
    schema: 'SYSTEM',
    table_name: 'LANGCHAIN_VECTORS',
  },
  embeddings: {
    model: 'all-MiniLM-L6-v2',
    internal_model_id: 'SAP_NEB.20240715',
    use_internal_embeddings: true,
    dimension: 384,
  },
  gpu: {
    enabled: true,
    device: 'cuda',
    batch_size: 32,
    use_tensorrt: true,
    tensorrt_precision: 'fp16',
    use_multi_gpu: true,
  },
  api: {
    host: '0.0.0.0',
    port: 8000,
    log_level: 'INFO',
    enable_cache: true,
    cache_ttl: 3600,
  },
};

const embeddingModels = [
  { name: 'all-MiniLM-L6-v2', dimension: 384 },
  { name: 'all-mpnet-base-v2', dimension: 768 },
  { name: 'paraphrase-multilingual-MiniLM-L12-v2', dimension: 384 },
  { name: 'bge-base-en-v1.5', dimension: 768 },
  { name: 'bge-small-en-v1.5', dimension: 384 },
];

const Settings: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [config, setConfig] = useState<Config>(defaultConfig);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'untested' | 'success' | 'error'>('untested');
  const [animationsVisible, setAnimationsVisible] = useState<boolean>(false);
  
  // Animation spring refs for sequence chaining
  const headerSpringRef = useSpringRef();
  const tabsSpringRef = useSpringRef();
  const cardSpringRef = useSpringRef();
  const formSpringRef = useSpringRef();
  const buttonSpringRef = useSpringRef();
  
  // Animation springs
  const headerAnimation = useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const tabsAnimation = useSpring({
    ref: tabsSpringRef,
    from: { opacity: 0, transform: 'translateY(-10px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const cardAnimation = useSpring({
    ref: cardSpringRef,
    from: { opacity: 0, transform: 'scale(0.95)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'scale(1)' : 'scale(0.95)' },
    config: { tension: 280, friction: 60 }
  });
  
  const formAnimation = useSpring({
    ref: formSpringRef,
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const buttonAnimation = useSpring({
    ref: buttonSpringRef,
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Chain animations sequence
  useChain(
    animationsVisible 
      ? [headerSpringRef, tabsSpringRef, cardSpringRef, formSpringRef, buttonSpringRef] 
      : [buttonSpringRef, formSpringRef, cardSpringRef, tabsSpringRef, headerSpringRef],
    animationsVisible 
      ? [0, 0.1, 0.2, 0.3, 0.4] 
      : [0, 0.1, 0.2, 0.3, 0.4]
  );

  // Fetch the actual config from the API
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        // Make an actual API call to get the config
        const response = await axios.get('/api/config');
        setConfig(response.data);
      } catch (err) {
        setError('Failed to load configuration');
        console.error('Error loading config:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchConfig();
  }, []);
  
  // Trigger animations on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleConfigChange = (section: keyof Config, field: string, value: any) => {
    setConfig({
      ...config,
      [section]: {
        ...config[section],
        [field]: value,
      },
    });
  };

  const handleEmbeddingModelChange = (modelName: string) => {
    const model = embeddingModels.find(m => m.name === modelName);
    if (model) {
      setConfig({
        ...config,
        embeddings: {
          ...config.embeddings,
          model: modelName,
          dimension: model.dimension,
        },
      });
    }
  };

  const handleSaveConfig = async () => {
    try {
      setLoading(true);
      // Make an actual API call to save the config
      await axios.post('/api/config', config);
      setSaved(true);
      setTimeout(() => setSaved(false), 5000);
    } catch (err) {
      setError('Failed to save configuration');
      console.error('Error saving config:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async () => {
    try {
      setTestingConnection(true);
      setConnectionStatus('untested');
      
      // Make an actual API call to test the connection
      const response = await axios.post('/api/test-connection', config.database);
      
      setConnectionStatus(response.data.success ? 'success' : 'error');
    } catch (err) {
      setConnectionStatus('error');
      console.error('Error testing connection:', err);
    } finally {
      setTestingConnection(false);
    }
  };

  const handleResetConfig = () => {
    setConfig(defaultConfig);
  };

  return (
    <AnimatedBox>
      <ExperienceManager 
        currentComponent="settings"
        onOpenAdvanced={() => {}}
      />
      
      {/* Header Section */}
      <AnimatedBox sx={{ mb: 3 }} style={headerAnimation}>
        <AnimatedTypography 
          variant="h4" 
          fontWeight="600"
          sx={{
            background: `linear-gradient(90deg, #0066B3, #2a8fd8)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            letterSpacing: '0.02em',
          }}
        >
          Settings
        </AnimatedTypography>
        <AnimatedTypography 
          variant="body1" 
          color="text.secondary"
          sx={{
            maxWidth: '800px',
            lineHeight: 1.6,
          }}
        >
          Configure your SAP HANA Cloud connection, embedding models, and GPU acceleration
        </AnimatedTypography>
      </AnimatedBox>
      
      {/* Tabs Navigation */}
      <animated.div style={tabsAnimation}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          sx={{ 
            borderBottom: 1, 
            borderColor: 'divider', 
            mb: 3,
            '& .MuiTab-root': {
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                color: '#0066B3',
              },
            },
            '& .Mui-selected': {
              fontWeight: 600,
              color: '#0066B3',
            },
            '& .MuiTabs-indicator': {
              height: 3,
              borderRadius: '3px 3px 0 0',
              background: 'linear-gradient(90deg, #0066B3, #2a8fd8)',
            }
          }}
        >
          <Tab 
            icon={<StorageIcon sx={{ filter: tabValue === 0 ? 'drop-shadow(0 0 4px rgba(0, 102, 179, 0.5))' : 'none' }} />} 
            label="Database" 
            iconPosition="start" 
          />
          <Tab 
            icon={<DataObjectIcon sx={{ filter: tabValue === 1 ? 'drop-shadow(0 0 4px rgba(0, 102, 179, 0.5))' : 'none' }} />} 
            label="Embeddings" 
            iconPosition="start" 
          />
          <Tab 
            icon={<MemoryIcon sx={{ filter: tabValue === 2 ? 'drop-shadow(0 0 4px rgba(0, 102, 179, 0.5))' : 'none' }} />} 
            label="GPU" 
            iconPosition="start" 
          />
          <Tab 
            icon={<SettingsIcon sx={{ filter: tabValue === 3 ? 'drop-shadow(0 0 4px rgba(0, 102, 179, 0.5))' : 'none' }} />} 
            label="API" 
            iconPosition="start" 
          />
        </Tabs>
      </animated.div>
      
      {/* Main Card */}
      <AnimatedCard 
        style={cardAnimation}
        sx={{ 
          borderRadius: '12px',
          boxShadow: '0 6px 20px rgba(0,0,0,0.05)',
          transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
          border: '1px solid rgba(0, 102, 179, 0.1)',
          '&:hover': {
            boxShadow: '0 12px 28px rgba(0,0,0,0.1), 0 8px 10px rgba(0,0,0,0.08)',
          }
        }}
      >
        <CardContent>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
              <CircularProgress 
                size={60} 
                sx={{
                  color: '#0066B3',
                  filter: 'drop-shadow(0 0 4px rgba(0, 102, 179, 0.3))'
                }}
              />
            </Box>
          ) : (
            <AnimatedBox style={formAnimation}>
              {/* Database Settings */}
              {tabValue === 0 && (
                <Box>
                  <AnimatedTypography 
                    variant="h6" 
                    gutterBottom
                    sx={{
                      fontWeight: 600,
                      background: `linear-gradient(90deg, #0066B3, #2a8fd8)`,
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                      textFillColor: 'transparent',
                    }}
                  >
                    SAP HANA Cloud Database Connection
                  </AnimatedTypography>
                  <AnimatedTypography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Configure the connection to your SAP HANA Cloud database instance
                  </AnimatedTypography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Host"
                        fullWidth
                        margin="normal"
                        value={config.database.host}
                        onChange={(e) => handleConfigChange('database', 'host', e.target.value)}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <StorageIcon color="action" fontSize="small" />
                            </InputAdornment>
                          ),
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Port"
                        fullWidth
                        margin="normal"
                        type="number"
                        value={config.database.port}
                        onChange={(e) => handleConfigChange('database', 'port', Number(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Username"
                        fullWidth
                        margin="normal"
                        value={config.database.user}
                        onChange={(e) => handleConfigChange('database', 'user', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Password"
                        fullWidth
                        margin="normal"
                        type={showPassword ? "text" : "password"}
                        value={config.database.password}
                        onChange={(e) => handleConfigChange('database', 'password', e.target.value)}
                        InputProps={{
                          endAdornment: (
                            <InputAdornment position="end">
                              <Tooltip title={showPassword ? "Hide password" : "Show password"}>
                                <IconButton
                                  onClick={() => setShowPassword(!showPassword)}
                                  edge="end"
                                >
                                  <SecurityIcon />
                                </IconButton>
                              </Tooltip>
                            </InputAdornment>
                          ),
                        }}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Schema"
                        fullWidth
                        margin="normal"
                        value={config.database.schema}
                        onChange={(e) => handleConfigChange('database', 'schema', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Vector Table Name"
                        fullWidth
                        margin="normal"
                        value={config.database.table_name}
                        onChange={(e) => handleConfigChange('database', 'table_name', e.target.value)}
                      />
                    </Grid>
                  </Grid>

                  <Box sx={{ mt: 4, display: 'flex', gap: 2 }}>
                    <Button
                      variant="outlined"
                      color="primary"
                      startIcon={testingConnection ? <CircularProgress size={20} /> : <InfoIcon />}
                      onClick={handleTestConnection}
                      disabled={testingConnection}
                    >
                      Test Connection
                    </Button>
                    
                    {connectionStatus === 'success' && (
                      <Alert severity="success" sx={{ flex: 1 }}>
                        Connection successful! Database is accessible.
                      </Alert>
                    )}
                    
                    {connectionStatus === 'error' && (
                      <Alert severity="error" sx={{ flex: 1 }}>
                        Connection failed. Please check your database settings.
                      </Alert>
                    )}
                  </Box>
                </Box>
              )}

              {/* Embeddings Settings */}
              {tabValue === 1 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Embedding Model Configuration
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Configure the embedding model settings for vector similarity search
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel id="model-label">Embedding Model</InputLabel>
                        <Select
                          labelId="model-label"
                          value={config.embeddings.model}
                          label="Embedding Model"
                          onChange={(e) => handleEmbeddingModelChange(e.target.value)}
                        >
                          {embeddingModels.map((model) => (
                            <MenuItem key={model.name} value={model.name}>
                              {model.name} ({model.dimension}d)
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Vector Dimension"
                        fullWidth
                        margin="normal"
                        type="number"
                        value={config.embeddings.dimension}
                        disabled
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="SAP HANA Internal Model ID"
                        fullWidth
                        margin="normal"
                        value={config.embeddings.internal_model_id}
                        onChange={(e) => handleConfigChange('embeddings', 'internal_model_id', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={config.embeddings.use_internal_embeddings}
                            onChange={(e) => handleConfigChange('embeddings', 'use_internal_embeddings', e.target.checked)}
                          />
                        }
                        label="Use SAP HANA Internal Embeddings"
                        sx={{ mt: 3 }}
                      />
                    </Grid>
                  </Grid>

                  <Paper variant="outlined" sx={{ p: 2, mt: 3, borderRadius: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      <InfoIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
                      Hybrid Embedding Mode
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      When enabled, the system will use both external GPU-accelerated embeddings and SAP HANA 
                      internal embeddings for optimal performance. Queries will be processed using GPU 
                      acceleration, while indexed documents can use internal embeddings.
                    </Typography>
                  </Paper>
                </Box>
              )}

              {/* GPU Settings */}
              {tabValue === 2 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    GPU Acceleration Settings
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Configure NVIDIA GPU acceleration for embedding generation
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={config.gpu.enabled}
                            onChange={(e) => handleConfigChange('gpu', 'enabled', e.target.checked)}
                          />
                        }
                        label="Enable GPU Acceleration"
                        sx={{ mb: 2 }}
                      />
                      
                      <FormControl fullWidth margin="normal" disabled={!config.gpu.enabled}>
                        <InputLabel id="device-label">GPU Device</InputLabel>
                        <Select
                          labelId="device-label"
                          value={config.gpu.device}
                          label="GPU Device"
                          onChange={(e) => handleConfigChange('gpu', 'device', e.target.value)}
                        >
                          <MenuItem value="cuda">CUDA (NVIDIA GPU)</MenuItem>
                          <MenuItem value="cpu">CPU (Fallback)</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <Typography gutterBottom variant="body2" sx={{ mt: 3 }}>
                        Batch Size: {config.gpu.batch_size}
                      </Typography>
                      <Slider
                        value={config.gpu.batch_size}
                        onChange={(_, value) => handleConfigChange('gpu', 'batch_size', value)}
                        min={1}
                        max={128}
                        step={1}
                        valueLabelDisplay="auto"
                        disabled={!config.gpu.enabled}
                        marks={[
                          { value: 1, label: '1' },
                          { value: 32, label: '32' },
                          { value: 64, label: '64' },
                          { value: 128, label: '128' },
                        ]}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={config.gpu.use_tensorrt}
                            onChange={(e) => handleConfigChange('gpu', 'use_tensorrt', e.target.checked)}
                            disabled={!config.gpu.enabled}
                          />
                        }
                        label="Use TensorRT Optimization"
                        sx={{ mb: 2 }}
                      />
                      
                      <FormControl 
                        fullWidth 
                        margin="normal" 
                        disabled={!config.gpu.enabled || !config.gpu.use_tensorrt}
                      >
                        <InputLabel id="precision-label">TensorRT Precision</InputLabel>
                        <Select
                          labelId="precision-label"
                          value={config.gpu.tensorrt_precision}
                          label="TensorRT Precision"
                          onChange={(e) => handleConfigChange('gpu', 'tensorrt_precision', e.target.value)}
                        >
                          <MenuItem value="fp32">FP32 (32-bit)</MenuItem>
                          <MenuItem value="fp16">FP16 (16-bit)</MenuItem>
                          <MenuItem value="int8">INT8 (8-bit)</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <FormControlLabel
                        control={
                          <Switch
                            checked={config.gpu.use_multi_gpu}
                            onChange={(e) => handleConfigChange('gpu', 'use_multi_gpu', e.target.checked)}
                            disabled={!config.gpu.enabled}
                          />
                        }
                        label="Enable Multi-GPU Support"
                        sx={{ mt: 3 }}
                      />
                    </Grid>
                  </Grid>

                  <Paper variant="outlined" sx={{ p: 2, mt: 3, borderRadius: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      <InfoIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
                      GPU Performance Recommendations
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      For optimal performance, use TensorRT with FP16 precision and a batch size of 32-64.
                      Multi-GPU support distributes workloads across multiple GPUs if available. If running on
                      consumer GPUs, consider lower batch sizes to avoid memory issues.
                    </Typography>
                  </Paper>
                </Box>
              )}

              {/* API Settings */}
              {tabValue === 3 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    API Configuration
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Configure the FastAPI server settings
                  </Typography>

                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Host"
                        fullWidth
                        margin="normal"
                        value={config.api.host}
                        onChange={(e) => handleConfigChange('api', 'host', e.target.value)}
                        helperText="Use 0.0.0.0 to listen on all interfaces"
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Port"
                        fullWidth
                        margin="normal"
                        type="number"
                        value={config.api.port}
                        onChange={(e) => handleConfigChange('api', 'port', Number(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel id="log-level-label">Log Level</InputLabel>
                        <Select
                          labelId="log-level-label"
                          value={config.api.log_level}
                          label="Log Level"
                          onChange={(e) => handleConfigChange('api', 'log_level', e.target.value)}
                        >
                          <MenuItem value="DEBUG">DEBUG</MenuItem>
                          <MenuItem value="INFO">INFO</MenuItem>
                          <MenuItem value="WARNING">WARNING</MenuItem>
                          <MenuItem value="ERROR">ERROR</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={config.api.enable_cache}
                            onChange={(e) => handleConfigChange('api', 'enable_cache', e.target.checked)}
                          />
                        }
                        label="Enable Result Caching"
                        sx={{ mt: 3 }}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Cache TTL (seconds)"
                        fullWidth
                        margin="normal"
                        type="number"
                        value={config.api.cache_ttl}
                        onChange={(e) => handleConfigChange('api', 'cache_ttl', Number(e.target.value))}
                        disabled={!config.api.enable_cache}
                      />
                    </Grid>
                  </Grid>
                </Box>
              )}

              <AnimatedDivider 
                sx={{ 
                  my: 3,
                  '&::before, &::after': {
                    borderColor: 'rgba(0, 102, 179, 0.2)',
                  }
                }} 
              />

              <AnimatedBox 
                sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}
                style={buttonAnimation}
              >
                <AnimatedButton
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={handleResetConfig}
                  sx={{
                    position: 'relative',
                    overflow: 'hidden',
                    borderColor: '#d32f2f',
                    color: '#d32f2f',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 16px rgba(211, 47, 47, 0.1)',
                      borderColor: '#f44336',
                      background: 'rgba(211, 47, 47, 0.05)',
                    },
                    '&:after': {
                      content: '""',
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      top: 0,
                      left: 0,
                      pointerEvents: 'none',
                      background: 'radial-gradient(circle, rgba(211, 47, 47, 0.2) 0%, rgba(211, 47, 47, 0) 60%)',
                      transform: 'scale(0, 0)',
                      opacity: 0,
                      transition: 'transform 0.4s, opacity 0.3s',
                    },
                    '&:active:after': {
                      transform: 'scale(2, 2)',
                      opacity: 0,
                      transition: '0s',
                    },
                  }}
                >
                  Reset to Default
                </AnimatedButton>
                <AnimatedButton
                  variant="outlined"
                  color="primary"
                  startIcon={<RefreshIcon />}
                  onClick={() => setConfig(defaultConfig)}
                  sx={{
                    position: 'relative',
                    overflow: 'hidden',
                    borderColor: '#0066B3',
                    color: '#0066B3',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 16px rgba(0, 102, 179, 0.1)',
                      borderColor: '#2a8fd8',
                      background: 'rgba(0, 102, 179, 0.05)',
                    },
                    '&:after': {
                      content: '""',
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      top: 0,
                      left: 0,
                      pointerEvents: 'none',
                      background: 'radial-gradient(circle, rgba(0, 102, 179, 0.2) 0%, rgba(0, 102, 179, 0) 60%)',
                      transform: 'scale(0, 0)',
                      opacity: 0,
                      transition: 'transform 0.4s, opacity 0.3s',
                    },
                    '&:active:after': {
                      transform: 'scale(2, 2)',
                      opacity: 0,
                      transition: '0s',
                    },
                  }}
                >
                  Reload
                </AnimatedButton>
                <AnimatedButton
                  variant="contained"
                  color="primary"
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
                  onClick={handleSaveConfig}
                  disabled={loading}
                  sx={{
                    position: 'relative',
                    overflow: 'hidden',
                    background: loading ? '#0066B3' : 'linear-gradient(90deg, #0066B3, #2a8fd8)',
                    boxShadow: '0 4px 12px rgba(0, 102, 179, 0.2)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 16px rgba(0, 102, 179, 0.3)',
                    },
                    '&:after': {
                      content: '""',
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      top: 0,
                      left: 0,
                      pointerEvents: 'none',
                      background: 'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 60%)',
                      transform: 'scale(0, 0)',
                      opacity: 0,
                      transition: 'transform 0.4s, opacity 0.3s',
                    },
                    '&:active:after': {
                      transform: 'scale(2, 2)',
                      opacity: 0,
                      transition: '0s',
                    },
                  }}
                >
                  Save Configuration
                </AnimatedButton>
              </AnimatedBox>
            </AnimatedBox>
          )}
        </CardContent>
      </AnimatedCard>
      
      {/* Success Snackbar */}
      <Snackbar
        open={saved}
        autoHideDuration={5000}
        onClose={() => setSaved(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSaved(false)}
          severity="success"
          variant="filled"
          sx={{ 
            width: '100%',
            boxShadow: '0 4px 12px rgba(0, 102, 179, 0.2)',
            '& .MuiAlert-icon': {
              filter: 'drop-shadow(0 0 2px rgba(255, 255, 255, 0.5))'
            }
          }}
          icon={<CheckIcon sx={{ animation: 'pulse 2s infinite' }} />}
        >
          Configuration saved successfully!
        </Alert>
      </Snackbar>
    </AnimatedBox>
  );
};

export default Settings;