import React, { useState, useEffect } from 'react';
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
import axios from 'axios';

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
    <Box className="fade-in">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="500">
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure the SAP HANA Cloud LangChain integration
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Snackbar
        open={saved}
        autoHideDuration={5000}
        onClose={() => setSaved(false)}
        message="Configuration saved successfully"
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        ContentProps={{
          sx: {
            bgcolor: 'success.main',
            '& .MuiSnackbarContent-message': {
              display: 'flex',
              alignItems: 'center',
            },
          },
        }}
        message={
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CheckIcon sx={{ mr: 1 }} />
            Configuration saved successfully
          </Box>
        }
      />

      <Card sx={{ mb: 3 }}>
        <CardContent sx={{ p: 0 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab 
              label="Database" 
              icon={<StorageIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="Embeddings" 
              icon={<DataObjectIcon />}
              iconPosition="start"
            />
            <Tab 
              label="GPU Acceleration" 
              icon={<MemoryIcon />}
              iconPosition="start"
            />
            <Tab 
              label="API Settings" 
              icon={<SettingsIcon />}
              iconPosition="start"
            />
          </Tabs>

          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
              <CircularProgress />
            </Box>
          ) : (
            <Box sx={{ p: 3 }}>
              {/* Database Settings */}
              {tabValue === 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    SAP HANA Cloud Database Connection
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Configure the connection to your SAP HANA Cloud database instance
                  </Typography>

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

              <Divider sx={{ my: 3 }} />

              <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={handleResetConfig}
                >
                  Reset to Default
                </Button>
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={<RefreshIcon />}
                  onClick={() => setConfig(defaultConfig)}
                >
                  Reload
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
                  onClick={handleSaveConfig}
                  disabled={loading}
                >
                  Save Configuration
                </Button>
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default Settings;