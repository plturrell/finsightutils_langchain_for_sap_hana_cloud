import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  CircularProgress,
  Alert,
  Paper,
  Grid,
  Slider,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useTheme,
  useMediaQuery,
  IconButton,
  Tooltip,
  Chip,
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Info as InfoIcon,
  Close as CloseIcon,
  Tune as TuneIcon,
  Check as CheckIcon,
} from '@mui/icons-material';
import HumanText from './HumanText';
import { humanize } from '../utils/humanLanguage';
import axios from 'axios';

// Common types
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

// SimpleSettings props
interface SimpleSettingsProps {
  onSave?: () => void;
  onCancel?: () => void;
}

/**
 * SimpleSettings provides a streamlined configuration experience
 * with intelligent defaults and progressive disclosure of complexity.
 */
const SimpleSettings: React.FC<SimpleSettingsProps> = ({
  onSave,
  onCancel,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  // State for settings
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [showAdvancedDialog, setShowAdvancedDialog] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  
  // Simplified settings
  const [performanceLevel, setPerformanceLevel] = useState(1); // 0: Low, 1: Balanced, 2: High
  const [accuracy, setAccuracy] = useState(1); // 0: Fast, 1: Balanced, 2: Precise
  const [connectionType, setConnectionType] = useState('automatic');
  
  // Hardware detection
  const [detectedHardware, setDetectedHardware] = useState<{
    hasGpu: boolean;
    gpuName: string;
    gpuMemory: string;
    cpuName: string;
    cpuCores: number;
  }>({
    hasGpu: false,
    gpuName: 'Unknown',
    gpuMemory: '0 GB',
    cpuName: 'Unknown',
    cpuCores: 0,
  });
  
  // For converting simple settings to actual config
  const [actualConfig, setActualConfig] = useState<Config | null>(null);
  
  // Fetch hardware info and current config
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        
        // Fetch hardware info
        const hardwareResponse = await axios.get('/api/hardware-info');
        setDetectedHardware(hardwareResponse.data);
        
        // Fetch current config
        const configResponse = await axios.get('/api/config');
        const config = configResponse.data as Config;
        setActualConfig(config);
        
        // Set simplified settings based on config
        // Performance level
        if (config.gpu.use_tensorrt && config.gpu.tensorrt_precision === 'fp16' && config.gpu.batch_size >= 32) {
          setPerformanceLevel(2); // High
        } else if (config.gpu.enabled) {
          setPerformanceLevel(1); // Balanced
        } else {
          setPerformanceLevel(0); // Low
        }
        
        // Accuracy
        if (config.embeddings.model.includes('large') || config.embeddings.dimension >= 768) {
          setAccuracy(2); // Precise
        } else if (config.embeddings.model.includes('base')) {
          setAccuracy(1); // Balanced
        } else {
          setAccuracy(0); // Fast
        }
        
        // Connection type
        setConnectionType(config.database.host === 'auto-detect' ? 'automatic' : 'manual');
        
      } catch (err) {
        console.error('Error fetching initial data:', err);
        setError('Failed to load current settings and hardware information');
      } finally {
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, []);
  
  // Update actual config when simplified settings change
  useEffect(() => {
    if (!actualConfig) return;
    
    const newConfig = { ...actualConfig };
    
    // Performance level
    switch (performanceLevel) {
      case 0: // Low
        newConfig.gpu.enabled = false;
        newConfig.gpu.use_tensorrt = false;
        newConfig.gpu.batch_size = 16;
        break;
      case 1: // Balanced
        newConfig.gpu.enabled = detectedHardware.hasGpu;
        newConfig.gpu.use_tensorrt = detectedHardware.hasGpu;
        newConfig.gpu.batch_size = 32;
        newConfig.gpu.tensorrt_precision = 'fp16';
        break;
      case 2: // High
        newConfig.gpu.enabled = detectedHardware.hasGpu;
        newConfig.gpu.use_tensorrt = detectedHardware.hasGpu;
        newConfig.gpu.batch_size = 64;
        newConfig.gpu.tensorrt_precision = 'fp16';
        newConfig.gpu.use_multi_gpu = true;
        break;
    }
    
    // Accuracy
    switch (accuracy) {
      case 0: // Fast
        newConfig.embeddings.model = 'all-MiniLM-L6-v2';
        newConfig.embeddings.dimension = 384;
        break;
      case 1: // Balanced
        newConfig.embeddings.model = 'all-mpnet-base-v2';
        newConfig.embeddings.dimension = 768;
        break;
      case 2: // Precise
        newConfig.embeddings.model = 'bge-large-en-v1.5';
        newConfig.embeddings.dimension = 1024;
        break;
    }
    
    // Connection type
    if (connectionType === 'automatic') {
      newConfig.database.host = 'auto-detect';
    }
    
    setActualConfig(newConfig);
  }, [performanceLevel, accuracy, connectionType, detectedHardware]);
  
  // Handle save
  const handleSave = async () => {
    if (!actualConfig) return;
    
    try {
      setLoading(true);
      await axios.post('/api/config', actualConfig);
      setSavedMessage('Settings saved successfully');
      setTimeout(() => setSavedMessage(null), 3000);
      if (onSave) onSave();
    } catch (err) {
      console.error('Error saving config:', err);
      setError('Failed to save configuration');
    } finally {
      setLoading(false);
    }
  };
  
  // Performance level labels
  const performanceLabels = [
    { value: 0, label: 'Low', description: 'CPU only, best for compatibility' },
    { value: 1, label: 'Balanced', description: 'Optimal performance/quality balance' },
    { value: 2, label: 'High', description: 'Maximum GPU performance' },
  ];
  
  // Accuracy level labels
  const accuracyLabels = [
    { value: 0, label: 'Fast', description: 'Smaller models, faster response' },
    { value: 1, label: 'Balanced', description: 'Good accuracy/speed balance' },
    { value: 2, label: 'Precise', description: 'Larger models, highest accuracy' },
  ];
  
  // Connection type options
  const connectionOptions = [
    { value: 'automatic', label: 'Automatic', description: 'Auto-detect database' },
    { value: 'manual', label: 'Manual', description: 'Custom connection settings' },
  ];
  
  return (
    <Box>
      <HumanText variant="h5" gutterBottom>
        Settings
      </HumanText>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {savedMessage && (
        <Alert 
          severity="success" 
          sx={{ mb: 3 }} 
          icon={<CheckIcon />}
          onClose={() => setSavedMessage(null)}
        >
          {savedMessage}
        </Alert>
      )}
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {/* Hardware Detection Card */}
          <Card sx={{ mb: 3, borderRadius: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <MemoryIcon sx={{ mr: 1, color: 'primary.main' }} />
                <HumanText variant="h6">Detected Hardware</HumanText>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper 
                    variant="outlined" 
                    sx={{ 
                      p: 2, 
                      height: '100%', 
                      borderRadius: 2,
                      borderColor: detectedHardware.hasGpu ? 'success.main' : 'warning.main',
                      borderWidth: 2
                    }}
                  >
                    <HumanText variant="subtitle1" gutterBottom>
                      GPU
                    </HumanText>
                    
                    {detectedHardware.hasGpu ? (
                      <>
                        <HumanText variant="body1">
                          {detectedHardware.gpuName}
                        </HumanText>
                        <HumanText variant="body2" color="text.secondary">
                          {detectedHardware.gpuMemory} Memory
                        </HumanText>
                        <Chip 
                          label="GPU Detected" 
                          color="success" 
                          size="small" 
                          sx={{ mt: 1 }} 
                        />
                      </>
                    ) : (
                      <>
                        <HumanText variant="body1">
                          No GPU Detected
                        </HumanText>
                        <HumanText variant="body2" color="text.secondary">
                          Using CPU fallback mode
                        </HumanText>
                        <Chip 
                          label="CPU Only" 
                          color="warning" 
                          size="small" 
                          sx={{ mt: 1 }} 
                        />
                      </>
                    )}
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper 
                    variant="outlined" 
                    sx={{ 
                      p: 2, 
                      height: '100%', 
                      borderRadius: 2 
                    }}
                  >
                    <HumanText variant="subtitle1" gutterBottom>
                      CPU
                    </HumanText>
                    <HumanText variant="body1">
                      {detectedHardware.cpuName}
                    </HumanText>
                    <HumanText variant="body2" color="text.secondary">
                      {detectedHardware.cpuCores} Cores
                    </HumanText>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
          
          {/* Performance Settings Card */}
          <Card sx={{ mb: 3, borderRadius: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SpeedIcon sx={{ mr: 1, color: 'primary.main' }} />
                <HumanText variant="h6">Performance & Accuracy</HumanText>
              </Box>
              
              <Grid container spacing={4}>
                <Grid item xs={12} md={6}>
                  <HumanText gutterBottom variant="subtitle1">
                    Performance Level
                  </HumanText>
                  
                  <Box sx={{ px: 1 }}>
                    <Slider
                      value={performanceLevel}
                      onChange={(_, value) => setPerformanceLevel(value as number)}
                      step={1}
                      marks={performanceLabels.map(l => ({ value: l.value, label: l.label }))}
                      min={0}
                      max={2}
                      disabled={!detectedHardware.hasGpu && performanceLevel > 0}
                    />
                  </Box>
                  
                  <HumanText variant="body2" color="text.secondary" sx={{ mt: 1, minHeight: 40 }}>
                    {performanceLabels[performanceLevel].description}
                    {!detectedHardware.hasGpu && performanceLevel === 0 && (
                      <Box component="span" sx={{ color: 'warning.main', display: 'block', mt: 1 }}>
                        No GPU detected. Only CPU mode available.
                      </Box>
                    )}
                  </HumanText>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <HumanText gutterBottom variant="subtitle1">
                    Accuracy Level
                  </HumanText>
                  
                  <Box sx={{ px: 1 }}>
                    <Slider
                      value={accuracy}
                      onChange={(_, value) => setAccuracy(value as number)}
                      step={1}
                      marks={accuracyLabels.map(l => ({ value: l.value, label: l.label }))}
                      min={0}
                      max={2}
                    />
                  </Box>
                  
                  <HumanText variant="body2" color="text.secondary" sx={{ mt: 1, minHeight: 40 }}>
                    {accuracyLabels[accuracy].description}
                  </HumanText>
                </Grid>
              </Grid>
              
              <Divider sx={{ my: 3 }} />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <InfoIcon fontSize="small" sx={{ mr: 1, color: 'info.main' }} />
                  <HumanText variant="body2" color="text.secondary">
                    Settings are automatically optimized for your hardware
                  </HumanText>
                </Box>
                
                <Tooltip title="Advanced Configuration">
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<TuneIcon />}
                    onClick={() => setShowAdvancedDialog(true)}
                  >
                    <HumanText>Advanced</HumanText>
                  </Button>
                </Tooltip>
              </Box>
            </CardContent>
          </Card>
          
          {/* Connection Settings Card */}
          <Card sx={{ mb: 3, borderRadius: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <StorageIcon sx={{ mr: 1, color: 'primary.main' }} />
                <HumanText variant="h6">Database Connection</HumanText>
              </Box>
              
              <Grid container spacing={2}>
                {connectionOptions.map((option) => (
                  <Grid item xs={12} sm={6} key={option.value}>
                    <Paper
                      elevation={0}
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        border: '2px solid',
                        borderColor: connectionType === option.value 
                          ? 'primary.main' 
                          : 'divider',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        '&:hover': {
                          borderColor: connectionType === option.value 
                            ? 'primary.main' 
                            : 'primary.light',
                        },
                      }}
                      onClick={() => setConnectionType(option.value)}
                    >
                      <HumanText variant="subtitle1" gutterBottom>
                        {option.label}
                      </HumanText>
                      <HumanText variant="body2" color="text.secondary">
                        {option.description}
                      </HumanText>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
              
              {connectionType === 'manual' && (
                <Box sx={{ mt: 2 }}>
                  <HumanText variant="body2" color="text.secondary">
                    Use the advanced settings to configure your database connection parameters
                  </HumanText>
                  <Button
                    variant="text"
                    size="small"
                    sx={{ mt: 1 }}
                    onClick={() => setShowAdvancedDialog(true)}
                  >
                    <HumanText>Configure Database Settings</HumanText>
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
          
          {/* Buttons */}
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
            {onCancel && (
              <Button
                variant="outlined"
                onClick={onCancel}
              >
                <HumanText>Cancel</HumanText>
              </Button>
            )}
            <Button
              variant="contained"
              onClick={handleSave}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}
            >
              <HumanText>Save Settings</HumanText>
            </Button>
          </Box>
        </Box>
      )}
      
      {/* Advanced Settings Dialog */}
      <Dialog
        open={showAdvancedDialog}
        onClose={() => setShowAdvancedDialog(false)}
        fullWidth
        maxWidth="md"
        PaperProps={{
          sx: {
            borderRadius: 3,
          }
        }}
      >
        <DialogTitle sx={{ m: 0, p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <HumanText variant="h6">Advanced Settings</HumanText>
          <IconButton
            aria-label="close"
            onClick={() => setShowAdvancedDialog(false)}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent dividers>
          <HumanText variant="body2" color="text.secondary" paragraph>
            These settings are normally handled automatically. Only modify them if you understand their impact.
          </HumanText>
          
          <Alert severity="info" sx={{ mb: 3 }}>
            For a simplified experience, close this dialog and use the main settings interface.
          </Alert>
          
          {/* Advanced settings content would go here */}
          <HumanText variant="body1">
            This would contain all technical parameters from the original Settings page.
          </HumanText>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setShowAdvancedDialog(false)}>
            <HumanText>Close</HumanText>
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SimpleSettings;