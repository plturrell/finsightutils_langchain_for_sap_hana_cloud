import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Collapse,
  Divider,
  FormControl,
  FormControlLabel,
  FormHelperText,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  Slider,
  Stack,
  Step,
  StepLabel,
  Stepper,
  Switch,
  TextField,
  Typography,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Check as CheckIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Tune as TuneIcon,
  ModelTraining as ModelTrainingIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { dataPipelineService, RegisterVectorRequest } from '../api/services';
import HumanText from './HumanText';

// Types
interface VectorCreatorProps {
  pipelineId?: string;
  sourceId?: string;
  tableName?: string;
  schemaName?: string;
  onVectorCreated?: (vectorId: string) => void;
  onClose?: () => void;
}

interface ModelOption {
  id: string;
  name: string;
  dimensions: number;
  description: string;
  performance: 'fast' | 'medium' | 'slow';
  quality: 'basic' | 'good' | 'excellent';
  gpuOptimized: boolean;
}

// Default model options - will be replaced with data from the API
const DEFAULT_VECTOR_MODELS: ModelOption[] = [
  {
    id: 'SAP_NEB.20240715',
    name: 'SAP NEB Standard',
    dimensions: 768,
    description: 'HANA Cloud\'s native embedding model for general text',
    performance: 'fast',
    quality: 'good',
    gpuOptimized: false,
  },
];

const VectorCreator: React.FC<VectorCreatorProps> = ({
  pipelineId,
  sourceId,
  tableName,
  schemaName,
  onVectorCreated,
  onClose,
}) => {
  const theme = useTheme();
  
  // State
  const [activeStep, setActiveStep] = useState<number>(0);
  const [selectedModel, setSelectedModel] = useState<string>('SAP_NEB.20240715');
  const [embeddingType, setEmbeddingType] = useState<string>('DOCUMENT');
  const [usePalService, setUsePalService] = useState<boolean>(false);
  const [palBatchSize, setPalBatchSize] = useState<number>(64);
  const [dimensions, setDimensions] = useState<number>(768); // SAP NEB is 768 dimensions
  const [isCreating, setIsCreating] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [vectorId, setVectorId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);
  const [normalizeVectors, setNormalizeVectors] = useState<boolean>(true);
  const [chunkStrategy, setChunkStrategy] = useState<string>('none');
  const [chunkSize, setChunkSize] = useState<number>(256);
  const [chunkOverlap, setChunkOverlap] = useState<number>(50);
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);
  const [showPerformanceEstimate, setShowPerformanceEstimate] = useState<boolean>(false);
  const [models, setModels] = useState<ModelOption[]>(DEFAULT_VECTOR_MODELS);
  const [palAvailable, setPalAvailable] = useState<boolean>(false);

  // Fetch available models from the API
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await vectorOperationsService.listModels();
        if (response.data && response.data.models) {
          // Convert API model format to component format
          const apiModels = response.data.models.map(model => ({
            id: model.id,
            name: model.name,
            dimensions: model.dimensions,
            description: model.description,
            performance: model.performance,
            quality: model.quality,
            gpuOptimized: false, // SAP HANA has native optimizations, no GPU needed
          }));
          
          setModels(apiModels);
          setPalAvailable(response.data.pal_available);
          
          // Set recommended model if available
          if (response.data.recommended_model) {
            setSelectedModel(response.data.recommended_model);
            
            // Find the recommended model's dimension
            const recommendedModel = apiModels.find(m => m.id === response.data.recommended_model);
            if (recommendedModel) {
              setDimensions(recommendedModel.dimensions);
            }
          }
        }
      } catch (error) {
        console.error('Error fetching model information:', error);
        // Continue with default models
      }
    };
    
    fetchModels();
  }, []);
  
  // Update dimensions when model changes
  useEffect(() => {
    const model = models.find(m => m.id === selectedModel);
    if (model) {
      setDimensions(model.dimensions);
      estimateProcessingTime();
    }
  }, [selectedModel, palBatchSize, usePalService, models]);

  // Estimate processing time
  const estimateProcessingTime = () => {
    const model = models.find(m => m.id === selectedModel);
    if (!model) return;
    
    let baseTime = 0;
    switch (model.performance) {
      case 'fast':
        baseTime = 5;
        break;
      case 'medium':
        baseTime = 15;
        break;
      case 'slow':
        baseTime = 30;
        break;
    }
    
    // Adjust for PAL service
    if (usePalService) {
      baseTime *= 0.5; // 50% faster with PAL Text Embedding Service
    }
    
    // Adjust for batch size with PAL
    if (usePalService) {
      const batchFactor = 64 / palBatchSize;
      baseTime *= Math.sqrt(batchFactor); // Non-linear relationship
    }
    
    // Set estimated time in seconds
    setEstimatedTime(Math.round(baseTime));
  };

  // Handle model selection change
  const handleModelChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    const modelId = event.target.value as string;
    setSelectedModel(modelId);
  };

  // Steps in the process
  const steps = [
    'Select Embedding Model',
    'Configure Settings',
    'Create Vectors',
  ];

  // Navigate to next step
  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      // Final step - create vectors
      createVectors();
    } else {
      setActiveStep(prevStep => prevStep + 1);
    }
  };

  // Navigate to previous step
  const handleBack = () => {
    setActiveStep(prevStep => prevStep - 1);
  };

  // Create vector embeddings
  const createVectors = async () => {
    if (!pipelineId || !sourceId) {
      setError('Pipeline ID and Source ID are required to create vectors');
      return;
    }
    
    setIsCreating(true);
    setProgress(0);
    setError(null);
    
    try {
      // Create progress simulation
      const progressInterval = setInterval(() => {
        setProgress(prevProgress => {
          if (prevProgress >= 95) {
            clearInterval(progressInterval);
            return prevProgress;
          }
          return prevProgress + Math.random() * 5;
        });
      }, 1000);
      
      // Get selected model details
      const model = models.find(m => m.id === selectedModel);
      
      // Prepare request using SAP HANA's native embedding capabilities
      const request: RegisterVectorRequest = {
        pipeline_id: pipelineId,
        source_id: sourceId,
        model_name: selectedModel, // Use the SAP NEB model ID
        vector_dimensions: dimensions,
        normalize_vectors: normalizeVectors,
        chunking_strategy: chunkStrategy,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedding_type: embeddingType,
        use_pal_service: usePalService,
        pal_batch_size: palBatchSize
      };
      
      // Send request to create vectors
      const response = await dataPipelineService.registerVector(request);
      
      clearInterval(progressInterval);
      setProgress(100);
      
      // Set vector ID from response
      if (response.data && response.data.vector_id) {
        setVectorId(response.data.vector_id);
        if (onVectorCreated) {
          onVectorCreated(response.data.vector_id);
        }
      }
    } catch (err) {
      console.error('Error creating vectors:', err);
      setError('Failed to create vector embeddings. Please try again.');
    } finally {
      setIsCreating(false);
    }
  };

  // Render model selection step
  const renderModelSelection = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <HumanText variant="h6" gutterBottom>
          Select an Embedding Model
        </HumanText>
        
        <HumanText variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Choose the model that best fits your use case. Models differ in quality, performance, and dimension size.
        </HumanText>
        
        <Grid container spacing={3}>
          {models.map(model => (
            <Grid item xs={12} key={model.id}>
              <Paper
                elevation={0}
                sx={{
                  p: 2,
                  border: '1px solid',
                  borderColor: selectedModel === model.id 
                    ? theme.palette.primary.main 
                    : alpha(theme.palette.divider, 0.8),
                  borderRadius: 2,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  position: 'relative',
                  bgcolor: selectedModel === model.id 
                    ? alpha(theme.palette.primary.main, 0.04)
                    : 'background.paper',
                  '&:hover': {
                    borderColor: selectedModel === model.id 
                      ? theme.palette.primary.main 
                      : theme.palette.primary.light,
                    boxShadow: selectedModel === model.id 
                      ? `0 0 0 1px ${theme.palette.primary.main}`
                      : 'none',
                  },
                }}
                onClick={() => setSelectedModel(model.id)}
              >
                <Grid container spacing={2} alignItems="center">
                  {/* Radio button */}
                  <Grid item xs={1}>
                    <Radio
                      checked={selectedModel === model.id}
                      onChange={() => setSelectedModel(model.id)}
                      sx={{ p: 0.5 }}
                    />
                  </Grid>
                  
                  {/* Model info */}
                  <Grid item xs={8}>
                    <HumanText variant="subtitle1" fontWeight={500}>
                      {model.name}
                    </HumanText>
                    <HumanText variant="body2" color="text.secondary">
                      {model.description}
                    </HumanText>
                  </Grid>
                  
                  {/* Dimensions and metrics */}
                  <Grid item xs={3}>
                    <Stack spacing={1} alignItems="flex-end">
                      <HumanText variant="caption" fontWeight={500}>
                        {model.dimensions} dimensions
                      </HumanText>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Box 
                          sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: 0.5,
                            color: 
                              model.performance === 'fast' ? theme.palette.success.main :
                              model.performance === 'medium' ? theme.palette.warning.main :
                              theme.palette.error.main
                          }}
                        >
                          <SpeedIcon sx={{ fontSize: 14 }} />
                          <HumanText 
                            variant="caption" 
                            fontWeight={500}
                            sx={{ fontSize: '0.675rem' }}
                          >
                            {model.performance === 'fast' ? 'Fast' : 
                             model.performance === 'medium' ? 'Medium' : 'Slow'}
                          </HumanText>
                        </Box>
                        <Box 
                          sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: 0.5,
                            color: 
                              model.quality === 'excellent' ? theme.palette.success.main :
                              model.quality === 'good' ? theme.palette.info.main :
                              theme.palette.warning.main
                          }}
                        >
                          <TuneIcon sx={{ fontSize: 14 }} />
                          <HumanText 
                            variant="caption" 
                            fontWeight={500}
                            sx={{ fontSize: '0.675rem' }}
                          >
                            {model.quality === 'excellent' ? 'High Quality' : 
                             model.quality === 'good' ? 'Good Quality' : 'Basic'}
                          </HumanText>
                        </Box>
                      </Box>
                    </Stack>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  // Render configuration step
  const renderConfiguration = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <HumanText variant="h6" gutterBottom>
          Configure Embedding Settings
        </HumanText>
        
        <HumanText variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Customize how your vector embeddings will be created. You can use the default settings or adjust them for your specific needs.
        </HumanText>
        
        <Grid container spacing={3}>
          {/* Basic Settings */}
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel id="embedding-type-label">Embedding Type</InputLabel>
              <Select
                labelId="embedding-type-label"
                value={embeddingType}
                label="Embedding Type"
                onChange={e => setEmbeddingType(e.target.value)}
              >
                <MenuItem value="DOCUMENT">DOCUMENT (For Content)</MenuItem>
                <MenuItem value="QUERY">QUERY (For Search Terms)</MenuItem>
                <MenuItem value="CODE">CODE (For Programming Code)</MenuItem>
              </Select>
              <FormHelperText>
                The type of text determines how the embedding model processes it
              </FormHelperText>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <FormControlLabel
                control={
                  <Switch 
                    checked={usePalService} 
                    onChange={e => setUsePalService(e.target.checked)} 
                    color="primary"
                    disabled={!palAvailable}
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <HumanText>Use PAL Text Embedding Service</HumanText>
                    <SpeedIcon 
                      color={usePalService && palAvailable ? "primary" : "disabled"} 
                      sx={{ fontSize: 16, ml: 0.5 }} 
                    />
                  </Box>
                }
              />
              <FormHelperText>
                {!palAvailable 
                  ? "PAL Text Embedding Service not available in this HANA instance" 
                  : usePalService 
                    ? "PAL service optimizes batch processing for higher throughput" 
                    : "Standard VECTOR_EMBEDDING function will be used"}
              </FormHelperText>
            </FormControl>
          </Grid>
          
          {usePalService && palAvailable && (
            <Grid item xs={12}>
              <FormControl fullWidth>
                <HumanText gutterBottom>
                  PAL Batch Size: {palBatchSize}
                </HumanText>
                <Slider
                  value={palBatchSize}
                  onChange={(e, newValue) => setPalBatchSize(newValue as number)}
                  min={16}
                  max={256}
                  step={16}
                  marks={[
                    { value: 16, label: '16' },
                    { value: 64, label: '64' },
                    { value: 128, label: '128' },
                    { value: 256, label: '256' },
                  ]}
                  valueLabelDisplay="auto"
                  aria-labelledby="pal-batch-size-slider"
                />
                <FormHelperText>
                  PAL batch size controls the number of texts processed in a single batch
                </FormHelperText>
              </FormControl>
            </Grid>
          )}
          
          {/* Performance Estimate */}
          {showPerformanceEstimate && estimatedTime !== null && (
            <Grid item xs={12}>
              <Paper
                elevation={0}
                sx={{
                  p: 2,
                  borderRadius: 2,
                  bgcolor: alpha(theme.palette.info.main, 0.08),
                  border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <InfoIcon color="info" sx={{ mr: 1.5 }} />
                <Box>
                  <HumanText variant="body2">
                    Estimated processing time: <strong>{estimatedTime} seconds</strong> per 1000 records
                  </HumanText>
                  <HumanText variant="caption" color="text.secondary">
                    Actual time may vary based on data complexity and system load
                  </HumanText>
                </Box>
              </Paper>
            </Grid>
          )}
          
          {/* Advanced Settings Toggle */}
          <Grid item xs={12}>
            <Box 
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                mt: 1, 
                cursor: 'pointer' 
              }}
              onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
            >
              <SettingsIcon 
                fontSize="small" 
                sx={{ mr: 1, color: showAdvancedSettings ? 'primary.main' : 'action.active' }} 
              />
              <HumanText 
                variant="subtitle2" 
                sx={{ color: showAdvancedSettings ? 'primary.main' : 'text.primary' }}
              >
                Advanced Settings
              </HumanText>
            </Box>
          </Grid>
          
          {/* Advanced Settings */}
          <Grid item xs={12}>
            <Collapse in={showAdvancedSettings}>
              <Paper 
                elevation={0}
                sx={{ 
                  p: 2, 
                  mt: 1, 
                  border: '1px solid',
                  borderColor: alpha(theme.palette.divider, 0.8),
                  borderRadius: 2,
                }}
              >
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <FormControlLabel
                        control={
                          <Switch 
                            checked={normalizeVectors} 
                            onChange={e => setNormalizeVectors(e.target.checked)} 
                            color="primary"
                          />
                        }
                        label="Normalize Vectors"
                      />
                      <FormHelperText>
                        Normalizing vectors to unit length improves similarity search results
                      </FormHelperText>
                    </FormControl>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel id="chunk-strategy-label">Text Chunking Strategy</InputLabel>
                      <Select
                        labelId="chunk-strategy-label"
                        value={chunkStrategy}
                        label="Text Chunking Strategy"
                        onChange={e => setChunkStrategy(e.target.value)}
                      >
                        <MenuItem value="none">No Chunking</MenuItem>
                        <MenuItem value="fixed">Fixed Size Chunks</MenuItem>
                        <MenuItem value="sentence">Sentence Boundaries</MenuItem>
                        <MenuItem value="paragraph">Paragraph Boundaries</MenuItem>
                      </Select>
                      <FormHelperText>
                        Chunking breaks long text into smaller pieces for better embeddings
                      </FormHelperText>
                    </FormControl>
                  </Grid>
                  
                  {chunkStrategy !== 'none' && (
                    <>
                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                          <HumanText gutterBottom>
                            Chunk Size: {chunkSize} tokens
                          </HumanText>
                          <Slider
                            value={chunkSize}
                            onChange={(e, newValue) => setChunkSize(newValue as number)}
                            min={64}
                            max={1024}
                            step={32}
                            marks={[
                              { value: 64, label: '64' },
                              { value: 256, label: '256' },
                              { value: 512, label: '512' },
                              { value: 1024, label: '1024' },
                            ]}
                            valueLabelDisplay="auto"
                            aria-labelledby="chunk-size-slider"
                          />
                        </FormControl>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                          <HumanText gutterBottom>
                            Chunk Overlap: {chunkOverlap} tokens
                          </HumanText>
                          <Slider
                            value={chunkOverlap}
                            onChange={(e, newValue) => setChunkOverlap(newValue as number)}
                            min={0}
                            max={128}
                            step={8}
                            marks={[
                              { value: 0, label: '0' },
                              { value: 32, label: '32' },
                              { value: 64, label: '64' },
                              { value: 128, label: '128' },
                            ]}
                            valueLabelDisplay="auto"
                            aria-labelledby="chunk-overlap-slider"
                          />
                        </FormControl>
                      </Grid>
                    </>
                  )}
                </Grid>
              </Paper>
            </Collapse>
          </Grid>
        </Grid>
        
        {/* Show Performance Estimate Button */}
        <Button
          variant="text"
          color="primary"
          sx={{ mt: 2 }}
          onClick={() => {
            estimateProcessingTime();
            setShowPerformanceEstimate(true);
          }}
          startIcon={<SpeedIcon />}
        >
          Estimate Performance
        </Button>
      </Box>
    );
  };

  // Render vector creation step
  const renderVectorCreation = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <HumanText variant="h6" gutterBottom>
          Create Vector Embeddings
        </HumanText>
        
        <HumanText variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Click the button below to start the vector embedding creation process. This may take some time depending on your data size and model selection.
        </HumanText>
        
        <Paper
          elevation={0}
          sx={{
            p: 3,
            border: '1px solid',
            borderColor: alpha(theme.palette.divider, 0.8),
            borderRadius: 2,
            mb: 3,
          }}
        >
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <HumanText variant="subtitle2" gutterBottom>
                Selected Model
              </HumanText>
              <HumanText variant="body2" color="text.secondary">
                {models.find(m => m.id === selectedModel)?.name || 'SAP NEB Standard'}
              </HumanText>
            </Grid>
            <Grid item xs={12} md={6}>
              <HumanText variant="subtitle2" gutterBottom>
                Dimensions
              </HumanText>
              <HumanText variant="body2" color="text.secondary">
                {dimensions}
              </HumanText>
            </Grid>
            <Grid item xs={12} md={6}>
              <HumanText variant="subtitle2" gutterBottom>
                Table
              </HumanText>
              <HumanText variant="body2" color="text.secondary">
                {schemaName ? `${schemaName}.${tableName}` : tableName || 'N/A'}
              </HumanText>
            </Grid>
            <Grid item xs={12} md={6}>
              <HumanText variant="subtitle2" gutterBottom>
                Embedding Type
              </HumanText>
              <HumanText variant="body2" color="text.secondary">
                {embeddingType} {usePalService ? '(PAL Service)' : '(VECTOR_EMBEDDING)'}
              </HumanText>
            </Grid>
          </Grid>
          
          {isCreating && (
            <Box sx={{ mt: 3 }}>
              <HumanText variant="body2" gutterBottom>
                Creating vector embeddings... ({Math.round(progress)}%)
              </HumanText>
              <LinearProgress 
                variant="determinate" 
                value={progress} 
                sx={{ height: 8, borderRadius: 1 }} 
              />
            </Box>
          )}
          
          {vectorId && (
            <Box 
              sx={{ 
                mt: 3, 
                p: 2, 
                bgcolor: alpha(theme.palette.success.main, 0.1),
                borderRadius: 1,
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <CheckIcon color="success" sx={{ mr: 1.5 }} />
              <Box>
                <HumanText variant="subtitle2" color="success.main">
                  Vector embeddings created successfully!
                </HumanText>
                <HumanText variant="body2">
                  Vector ID: {vectorId}
                </HumanText>
              </Box>
            </Box>
          )}
          
          {error && (
            <Box 
              sx={{ 
                mt: 3, 
                p: 2, 
                bgcolor: alpha(theme.palette.error.main, 0.1),
                borderRadius: 1,
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <ErrorIcon color="error" sx={{ mr: 1.5 }} />
              <HumanText variant="body2" color="error">
                {error}
              </HumanText>
            </Box>
          )}
        </Paper>
        
        {vectorId && (
          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Button
              variant="contained"
              color="primary"
              onClick={onClose}
            >
              Continue
            </Button>
          </Box>
        )}
      </Box>
    );
  };

  // Render active step content
  const getStepContent = (step: number) => {
    switch (step) {
      case 0:
        return renderModelSelection();
      case 1:
        return renderConfiguration();
      case 2:
        return renderVectorCreation();
      default:
        return 'Unknown step';
    }
  };

  return (
    <Card 
      sx={{ 
        width: '100%', 
        overflow: 'hidden',
        borderRadius: { xs: 2, md: 3 },
        boxShadow: 3,
      }}
    >
      <CardContent sx={{ p: { xs: 2, sm: 3 } }}>
        <HumanText variant="h5" gutterBottom>
          Vector Embedding Creation
        </HumanText>
        
        <HumanText variant="body2" color="text.secondary" paragraph>
          Create vector embeddings from your data to enable semantic search and similarity analysis.
        </HumanText>
        
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map(label => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Box>
          {getStepContent(activeStep)}
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button
              variant="outlined"
              disabled={activeStep === 0 || isCreating}
              onClick={handleBack}
            >
              Back
            </Button>
            <Box>
              {onClose && (
                <Button
                  variant="text"
                  onClick={onClose}
                  sx={{ mr: 1 }}
                  disabled={isCreating}
                >
                  Cancel
                </Button>
              )}
              <Button
                variant="contained"
                color="primary"
                onClick={handleNext}
                disabled={isCreating || (activeStep === steps.length - 1 && !!vectorId)}
              >
                {activeStep === steps.length - 1 ? (
                  isCreating ? (
                    <CircularProgress size={24} sx={{ color: '#fff' }} />
                  ) : vectorId ? 'Complete' : 'Create Vectors'
                ) : (
                  'Next'
                )}
              </Button>
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default VectorCreator;