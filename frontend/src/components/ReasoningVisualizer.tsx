import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  LinearProgress,
  alpha,
  Menu,
  MenuItem,
  Collapse,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Stack,
} from '@mui/material';
import {
  Search as SearchIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Lightbulb as LightbulbIcon,
  Visibility as VisibilityIcon,
  Feedback as FeedbackIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  FilterList as FilterListIcon,
  MoreVert as MoreVertIcon,
  Refresh as RefreshIcon,
  StarBorder as StarBorderIcon,
  Star as StarIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
} from '@mui/icons-material';
import { 
  reasoningService, 
  ReasoningPathRequest, 
  ReasoningPathResponse,
  ValidationRequest,
  ValidationResponse,
  TransformationRequest,
  TransformationResponse,
  MetricsRequest,
  MetricsResponse,
  FeedbackRequest,
  FingerprintRequest,
  FingerprintResponse
} from '../api/services';
import HumanText from './HumanText';

interface ReasoningVisualizerProps {
  tableName?: string;
  documentId?: string;
  initialQuery?: string;
}

const ReasoningVisualizer: React.FC<ReasoningVisualizerProps> = ({
  tableName = 'EMBEDDINGS',
  documentId,
  initialQuery = '',
}) => {
  // State for the query
  const [query, setQuery] = useState<string>(initialQuery);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // State for reasoning path
  const [reasoningPath, setReasoningPath] = useState<ReasoningPathResponse | null>(null);
  const [activeStep, setActiveStep] = useState<number>(0);
  
  // State for validation
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  
  // State for transformation visualization
  const [transformation, setTransformation] = useState<TransformationResponse | null>(null);
  const [isLoadingTransformation, setIsLoadingTransformation] = useState<boolean>(false);
  
  // State for metrics
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState<boolean>(false);
  
  // State for fingerprint
  const [fingerprint, setFingerprint] = useState<FingerprintResponse | null>(null);
  const [isLoadingFingerprint, setIsLoadingFingerprint] = useState<boolean>(false);
  
  // State for tab selection
  const [activeTab, setActiveTab] = useState<number>(0);
  
  // State for expanded content
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});
  
  // State for feedback
  const [feedbackRating, setFeedbackRating] = useState<number | null>(null);
  const [feedbackText, setFeedbackText] = useState<string>('');
  const [isFeedbackSubmitting, setIsFeedbackSubmitting] = useState<boolean>(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);
  
  // Track reasoning path
  const trackReasoningPath = async () => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setReasoningPath(null);
    setValidation(null);
    setActiveStep(0);
    
    try {
      const request: ReasoningPathRequest = {
        query,
        table_name: tableName,
        document_id: documentId,
        include_content: true,
        max_steps: 5,
      };
      
      const response = await reasoningService.trackReasoningPath(request);
      
      // Set the reasoning path data
      setReasoningPath(response.data);
      
      // Auto-expand the first step
      setExpandedSteps({ 0: true });
      
    } catch (err: any) {
      console.error('Error tracking reasoning path:', err);
      setError(err.response?.data?.message || 'Failed to track reasoning path');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Validate reasoning
  const validateReasoning = async () => {
    if (!reasoningPath) return;
    
    setIsValidating(true);
    
    try {
      const request: ValidationRequest = {
        reasoning_path_id: reasoningPath.path_id,
        validation_types: ['consistency', 'calculations', 'citations', 'hallucination'],
      };
      
      const response = await reasoningService.validateReasoning(request);
      
      // Set the validation data
      setValidation(response.data);
      
    } catch (err: any) {
      console.error('Error validating reasoning:', err);
      setError(err.response?.data?.message || 'Failed to validate reasoning');
    } finally {
      setIsValidating(false);
    }
  };
  
  // Get transformation data
  const getTransformation = async (docId: string) => {
    setIsLoadingTransformation(true);
    
    try {
      const request: TransformationRequest = {
        document_id: docId,
        table_name: tableName,
        include_intermediate: true,
      };
      
      const response = await reasoningService.trackTransformation(request);
      
      // Set the transformation data
      setTransformation(response.data);
      
    } catch (err: any) {
      console.error('Error getting transformation data:', err);
      // Don't set global error for this, just log it
    } finally {
      setIsLoadingTransformation(false);
    }
  };
  
  // Get metrics data
  const getMetrics = async (docId?: string) => {
    setIsLoadingMetrics(true);
    
    try {
      const request: MetricsRequest = {
        document_id: docId,
        table_name: tableName,
        metric_types: ['cosine_similarity', 'information_retention', 'structural_integrity'],
      };
      
      const response = await reasoningService.calculateMetrics(request);
      
      // Set the metrics data
      setMetrics(response.data);
      
    } catch (err: any) {
      console.error('Error getting metrics data:', err);
      // Don't set global error for this, just log it
    } finally {
      setIsLoadingMetrics(false);
    }
  };
  
  // Get fingerprint data
  const getFingerprint = async (docId: string) => {
    setIsLoadingFingerprint(true);
    
    try {
      const request: FingerprintRequest = {
        document_id: docId,
        table_name: tableName,
        include_lineage: true,
      };
      
      const response = await reasoningService.getFingerprint(request);
      
      // Set the fingerprint data
      setFingerprint(response.data);
      
    } catch (err: any) {
      console.error('Error getting fingerprint data:', err);
      // Don't set global error for this, just log it
    } finally {
      setIsLoadingFingerprint(false);
    }
  };
  
  // Submit feedback
  const submitFeedback = async () => {
    if (!reasoningPath) return;
    
    setIsFeedbackSubmitting(true);
    
    try {
      const request: FeedbackRequest = {
        query: reasoningPath.query,
        reasoning_path_id: reasoningPath.path_id,
        feedback_type: 'reasoning_quality',
        feedback_content: feedbackText,
        rating: feedbackRating,
      };
      
      await reasoningService.submitFeedback(request);
      
      // Reset feedback form
      setFeedbackSubmitted(true);
      setTimeout(() => {
        setFeedbackSubmitted(false);
        setFeedbackText('');
        setFeedbackRating(null);
      }, 3000);
      
    } catch (err: any) {
      console.error('Error submitting feedback:', err);
      setError(err.response?.data?.message || 'Failed to submit feedback');
    } finally {
      setIsFeedbackSubmitting(false);
    }
  };
  
  // Toggle expanded step
  const toggleExpandStep = (stepIndex: number) => {
    setExpandedSteps(prev => ({
      ...prev,
      [stepIndex]: !prev[stepIndex]
    }));
  };
  
  // Handle document click to get transformation and fingerprint
  const handleDocumentClick = (docId: string) => {
    getTransformation(docId);
    getFingerprint(docId);
    
    // Switch to the transformation tab
    setActiveTab(1);
  };
  
  // Format confidence as percentage
  const formatConfidence = (confidence: number): string => {
    return `${(confidence * 100).toFixed(0)}%`;
  };
  
  // Render validation badge
  const renderValidationBadge = () => {
    if (!validation) return null;
    
    const score = validation.score;
    let color = 'success';
    let icon = <CheckIcon />;
    
    if (score < 0.7) {
      color = 'error';
      icon = <ErrorIcon />;
    } else if (score < 0.9) {
      color = 'warning';
      icon = <WarningIcon />;
    }
    
    return (
      <Chip
        icon={icon}
        label={`${(score * 100).toFixed(0)}% Valid`}
        color={color as 'success' | 'error' | 'warning'}
        size="small"
        sx={{ ml: 2 }}
      />
    );
  };
  
  // Render the reasoning steps
  const renderReasoningSteps = () => {
    if (!reasoningPath) return null;
    
    return (
      <Stepper orientation="vertical" activeStep={activeStep} nonLinear>
        {reasoningPath.steps.map((step, index) => (
          <Step key={index} expanded={expandedSteps[index] || false}>
            <StepLabel
              onClick={() => toggleExpandStep(index)}
              sx={{ cursor: 'pointer' }}
              StepIconProps={{
                sx: {
                  color: index === activeStep ? 'primary.main' : undefined,
                }
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                <HumanText variant="subtitle2" sx={{ fontWeight: 600, color: index === activeStep ? 'primary.main' : 'text.primary' }}>
                  {step.description}
                </HumanText>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Chip 
                    size="small" 
                    label={formatConfidence(step.confidence)}
                    color={step.confidence > 0.8 ? 'success' : step.confidence > 0.6 ? 'warning' : 'error'}
                    sx={{ height: 24, mr: 1 }}
                  />
                  {expandedSteps[index] ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
                </Box>
              </Box>
            </StepLabel>
            <StepContent>
              <Box sx={{ mt: 1, mb: 2 }}>
                <HumanText variant="body2" color="text.secondary" sx={{ mb: 2, whiteSpace: 'pre-line' }}>
                  {step.reasoning}
                </HumanText>
                
                <Divider sx={{ my: 2 }} />
                
                <HumanText variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  References:
                </HumanText>
                
                {step.references.map((ref, refIndex) => (
                  <Paper
                    key={refIndex}
                    sx={{
                      p: 2,
                      mb: 2,
                      backgroundColor: alpha('#f5f8fc', 0.7),
                      borderRadius: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                      cursor: 'pointer',
                    }}
                    onClick={() => handleDocumentClick(ref.id)}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Chip 
                        size="small" 
                        label={`Relevance: ${(ref.relevance * 100).toFixed(0)}%`}
                        color={ref.relevance > 0.8 ? 'success' : ref.relevance > 0.6 ? 'warning' : 'error'}
                        sx={{ height: 24 }}
                      />
                      <Tooltip title="View document details">
                        <IconButton size="small">
                          <VisibilityIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                    
                    {ref.content && (
                      <HumanText variant="body2" sx={{ fontSize: '0.875rem' }}>
                        {ref.content}
                      </HumanText>
                    )}
                  </Paper>
                ))}
              </Box>
            </StepContent>
          </Step>
        ))}
      </Stepper>
    );
  };
  
  // Render transformation visualization
  const renderTransformation = () => {
    if (isLoadingTransformation) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
          <CircularProgress size={40} />
        </Box>
      );
    }
    
    if (!transformation) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200, p: 3 }}>
          <HumanText variant="body2" color="text.secondary">
            Click on a document reference to view its transformation process
          </HumanText>
        </Box>
      );
    }
    
    return (
      <Box>
        <Box sx={{ mb: 3 }}>
          <HumanText variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Transformation Pipeline for Document: {transformation.document_id}
          </HumanText>
          <HumanText variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This visualizes how the document transforms through each stage of the vector embedding process.
          </HumanText>
        </Box>
        
        <Stepper orientation="vertical">
          {transformation.stages.map((stage, index) => (
            <Step key={index} active={true} completed={true}>
              <StepLabel>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                  <HumanText variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {stage.name}
                  </HumanText>
                  <Chip 
                    size="small" 
                    label={`${stage.duration_ms.toFixed(2)}ms`}
                    color="primary"
                    variant="outlined"
                    sx={{ height: 24 }}
                  />
                </Box>
              </StepLabel>
              <StepContent>
                <Box sx={{ mt: 1, mb: 2 }}>
                  <HumanText variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    {stage.description}
                  </HumanText>
                  
                  <Box sx={{ display: 'flex', gap: 2, mb: 2, mt: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      size="small" 
                      label={`Input: ${stage.input_type}`}
                      variant="outlined"
                      sx={{ height: 24 }}
                    />
                    <Chip 
                      size="small" 
                      label={`Output: ${stage.output_type}`}
                      variant="outlined"
                      sx={{ height: 24 }}
                    />
                  </Box>
                  
                  {stage.sample_output && (
                    <Paper 
                      sx={{ 
                        p: 2, 
                        backgroundColor: alpha('#f5f8fc', 0.7),
                        borderRadius: 1,
                        maxHeight: 100,
                        overflow: 'auto',
                        mt: 1
                      }}
                    >
                      <HumanText variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                        Sample Output:
                      </HumanText>
                      <HumanText variant="body2" sx={{ fontSize: '0.8rem', whiteSpace: 'pre-wrap' }}>
                        {typeof stage.sample_output === 'object' 
                          ? JSON.stringify(stage.sample_output, null, 2) 
                          : String(stage.sample_output)}
                      </HumanText>
                    </Paper>
                  )}
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>
      </Box>
    );
  };
  
  // Render metrics visualization
  const renderMetrics = () => {
    if (isLoadingMetrics) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
          <CircularProgress size={40} />
        </Box>
      );
    }
    
    if (!metrics) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200, p: 3 }}>
          <HumanText variant="body2" color="text.secondary">
            Select a document to view its information preservation metrics
          </HumanText>
        </Box>
      );
    }
    
    return (
      <Box>
        <Box sx={{ mb: 3 }}>
          <HumanText variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Information Preservation Metrics
            {metrics.document_id ? ` for Document: ${metrics.document_id}` : ' (Global)'}
          </HumanText>
          <HumanText variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            These metrics show how well information is preserved through the vector transformation process.
          </HumanText>
        </Box>
        
        <Grid container spacing={2}>
          {Object.entries(metrics.metrics).map(([key, value]) => (
            <Grid item xs={12} md={6} key={key}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <HumanText variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                </HumanText>
                
                {typeof value === 'number' ? (
                  <Box sx={{ mt: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <LinearProgress 
                        variant="determinate" 
                        value={value * 100} 
                        sx={{ 
                          height: 10, 
                          borderRadius: 5, 
                          width: '100%',
                          backgroundColor: alpha('#000', 0.05),
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 5,
                            backgroundColor: value > 0.8 ? 'success.main' : value > 0.6 ? 'warning.main' : 'error.main',
                          }
                        }} 
                      />
                      <HumanText variant="body2" sx={{ ml: 2, fontWeight: 600, minWidth: 45 }}>
                        {(value * 100).toFixed(0)}%
                      </HumanText>
                    </Box>
                  </Box>
                ) : (
                  <HumanText variant="body2" color="text.secondary">
                    {JSON.stringify(value)}
                  </HumanText>
                )}
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };
  
  // Render fingerprint visualization
  const renderFingerprint = () => {
    if (isLoadingFingerprint) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
          <CircularProgress size={40} />
        </Box>
      );
    }
    
    if (!fingerprint) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200, p: 3 }}>
          <HumanText variant="body2" color="text.secondary">
            Select a document to view its information fingerprint
          </HumanText>
        </Box>
      );
    }
    
    return (
      <Box>
        <Box sx={{ mb: 3 }}>
          <HumanText variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Information Fingerprint for Document: {fingerprint.document_id}
          </HumanText>
          <HumanText variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This fingerprint identifies key information and tracks its lineage through transformations.
          </HumanText>
        </Box>
        
        <Paper sx={{ p: 2, mb: 3 }}>
          <HumanText variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Signatures
          </HumanText>
          
          <List dense>
            {Object.entries(fingerprint.signatures).map(([key, value]) => (
              <ListItem key={key} divider>
                <ListItemText
                  primary={key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                  secondary={typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  primaryTypographyProps={{ variant: 'body2', fontWeight: 600 }}
                  secondaryTypographyProps={{ variant: 'caption' }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
        
        {fingerprint.lineage && (
          <Paper sx={{ p: 2 }}>
            <HumanText variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Information Lineage
            </HumanText>
            
            <HumanText variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Trace how information evolves through transformations.
            </HumanText>
            
            {Object.entries(fingerprint.lineage).map(([key, value]) => (
              <Box key={key} sx={{ mb: 2 }}>
                <HumanText variant="body2" sx={{ fontWeight: 600 }}>
                  {key}:
                </HumanText>
                <Paper 
                  sx={{ 
                    p: 1.5, 
                    backgroundColor: alpha('#f5f8fc', 0.7),
                    borderRadius: 1,
                    mt: 0.5
                  }}
                >
                  <HumanText variant="body2" sx={{ fontSize: '0.8rem' }}>
                    {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                  </HumanText>
                </Paper>
              </Box>
            ))}
          </Paper>
        )}
      </Box>
    );
  };
  
  // Render feedback form
  const renderFeedbackForm = () => {
    if (!reasoningPath) return null;
    
    return (
      <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
        <HumanText variant="subtitle2" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center' }}>
          <FeedbackIcon fontSize="small" sx={{ mr: 1 }} />
          Feedback on this reasoning path
        </HumanText>
        
        {feedbackSubmitted ? (
          <Alert severity="success" sx={{ mb: 2 }}>
            Thank you for your feedback! It will help improve the system.
          </Alert>
        ) : (
          <>
            <Box sx={{ mb: 2 }}>
              <HumanText variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                How would you rate the quality of this reasoning?
              </HumanText>
              <Stack direction="row" spacing={1}>
                {[1, 2, 3, 4, 5].map((rating) => (
                  <IconButton
                    key={rating}
                    color={feedbackRating === rating ? 'primary' : 'default'}
                    onClick={() => setFeedbackRating(rating)}
                    sx={{ 
                      p: 1,
                      color: feedbackRating === rating ? 'primary.main' : feedbackRating && rating <= feedbackRating ? 'primary.light' : 'text.disabled',
                    }}
                  >
                    {rating <= (feedbackRating || 0) ? <StarIcon /> : <StarBorderIcon />}
                  </IconButton>
                ))}
              </Stack>
            </Box>
            
            <TextField
              fullWidth
              multiline
              rows={2}
              variant="outlined"
              placeholder="What did you think of this reasoning process? How could it be improved?"
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              sx={{ mb: 2 }}
            />
            
            <Button
              variant="contained"
              color="primary"
              onClick={submitFeedback}
              disabled={isFeedbackSubmitting || !feedbackRating}
              startIcon={isFeedbackSubmitting ? <CircularProgress size={16} /> : <FeedbackIcon />}
              sx={{ mt: 1 }}
            >
              Submit Feedback
            </Button>
          </>
        )}
      </Box>
    );
  };
  
  // Handle tab change
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
    
    // Load metrics when switching to the metrics tab
    if (newValue === 2 && !metrics) {
      const docId = fingerprint?.document_id;
      getMetrics(docId);
    }
  };
  
  // Initialize with document data if provided
  useEffect(() => {
    if (documentId) {
      getTransformation(documentId);
      getFingerprint(documentId);
    }
  }, [documentId]);
  
  return (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        borderRadius: { xs: 2, md: 3 },
        boxShadow: 3,
      }}
    >
      {/* Header */}
      <Box 
        sx={{ 
          p: { xs: 2, sm: 3 }, 
          borderBottom: '1px solid', 
          borderColor: 'divider',
          display: 'flex',
          flexDirection: { xs: 'column', sm: 'row' },
          alignItems: { xs: 'flex-start', sm: 'center' },
          justifyContent: 'space-between',
          gap: 2,
        }}
      >
        <Box>
          <HumanText variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
            Reasoning Transparency
            {validation && renderValidationBadge()}
          </HumanText>
          <HumanText variant="body2" color="text.secondary">
            Explore how the system reasons with your data and visualize the transformation process.
          </HumanText>
        </Box>
        
        {reasoningPath && (
          <Button
            variant="outlined"
            size="small"
            color="primary"
            startIcon={isValidating ? <CircularProgress size={16} /> : <LightbulbIcon />}
            onClick={validateReasoning}
            disabled={isValidating}
            sx={{ 
              height: 36,
              whiteSpace: 'nowrap',
            }}
          >
            Validate Reasoning
          </Button>
        )}
      </Box>
      
      {/* Query input and visualization */}
      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', p: { xs: 2, sm: 3 } }}>
        {/* Query input */}
        <Box 
          sx={{ 
            display: 'flex', 
            flexDirection: { xs: 'column', sm: 'row' }, 
            gap: { xs: 2, sm: 1 },
            mb: 3,
          }}
        >
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Enter a query to track reasoning..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            sx={{ flex: 1 }}
            InputProps={{
              startAdornment: (
                <SearchIcon color="action" sx={{ ml: 0.5, mr: 1, opacity: 0.7 }} />
              ),
            }}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={trackReasoningPath}
            disabled={isLoading}
            startIcon={isLoading ? <CircularProgress size={16} /> : null}
            sx={{ 
              height: { sm: 56 }, 
              px: { xs: 3, sm: 4 },
              boxShadow: 'none',
              '&:hover': {
                boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
              }
            }}
          >
            Track Reasoning
          </Button>
        </Box>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {/* Results area */}
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', flexDirection: 'column' }}>
              <CircularProgress size={40} sx={{ mb: 2 }} />
              <HumanText variant="body2" color="text.secondary">
                Tracking reasoning path...
              </HumanText>
            </Box>
          ) : reasoningPath ? (
            <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Tabs */}
              <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                <Tabs 
                  value={activeTab} 
                  onChange={handleTabChange}
                  variant="scrollable"
                  scrollButtons="auto"
                >
                  <Tab label="Reasoning Path" />
                  <Tab label="Transformation" />
                  <Tab label="Metrics" />
                  <Tab label="Fingerprint" />
                </Tabs>
              </Box>
              
              {/* Tab content */}
              <Box sx={{ flexGrow: 1, overflow: 'auto', px: 0.5 }}>
                {activeTab === 0 && (
                  <Box>
                    {/* Reasoning Result */}
                    {reasoningPath.final_result && (
                      <Paper 
                        sx={{ 
                          p: 2, 
                          mb: 3, 
                          backgroundColor: alpha('#f5f8fc', 0.7),
                          border: '1px solid',
                          borderColor: 'primary.light',
                          borderRadius: 2,
                        }}
                      >
                        <HumanText variant="subtitle2" sx={{ mb: 1, fontWeight: 600, color: 'primary.main' }}>
                          Final Result:
                        </HumanText>
                        <HumanText variant="body1">
                          {reasoningPath.final_result}
                        </HumanText>
                      </Paper>
                    )}
                    
                    {/* Validation results if available */}
                    {validation && (
                      <Paper 
                        sx={{ 
                          p: 2, 
                          mb: 3, 
                          backgroundColor: alpha(
                            validation.score > 0.8 ? 'success.light' : 
                            validation.score > 0.6 ? 'warning.light' : 
                            'error.light', 
                            0.1
                          ),
                          border: '1px solid',
                          borderColor: validation.score > 0.8 ? 'success.main' : 
                                       validation.score > 0.6 ? 'warning.main' : 
                                       'error.main',
                          borderRadius: 2,
                        }}
                      >
                        <HumanText variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                          Validation Results: {(validation.score * 100).toFixed(0)}% Valid
                        </HumanText>
                        
                        {Object.entries(validation.results).map(([key, value]: [string, any]) => (
                          <Box key={key} sx={{ mb: 1 }}>
                            <HumanText variant="body2" sx={{ fontWeight: 600 }}>
                              {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}:
                            </HumanText>
                            <HumanText variant="body2" color="text.secondary">
                              {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </HumanText>
                          </Box>
                        ))}
                        
                        {validation.suggestions.length > 0 && (
                          <Box sx={{ mt: 2 }}>
                            <HumanText variant="body2" sx={{ fontWeight: 600 }}>
                              Suggestions:
                            </HumanText>
                            <List dense>
                              {validation.suggestions.map((suggestion, i) => (
                                <ListItem key={i}>
                                  <ListItemText
                                    primary={suggestion}
                                    primaryTypographyProps={{ variant: 'body2' }}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </Box>
                        )}
                      </Paper>
                    )}
                    
                    {/* Reasoning Steps */}
                    <HumanText variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
                      Reasoning Steps:
                    </HumanText>
                    {renderReasoningSteps()}
                    
                    {/* Feedback form */}
                    {renderFeedbackForm()}
                  </Box>
                )}
                
                {activeTab === 1 && renderTransformation()}
                {activeTab === 2 && renderMetrics()}
                {activeTab === 3 && renderFingerprint()}
              </Box>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', flexDirection: 'column' }}>
              <Box
                sx={{
                  width: 80,
                  height: 80,
                  borderRadius: '50%',
                  backgroundColor: alpha('#f5f8fc', 0.7),
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  mb: 3,
                }}
              >
                <LightbulbIcon 
                  sx={{ 
                    fontSize: 40,
                    color: alpha('#0066B3', 0.6),
                  }} 
                />
              </Box>
              <HumanText variant="h6" sx={{ fontWeight: 600, mb: 1, textAlign: 'center' }}>
                Track Reasoning Paths
              </HumanText>
              <HumanText variant="body2" color="text.secondary" sx={{ textAlign: 'center', maxWidth: 500 }}>
                Enter a query to explore how the system reasons with your data. See the step-by-step process
                and examine how information transforms into vector embeddings.
              </HumanText>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ReasoningVisualizer;