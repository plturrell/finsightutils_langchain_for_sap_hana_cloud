import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Divider,
  Chip,
  Paper,
  Grid,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Collapse,
  Slider,
  Alert,
  Tooltip,
  CircularProgress,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Transform as TransformIcon,
  Check as CheckIcon,
  AddCircleOutline as AddIcon,
  Delete as DeleteIcon,
  FilterList as FilterIcon,
  Functions as FunctionsIcon,
  TextFormat as TextFormatIcon,
  DateRange as DateIcon,
  Code as CodeIcon,
  Merge as MergeIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Info as InfoIcon,
  Save as SaveIcon,
  PlayArrow as PlayArrowIcon,
  ArrowForward as ArrowForwardIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import HumanText from './HumanText';

// Types for transformation configuration
interface TableColumn {
  name: string;
  dataType: string;
  isPrimaryKey: boolean;
  isNullable: boolean;
}

interface TransformationStep {
  id: string;
  type: 'filter' | 'join' | 'text' | 'numeric' | 'date' | 'merge' | 'custom';
  name: string;
  config: Record<string, any>;
  enabled: boolean;
}

interface TransformationConfig {
  sourceSchema: string;
  sourceTable: string;
  targetVectorTable: string;
  primaryKeyColumn: string;
  textColumns: string[];
  numericColumns: string[];
  dateColumns: string[];
  otherColumns: string[];
  steps: TransformationStep[];
  vectorDimensions: number;
  vectorModel: string;
  includeMetadata: boolean;
}

// Props for the TransformationConfigurator component
interface TransformationConfiguratorProps {
  sourceSchema?: string;
  sourceTable?: string;
  onComplete?: (config: TransformationConfig) => void;
}

const TransformationConfigurator: React.FC<TransformationConfiguratorProps> = ({
  sourceSchema = '',
  sourceTable = '',
  onComplete,
}) => {
  const theme = useTheme();
  
  // State for column information
  const [columns, setColumns] = useState<TableColumn[]>([]);
  const [isLoadingColumns, setIsLoadingColumns] = useState<boolean>(false);
  
  // State for transformation configuration
  const [config, setConfig] = useState<TransformationConfig>({
    sourceSchema,
    sourceTable,
    targetVectorTable: `${sourceTable}_VECTORS`.replace(/\s+/g, '_'),
    primaryKeyColumn: '',
    textColumns: [],
    numericColumns: [],
    dateColumns: [],
    otherColumns: [],
    steps: [],
    vectorDimensions: 384,
    vectorModel: 'all-MiniLM-L6-v2',
    includeMetadata: true,
  });
  
  // State for UI
  const [expandedStep, setExpandedStep] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  
  // Load columns for the source table
  const fetchColumns = async () => {
    try {
      setIsLoadingColumns(true);
      setError(null);
      
      // In a real implementation, this would call an API endpoint
      // For this demo, we'll use mock data
      
      // Mock data for CUSTOMERS table
      if (sourceTable === 'CUSTOMERS' && sourceSchema === 'CUSTOMER_DATA') {
        setColumns([
          { name: 'CUSTOMER_ID', dataType: 'NVARCHAR', isPrimaryKey: true, isNullable: false },
          { name: 'FIRST_NAME', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'LAST_NAME', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'EMAIL', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'PHONE', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: true },
          { name: 'BIRTH_DATE', dataType: 'DATE', isPrimaryKey: false, isNullable: true },
          { name: 'REGISTRATION_DATE', dataType: 'TIMESTAMP', isPrimaryKey: false, isNullable: false },
          { name: 'LAST_LOGIN', dataType: 'TIMESTAMP', isPrimaryKey: false, isNullable: true },
          { name: 'STATUS', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'CUSTOMER_TYPE', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'LOYALTY_POINTS', dataType: 'INTEGER', isPrimaryKey: false, isNullable: false },
          { name: 'NOTES', dataType: 'NCLOB', isPrimaryKey: false, isNullable: true },
          { name: 'PREFERENCES_JSON', dataType: 'NCLOB', isPrimaryKey: false, isNullable: true },
          { name: 'CREATED_BY', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
        ]);
      }
      // Mock data for ORDERS table
      else if (sourceTable === 'ORDERS' && sourceSchema === 'SALES') {
        setColumns([
          { name: 'ORDER_ID', dataType: 'NVARCHAR', isPrimaryKey: true, isNullable: false },
          { name: 'CUSTOMER_ID', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'ORDER_DATE', dataType: 'TIMESTAMP', isPrimaryKey: false, isNullable: false },
          { name: 'STATUS', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'TOTAL_AMOUNT', dataType: 'DECIMAL', isPrimaryKey: false, isNullable: false },
          { name: 'TAX_AMOUNT', dataType: 'DECIMAL', isPrimaryKey: false, isNullable: false },
          { name: 'SHIPPING_AMOUNT', dataType: 'DECIMAL', isPrimaryKey: false, isNullable: false },
          { name: 'SHIPPING_ADDRESS_ID', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'BILLING_ADDRESS_ID', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'PAYMENT_METHOD', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'NOTES', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: true },
          { name: 'CREATED_BY', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
        ]);
      }
      // Mock data for PRODUCT_CATALOG table
      else if (sourceTable === 'PRODUCT_CATALOG' && sourceSchema === 'PRODUCTS') {
        setColumns([
          { name: 'PRODUCT_ID', dataType: 'NVARCHAR', isPrimaryKey: true, isNullable: false },
          { name: 'SKU', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'NAME', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'DESCRIPTION', dataType: 'NCLOB', isPrimaryKey: false, isNullable: true },
          { name: 'CATEGORY_ID', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'PRICE', dataType: 'DECIMAL', isPrimaryKey: false, isNullable: false },
          { name: 'COST', dataType: 'DECIMAL', isPrimaryKey: false, isNullable: false },
          { name: 'WEIGHT', dataType: 'DECIMAL', isPrimaryKey: false, isNullable: true },
          { name: 'DIMENSIONS', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: true },
          { name: 'RELEASE_DATE', dataType: 'DATE', isPrimaryKey: false, isNullable: true },
          { name: 'AVAILABLE', dataType: 'BOOLEAN', isPrimaryKey: false, isNullable: false },
          { name: 'FEATURED', dataType: 'BOOLEAN', isPrimaryKey: false, isNullable: false },
          { name: 'RATING', dataType: 'DECIMAL', isPrimaryKey: false, isNullable: true },
          { name: 'REVIEW_COUNT', dataType: 'INTEGER', isPrimaryKey: false, isNullable: false },
          { name: 'TAGS', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: true },
          { name: 'CREATED_BY', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
        ]);
      }
      // Generic columns for other tables
      else {
        setColumns([
          { name: 'ID', dataType: 'NVARCHAR', isPrimaryKey: true, isNullable: false },
          { name: 'NAME', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: false },
          { name: 'DESCRIPTION', dataType: 'NVARCHAR', isPrimaryKey: false, isNullable: true },
          { name: 'CREATED_AT', dataType: 'TIMESTAMP', isPrimaryKey: false, isNullable: false },
          { name: 'UPDATED_AT', dataType: 'TIMESTAMP', isPrimaryKey: false, isNullable: true },
        ]);
      }
      
      // Initialize configuration based on columns
      initializeConfiguration();
    } catch (err: any) {
      console.error('Error fetching columns:', err);
      setError('Failed to fetch column information');
    } finally {
      setIsLoadingColumns(false);
    }
  };
  
  // Initialize configuration based on columns
  const initializeConfiguration = () => {
    // Find primary key column
    const primaryKey = columns.find(col => col.isPrimaryKey)?.name || columns[0]?.name || '';
    
    // Categorize columns by data type
    const textCols: string[] = [];
    const numericCols: string[] = [];
    const dateCols: string[] = [];
    const otherCols: string[] = [];
    
    columns.forEach(col => {
      if (col.dataType.includes('CHAR') || col.dataType.includes('CLOB')) {
        textCols.push(col.name);
      } else if (col.dataType.includes('INT') || col.dataType.includes('DEC') || col.dataType.includes('NUM')) {
        numericCols.push(col.name);
      } else if (col.dataType.includes('DATE') || col.dataType.includes('TIME')) {
        dateCols.push(col.name);
      } else {
        otherCols.push(col.name);
      }
    });
    
    // Update config
    setConfig(prevConfig => ({
      ...prevConfig,
      primaryKeyColumn: primaryKey,
      textColumns: textCols,
      numericColumns: numericCols,
      dateColumns: dateCols,
      otherColumns: otherCols,
    }));
    
    // Add default text transformation step if there are text columns
    if (textCols.length > 0) {
      addTransformationStep('text', 'Text Preprocessing');
    }
  };
  
  // Add a transformation step
  const addTransformationStep = (type: TransformationStep['type'], name: string) => {
    const newStep: TransformationStep = {
      id: `step-${Date.now()}`,
      type,
      name,
      config: getDefaultConfigForType(type),
      enabled: true,
    };
    
    setConfig(prevConfig => ({
      ...prevConfig,
      steps: [...prevConfig.steps, newStep],
    }));
    
    // Expand the newly added step
    setExpandedStep(newStep.id);
  };
  
  // Get default configuration for a step type
  const getDefaultConfigForType = (type: TransformationStep['type']): Record<string, any> => {
    switch (type) {
      case 'filter':
        return {
          condition: '',
          columns: [],
        };
      case 'join':
        return {
          joinTable: '',
          joinSchema: '',
          joinType: 'INNER',
          joinCondition: '',
          columnsToInclude: [],
        };
      case 'text':
        return {
          columns: config.textColumns,
          removeStopwords: true,
          lowercase: true,
          stemming: false,
          mergeStrategy: 'concatenate',
        };
      case 'numeric':
        return {
          columns: config.numericColumns,
          normalization: 'minmax',
          handleMissing: 'mean',
        };
      case 'date':
        return {
          columns: config.dateColumns,
          format: 'relative',
          reference: 'now',
        };
      case 'merge':
        return {
          columns: [],
          separator: ' ',
          targetColumn: 'MERGED_TEXT',
        };
      case 'custom':
        return {
          sql: '',
          description: '',
        };
      default:
        return {};
    }
  };
  
  // Toggle expanded step
  const toggleExpandStep = (stepId: string) => {
    setExpandedStep(expandedStep === stepId ? null : stepId);
  };
  
  // Update a step's configuration
  const updateStepConfig = (stepId: string, updatedConfig: Record<string, any>) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      steps: prevConfig.steps.map(step => 
        step.id === stepId 
          ? { ...step, config: { ...step.config, ...updatedConfig } } 
          : step
      ),
    }));
  };
  
  // Toggle a step's enabled state
  const toggleStepEnabled = (stepId: string) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      steps: prevConfig.steps.map(step => 
        step.id === stepId 
          ? { ...step, enabled: !step.enabled } 
          : step
      ),
    }));
  };
  
  // Remove a step
  const removeStep = (stepId: string) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      steps: prevConfig.steps.filter(step => step.id !== stepId),
    }));
    
    // If removing the expanded step, collapse it
    if (expandedStep === stepId) {
      setExpandedStep(null);
    }
  };
  
  // Save and apply the configuration
  const saveConfiguration = () => {
    try {
      setIsProcessing(true);
      setError(null);
      setSuccess(null);
      
      // In a real implementation, this would call an API endpoint
      // For this demo, we'll just simulate processing
      
      setTimeout(() => {
        setSuccess('Transformation configuration saved successfully');
        setIsProcessing(false);
        
        // Call the onComplete callback if provided
        if (onComplete) {
          onComplete(config);
        }
      }, 1500);
    } catch (err: any) {
      console.error('Error saving configuration:', err);
      setError('Failed to save transformation configuration');
      setIsProcessing(false);
    }
  };
  
  // Initialize data on component mount
  useEffect(() => {
    if (sourceSchema && sourceTable) {
      fetchColumns();
    }
  }, [sourceSchema, sourceTable]);
  
  // Render a step configuration form based on the step type
  const renderStepConfig = (step: TransformationStep) => {
    switch (step.type) {
      case 'filter':
        return (
          <Box>
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Filter Condition</HumanText>
            <TextField
              fullWidth
              placeholder="e.g. STATUS = 'ACTIVE' AND CREATED_AT > '2023-01-01'"
              value={step.config.condition}
              onChange={(e) => updateStepConfig(step.id, { condition: e.target.value })}
              size="small"
              sx={{ mb: 2 }}
            />
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Columns to Filter</HumanText>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Select columns</InputLabel>
              <Select
                multiple
                value={step.config.columns}
                onChange={(e) => updateStepConfig(step.id, { columns: e.target.value })}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {columns.map((column) => (
                  <MenuItem key={column.name} value={column.name}>
                    {column.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        );
      
      case 'text':
        return (
          <Box>
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Text Columns to Process</HumanText>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Select columns</InputLabel>
              <Select
                multiple
                value={step.config.columns}
                onChange={(e) => updateStepConfig(step.id, { columns: e.target.value })}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {config.textColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} sm={4}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={step.config.removeStopwords}
                      onChange={(e) => updateStepConfig(step.id, { removeStopwords: e.target.checked })}
                      size="small"
                    />
                  }
                  label="Remove Stopwords"
                />
              </Grid>
              <Grid item xs={12} sm={4}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={step.config.lowercase}
                      onChange={(e) => updateStepConfig(step.id, { lowercase: e.target.checked })}
                      size="small"
                    />
                  }
                  label="Convert to Lowercase"
                />
              </Grid>
              <Grid item xs={12} sm={4}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={step.config.stemming}
                      onChange={(e) => updateStepConfig(step.id, { stemming: e.target.checked })}
                      size="small"
                    />
                  }
                  label="Apply Stemming"
                />
              </Grid>
            </Grid>
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Merge Strategy</HumanText>
            <FormControl fullWidth size="small">
              <InputLabel>Strategy</InputLabel>
              <Select
                value={step.config.mergeStrategy}
                onChange={(e) => updateStepConfig(step.id, { mergeStrategy: e.target.value })}
              >
                <MenuItem value="concatenate">Concatenate all text columns</MenuItem>
                <MenuItem value="separate">Keep columns separate</MenuItem>
                <MenuItem value="weighted">Weighted concatenation</MenuItem>
              </Select>
            </FormControl>
          </Box>
        );
      
      case 'numeric':
        return (
          <Box>
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Numeric Columns to Process</HumanText>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Select columns</InputLabel>
              <Select
                multiple
                value={step.config.columns}
                onChange={(e) => updateStepConfig(step.id, { columns: e.target.value })}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {config.numericColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Normalization</HumanText>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Method</InputLabel>
              <Select
                value={step.config.normalization}
                onChange={(e) => updateStepConfig(step.id, { normalization: e.target.value })}
              >
                <MenuItem value="none">None</MenuItem>
                <MenuItem value="minmax">Min-Max Scaling</MenuItem>
                <MenuItem value="zscore">Z-Score Normalization</MenuItem>
                <MenuItem value="log">Logarithmic Scaling</MenuItem>
              </Select>
            </FormControl>
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Handle Missing Values</HumanText>
            <FormControl fullWidth size="small">
              <InputLabel>Method</InputLabel>
              <Select
                value={step.config.handleMissing}
                onChange={(e) => updateStepConfig(step.id, { handleMissing: e.target.value })}
              >
                <MenuItem value="zero">Replace with zero</MenuItem>
                <MenuItem value="mean">Replace with mean</MenuItem>
                <MenuItem value="median">Replace with median</MenuItem>
                <MenuItem value="drop">Drop rows with missing values</MenuItem>
              </Select>
            </FormControl>
          </Box>
        );
      
      case 'date':
        return (
          <Box>
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Date/Time Columns to Process</HumanText>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Select columns</InputLabel>
              <Select
                multiple
                value={step.config.columns}
                onChange={(e) => updateStepConfig(step.id, { columns: e.target.value })}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {config.dateColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Format</HumanText>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Format</InputLabel>
              <Select
                value={step.config.format}
                onChange={(e) => updateStepConfig(step.id, { format: e.target.value })}
              >
                <MenuItem value="iso">ISO Format (YYYY-MM-DD)</MenuItem>
                <MenuItem value="relative">Relative to reference date</MenuItem>
                <MenuItem value="extract">Extract components (year, month, day)</MenuItem>
                <MenuItem value="categorical">Convert to categorical features</MenuItem>
              </Select>
            </FormControl>
            
            {step.config.format === 'relative' && (
              <>
                <HumanText variant="subtitle2" sx={{ mb: 1 }}>Reference Date</HumanText>
                <FormControl fullWidth size="small">
                  <InputLabel>Reference</InputLabel>
                  <Select
                    value={step.config.reference}
                    onChange={(e) => updateStepConfig(step.id, { reference: e.target.value })}
                  >
                    <MenuItem value="now">Current date</MenuItem>
                    <MenuItem value="custom">Custom date</MenuItem>
                    <MenuItem value="earliest">Earliest date in data</MenuItem>
                    <MenuItem value="latest">Latest date in data</MenuItem>
                  </Select>
                </FormControl>
              </>
            )}
          </Box>
        );
      
      case 'merge':
        return (
          <Box>
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Columns to Merge</HumanText>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Select columns</InputLabel>
              <Select
                multiple
                value={step.config.columns}
                onChange={(e) => updateStepConfig(step.id, { columns: e.target.value })}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {columns.map((column) => (
                  <MenuItem key={column.name} value={column.name}>
                    {column.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Separator</HumanText>
            <TextField
              fullWidth
              placeholder="Space, comma, or other separator"
              value={step.config.separator}
              onChange={(e) => updateStepConfig(step.id, { separator: e.target.value })}
              size="small"
              sx={{ mb: 2 }}
            />
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Target Column Name</HumanText>
            <TextField
              fullWidth
              placeholder="Name for the new merged column"
              value={step.config.targetColumn}
              onChange={(e) => updateStepConfig(step.id, { targetColumn: e.target.value })}
              size="small"
            />
          </Box>
        );
      
      case 'custom':
        return (
          <Box>
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Custom SQL</HumanText>
            <TextField
              fullWidth
              multiline
              rows={4}
              placeholder="Enter custom SQL transformation"
              value={step.config.sql}
              onChange={(e) => updateStepConfig(step.id, { sql: e.target.value })}
              size="small"
              sx={{ mb: 2, fontFamily: 'monospace' }}
            />
            
            <HumanText variant="subtitle2" sx={{ mb: 1 }}>Description</HumanText>
            <TextField
              fullWidth
              placeholder="Describe what this custom transformation does"
              value={step.config.description}
              onChange={(e) => updateStepConfig(step.id, { description: e.target.value })}
              size="small"
            />
          </Box>
        );
      
      default:
        return (
          <Box>
            <Alert severity="warning">
              Configuration not available for this step type.
            </Alert>
          </Box>
        );
    }
  };
  
  // Get icon for a step type
  const getStepIcon = (type: TransformationStep['type']) => {
    switch (type) {
      case 'filter':
        return <FilterIcon />;
      case 'join':
        return <MergeIcon />;
      case 'text':
        return <TextFormatIcon />;
      case 'numeric':
        return <FunctionsIcon />;
      case 'date':
        return <DateIcon />;
      case 'merge':
        return <MergeIcon />;
      case 'custom':
        return <CodeIcon />;
      default:
        return <TransformIcon />;
    }
  };
  
  // Get color for a step type
  const getStepColor = (type: TransformationStep['type']) => {
    switch (type) {
      case 'filter':
        return theme.palette.info.main;
      case 'join':
        return theme.palette.success.main;
      case 'text':
        return theme.palette.primary.main;
      case 'numeric':
        return theme.palette.warning.main;
      case 'date':
        return theme.palette.secondary.main;
      case 'merge':
        return theme.palette.success.main;
      case 'custom':
        return theme.palette.error.main;
      default:
        return theme.palette.text.primary;
    }
  };
  
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
      <CardContent sx={{ p: 0, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
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
              Configure Transformation Pipeline
            </HumanText>
            <HumanText variant="body2" color="text.secondary">
              Define how data from {sourceSchema}.{sourceTable} will be transformed for vectorization
            </HumanText>
          </Box>
          
          <Button
            variant="contained"
            color="primary"
            onClick={saveConfiguration}
            disabled={isProcessing || isLoadingColumns}
            startIcon={isProcessing ? <CircularProgress size={20} /> : <SaveIcon />}
          >
            Save & Continue
          </Button>
        </Box>
        
        {/* Messages */}
        {error && (
          <Box sx={{ p: 2 }}>
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          </Box>
        )}
        
        {success && (
          <Box sx={{ p: 2 }}>
            <Alert severity="success" onClose={() => setSuccess(null)}>
              {success}
            </Alert>
          </Box>
        )}
        
        {/* Main content */}
        {isLoadingColumns ? (
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              p: 3,
            }}
          >
            <CircularProgress />
          </Box>
        ) : (
          <Box sx={{ flexGrow: 1, display: 'flex', overflow: 'hidden' }}>
            {/* Left panel - Transformation steps */}
            <Box
              sx={{
                width: '65%',
                borderRight: '1px solid',
                borderColor: 'divider',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
              }}
            >
              <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
                <HumanText variant="subtitle1" sx={{ mb: 1, fontWeight: 500 }}>
                  Transformation Steps
                </HumanText>
                <HumanText variant="body2" color="text.secondary">
                  Add and configure steps to prepare your data for vectorization
                </HumanText>
              </Box>
              
              <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                {/* Steps list */}
                {config.steps.length === 0 ? (
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      height: 200,
                      border: '1px dashed',
                      borderColor: 'divider',
                      borderRadius: 1,
                      p: 3,
                    }}
                  >
                    <Box sx={{ textAlign: 'center' }}>
                      <TransformIcon sx={{ fontSize: 40, color: 'text.secondary', opacity: 0.5, mb: 2 }} />
                      <HumanText variant="subtitle1" sx={{ mb: 1 }}>
                        No Transformation Steps
                      </HumanText>
                      <HumanText variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Add steps to transform your data before vectorization
                      </HumanText>
                      <Button
                        variant="outlined"
                        color="primary"
                        startIcon={<AddIcon />}
                        onClick={() => addTransformationStep('text', 'Text Preprocessing')}
                      >
                        Add Step
                      </Button>
                    </Box>
                  </Box>
                ) : (
                  <List sx={{ mb: 3 }}>
                    {config.steps.map((step, index) => (
                      <Paper
                        key={step.id}
                        sx={{
                          mb: 2,
                          borderRadius: 1,
                          border: '1px solid',
                          borderColor: step.enabled ? 'divider' : 'divider',
                          opacity: step.enabled ? 1 : 0.7,
                          overflow: 'hidden',
                          boxShadow: expandedStep === step.id ? 2 : 0,
                        }}
                      >
                        <ListItem
                          button
                          onClick={() => toggleExpandStep(step.id)}
                          sx={{
                            borderLeft: '4px solid',
                            borderColor: step.enabled ? getStepColor(step.type) : 'divider',
                            transition: 'background-color 0.2s',
                            bgcolor: expandedStep === step.id ? alpha(getStepColor(step.type), 0.05) : 'transparent',
                          }}
                        >
                          <ListItemIcon sx={{ color: getStepColor(step.type) }}>
                            {getStepIcon(step.type)}
                          </ListItemIcon>
                          <ListItemText
                            primary={`${index + 1}. ${step.name}`}
                            secondary={step.type.charAt(0).toUpperCase() + step.type.slice(1)}
                            primaryTypographyProps={{ fontWeight: expandedStep === step.id ? 600 : 400 }}
                          />
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Tooltip title={step.enabled ? 'Disable step' : 'Enable step'}>
                              <Switch
                                checked={step.enabled}
                                onChange={(e) => {
                                  e.stopPropagation();
                                  toggleStepEnabled(step.id);
                                }}
                                size="small"
                                color="primary"
                                sx={{ mr: 1 }}
                              />
                            </Tooltip>
                            <Tooltip title="Remove step">
                              <IconButton
                                size="small"
                                color="error"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  removeStep(step.id);
                                }}
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                            {expandedStep === step.id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                          </Box>
                        </ListItem>
                        
                        {/* Step configuration */}
                        <Collapse in={expandedStep === step.id} timeout="auto">
                          <Box sx={{ p: 2, bgcolor: alpha(getStepColor(step.type), 0.03) }}>
                            {renderStepConfig(step)}
                          </Box>
                        </Collapse>
                      </Paper>
                    ))}
                  </List>
                )}
                
                {/* Add step button */}
                <Box sx={{ mt: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6} md={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        color="primary"
                        startIcon={<TextFormatIcon />}
                        onClick={() => addTransformationStep('text', 'Text Preprocessing')}
                        sx={{ justifyContent: 'flex-start' }}
                      >
                        Add Text Step
                      </Button>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        color="info"
                        startIcon={<FilterIcon />}
                        onClick={() => addTransformationStep('filter', 'Filter Data')}
                        sx={{ justifyContent: 'flex-start' }}
                      >
                        Add Filter
                      </Button>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        color="warning"
                        startIcon={<FunctionsIcon />}
                        onClick={() => addTransformationStep('numeric', 'Numeric Preprocessing')}
                        sx={{ justifyContent: 'flex-start' }}
                      >
                        Add Numeric
                      </Button>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        color="secondary"
                        startIcon={<DateIcon />}
                        onClick={() => addTransformationStep('date', 'Date Preprocessing')}
                        sx={{ justifyContent: 'flex-start' }}
                      >
                        Add Date
                      </Button>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        color="success"
                        startIcon={<MergeIcon />}
                        onClick={() => addTransformationStep('merge', 'Merge Columns')}
                        sx={{ justifyContent: 'flex-start' }}
                      >
                        Add Merge
                      </Button>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        color="error"
                        startIcon={<CodeIcon />}
                        onClick={() => addTransformationStep('custom', 'Custom Transformation')}
                        sx={{ justifyContent: 'flex-start' }}
                      >
                        Add Custom
                      </Button>
                    </Grid>
                  </Grid>
                </Box>
              </Box>
            </Box>
            
            {/* Right panel - Vector settings */}
            <Box
              sx={{
                width: '35%',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
              }}
            >
              <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
                <HumanText variant="subtitle1" sx={{ mb: 1, fontWeight: 500 }}>
                  Vectorization Settings
                </HumanText>
                <HumanText variant="body2" color="text.secondary">
                  Configure how data will be converted to vectors
                </HumanText>
              </Box>
              
              <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                {/* Basic settings */}
                <Paper sx={{ p: 2, mb: 3, borderRadius: 1 }}>
                  <HumanText variant="subtitle2" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                    <SettingsIcon fontSize="small" sx={{ mr: 1 }} />
                    Basic Settings
                  </HumanText>
                  
                  <HumanText variant="body2" sx={{ mb: 1 }}>Target Vector Table</HumanText>
                  <TextField
                    fullWidth
                    size="small"
                    value={config.targetVectorTable}
                    onChange={(e) => setConfig({ ...config, targetVectorTable: e.target.value })}
                    sx={{ mb: 2 }}
                  />
                  
                  <HumanText variant="body2" sx={{ mb: 1 }}>Primary Key Column</HumanText>
                  <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                    <InputLabel>Primary Key</InputLabel>
                    <Select
                      value={config.primaryKeyColumn}
                      onChange={(e) => setConfig({ ...config, primaryKeyColumn: e.target.value as string })}
                    >
                      {columns.map((column) => (
                        <MenuItem key={column.name} value={column.name}>
                          {column.name} {column.isPrimaryKey && ' (PK)'}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.includeMetadata}
                        onChange={(e) => setConfig({ ...config, includeMetadata: e.target.checked })}
                        size="small"
                      />
                    }
                    label="Include metadata in vector store"
                  />
                </Paper>
                
                {/* Vector model settings */}
                <Paper sx={{ p: 2, mb: 3, borderRadius: 1 }}>
                  <HumanText variant="subtitle2" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                    <ModelTrainingIcon fontSize="small" sx={{ mr: 1 }} />
                    Vector Model
                  </HumanText>
                  
                  <HumanText variant="body2" sx={{ mb: 1 }}>Embedding Model</HumanText>
                  <FormControl fullWidth size="small" sx={{ mb: 3 }}>
                    <InputLabel>Model</InputLabel>
                    <Select
                      value={config.vectorModel}
                      onChange={(e) => setConfig({ ...config, vectorModel: e.target.value as string })}
                    >
                      <MenuItem value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (384 dimensions)</MenuItem>
                      <MenuItem value="all-mpnet-base-v2">all-mpnet-base-v2 (768 dimensions)</MenuItem>
                      <MenuItem value="all-distilroberta-v1">all-distilroberta-v1 (768 dimensions)</MenuItem>
                      <MenuItem value="multi-qa-MiniLM-L6-dot-v1">multi-qa-MiniLM-L6-dot-v1 (384 dimensions)</MenuItem>
                      <MenuItem value="hana-internal">SAP HANA internal embedding model</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <HumanText variant="body2" sx={{ mb: 1 }}>
                    Vector Dimensions: {config.vectorDimensions}
                  </HumanText>
                  <Slider
                    value={config.vectorDimensions}
                    onChange={(_event, newValue) => setConfig({ ...config, vectorDimensions: newValue as number })}
                    min={128}
                    max={1024}
                    step={128}
                    marks={[
                      { value: 128, label: '128' },
                      { value: 384, label: '384' },
                      { value: 768, label: '768' },
                      { value: 1024, label: '1024' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Paper>
                
                {/* Advanced settings */}
                <Box sx={{ mb: 3 }}>
                  <Button
                    startIcon={showAdvanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    variant="text"
                  >
                    {showAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
                  </Button>
                  
                  <Collapse in={showAdvanced}>
                    <Paper sx={{ p: 2, mt: 2, borderRadius: 1 }}>
                      <HumanText variant="subtitle2" sx={{ mb: 2 }}>Advanced Settings</HumanText>
                      
                      <Alert severity="info" sx={{ mb: 2 }}>
                        Advanced settings allow fine-tuning of the vectorization process but are not required for most use cases.
                      </Alert>
                      
                      <HumanText variant="body2" sx={{ mb: 1 }}>Additional options coming soon...</HumanText>
                    </Paper>
                  </Collapse>
                </Box>
                
                {/* Preview and save */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<PlayArrowIcon />}
                    disabled={true} // Disabled in this demo
                  >
                    Preview
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={saveConfiguration}
                    disabled={isProcessing}
                    startIcon={isProcessing ? <CircularProgress size={20} /> : <ArrowForwardIcon />}
                    endIcon={<CheckIcon />}
                  >
                    Apply & Continue
                  </Button>
                </Box>
              </Box>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default TransformationConfigurator;