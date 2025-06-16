import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  Divider,
  Tabs,
  Tab,
  TextField,
  IconButton,
  Chip,
  CircularProgress,
  Alert,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Database as DatabaseIcon,
  Storage as StorageIcon,
  Transform as TransformIcon,
  ModelTraining as ModelTrainingIcon,
  Layers as LayersIcon,
  ArrowForward as ArrowForwardIcon,
  ArrowBack as ArrowBackIcon,
  Loop as LoopIcon,
  Code as CodeIcon,
  Add as AddIcon,
  Search as SearchIcon,
  MoreVert as MoreVertIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Visibility as VisibilityIcon,
  ViewInAr as ViewInArIcon,
  Info as InfoIcon,
  ShowChart as ShowChartIcon,
  Settings as SettingsIcon,
  Edit as EditIcon,
  DeleteOutline as DeleteOutlineIcon,
} from '@mui/icons-material';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  ConnectionLineType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  dataPipelineService,
  CreatePipelineRequest,
  RegisterDataSourceRequest,
  RegisterIntermediateStageRequest,
  RegisterVectorRequest,
  RegisterTransformationRuleRequest,
  GetPipelineRequest,
  GetDataLineageRequest,
  GetReverseMapRequest,
} from '../api/services';
import HumanText from './HumanText';

// Node types for ReactFlow
const nodeTypes = {
  dataSource: DataSourceNode,
  intermediate: IntermediateNode,
  vector: VectorNode,
  rule: RuleNode,
};

// Props for the DataPipelineVisualizer component
interface DataPipelineVisualizerProps {
  defaultSchemaName?: string;
  defaultTableName?: string;
  initialPipelineId?: string;
}

// Custom Node Components
function DataSourceNode({ data }) {
  return (
    <Box
      sx={{
        background: 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)',
        border: '1px solid #90caf9',
        borderRadius: 2,
        padding: 2,
        minWidth: 180,
        maxWidth: 220,
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <DatabaseIcon color="primary" sx={{ mr: 1 }} />
        <HumanText variant="subtitle1" sx={{ fontWeight: 600 }}>
          {data.label}
        </HumanText>
      </Box>
      <Divider sx={{ my: 1 }} />
      <Box sx={{ fontSize: '0.75rem' }}>
        {data.tableInfo && (
          <>
            <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
              Schema: {data.tableInfo.schema}
            </HumanText>
            <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
              Table: {data.tableInfo.table}
            </HumanText>
            <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
              Rows: {data.tableInfo.rows}
            </HumanText>
          </>
        )}
      </Box>
      {data.onView && (
        <Button
          size="small"
          startIcon={<VisibilityIcon fontSize="small" />}
          onClick={data.onView}
          sx={{ mt: 1, width: '100%' }}
        >
          View Details
        </Button>
      )}
    </Box>
  );
}

function IntermediateNode({ data }) {
  return (
    <Box
      sx={{
        background: 'linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%)',
        border: '1px solid #ce93d8',
        borderRadius: 2,
        padding: 2,
        minWidth: 180,
        maxWidth: 220,
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <TransformIcon color="secondary" sx={{ mr: 1 }} />
        <HumanText variant="subtitle1" sx={{ fontWeight: 600 }}>
          {data.label}
        </HumanText>
      </Box>
      <Divider sx={{ my: 1 }} />
      <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
        {data.description || 'Intermediate transformation stage'}
      </HumanText>
      {data.onView && (
        <Button
          size="small"
          startIcon={<VisibilityIcon fontSize="small" />}
          onClick={data.onView}
          sx={{ mt: 1, width: '100%' }}
        >
          View Details
        </Button>
      )}
    </Box>
  );
}

function VectorNode({ data }) {
  return (
    <Box
      sx={{
        background: 'linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)',
        border: '1px solid #a5d6a7',
        borderRadius: 2,
        padding: 2,
        minWidth: 180,
        maxWidth: 220,
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <ModelTrainingIcon color="success" sx={{ mr: 1 }} />
        <HumanText variant="subtitle1" sx={{ fontWeight: 600 }}>
          {data.label}
        </HumanText>
      </Box>
      <Divider sx={{ my: 1 }} />
      <Box sx={{ fontSize: '0.75rem' }}>
        {data.vectorInfo && (
          <>
            <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
              Model: {data.vectorInfo.model}
            </HumanText>
            <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
              Dimensions: {data.vectorInfo.dimensions}
            </HumanText>
          </>
        )}
      </Box>
      <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
        {data.onView && (
          <Button
            size="small"
            startIcon={<VisibilityIcon fontSize="small" />}
            onClick={data.onView}
            sx={{ flex: 1 }}
          >
            View
          </Button>
        )}
        {data.onLineage && (
          <Button
            size="small"
            color="secondary"
            startIcon={<ShowChartIcon fontSize="small" />}
            onClick={data.onLineage}
            sx={{ flex: 1 }}
          >
            Lineage
          </Button>
        )}
      </Box>
    </Box>
  );
}

function RuleNode({ data }) {
  return (
    <Box
      sx={{
        background: 'linear-gradient(135deg, #fff8e1 0%, #ffe0b2 100%)',
        border: '1px solid #ffcc80',
        borderRadius: 2,
        padding: 2,
        minWidth: 180,
        maxWidth: 220,
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <CodeIcon color="warning" sx={{ mr: 1 }} />
        <HumanText variant="subtitle1" sx={{ fontWeight: 600 }}>
          {data.label}
        </HumanText>
      </Box>
      <Divider sx={{ my: 1 }} />
      <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
        {data.description || 'Transformation rule'}
      </HumanText>
      <HumanText variant="caption" sx={{ display: 'block', color: 'text.secondary', mt: 0.5 }}>
        Type: {data.ruleType || 'Unknown'}
      </HumanText>
      {data.onView && (
        <Button
          size="small"
          startIcon={<VisibilityIcon fontSize="small" />}
          onClick={data.onView}
          sx={{ mt: 1, width: '100%' }}
        >
          View Details
        </Button>
      )}
    </Box>
  );
}

// Main Component
const DataPipelineVisualizer: React.FC<DataPipelineVisualizerProps> = ({
  defaultSchemaName = 'SYSTEM',
  defaultTableName = 'TABLES',
  initialPipelineId,
}) => {
  const theme = useTheme();
  
  // State for pipeline
  const [pipelineId, setPipelineId] = useState<string | null>(initialPipelineId || null);
  const [isCreatingPipeline, setIsCreatingPipeline] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // State for data sources
  const [dataSources, setDataSources] = useState<Record<string, any>>({});
  const [selectedDataSource, setSelectedDataSource] = useState<string | null>(null);
  
  // State for intermediate stages
  const [intermediateStages, setIntermediateStages] = useState<Record<string, any>>({});
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  
  // State for vectors
  const [vectors, setVectors] = useState<Record<string, any>>({});
  const [selectedVector, setSelectedVector] = useState<string | null>(null);
  
  // State for transformation rules
  const [transformationRules, setTransformationRules] = useState<Record<string, any>>({});
  const [selectedRule, setSelectedRule] = useState<string | null>(null);
  
  // State for flow visualization
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  // State for lineage and reverse mapping
  const [lineageData, setLineageData] = useState<Record<string, any> | null>(null);
  const [reverseMapData, setReverseMapData] = useState<Record<string, any> | null>(null);
  
  // State for forms
  const [showDataSourceForm, setShowDataSourceForm] = useState<boolean>(false);
  const [showIntermediateForm, setShowIntermediateForm] = useState<boolean>(false);
  const [showVectorForm, setShowVectorForm] = useState<boolean>(false);
  const [showRuleForm, setShowRuleForm] = useState<boolean>(false);
  
  // Form state for data source
  const [schemaName, setSchemaName] = useState<string>(defaultSchemaName);
  const [tableName, setTableName] = useState<string>(defaultTableName);
  const [sampleSize, setSampleSize] = useState<number>(5);
  
  // Form state for intermediate stage
  const [stageName, setStageName] = useState<string>('');
  const [stageDescription, setStageDescription] = useState<string>('');
  const [sourceId, setSourceId] = useState<string>('');
  const [columnMapping, setColumnMapping] = useState<Record<string, string[]>>({});
  
  // Form state for vector
  const [vectorModelName, setVectorModelName] = useState<string>('all-MiniLM-L6-v2');
  const [vectorDimensions, setVectorDimensions] = useState<number>(384);
  const [originalText, setOriginalText] = useState<string>('');
  
  // Form state for transformation rule
  const [ruleName, setRuleName] = useState<string>('');
  const [ruleDescription, setRuleDescription] = useState<string>('');
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [outputColumns, setOutputColumns] = useState<string[]>([]);
  const [transformationType, setTransformationType] = useState<string>('join');
  const [transformationParams, setTransformationParams] = useState<Record<string, any>>({});
  
  // State for tabs
  const [activeTab, setActiveTab] = useState<number>(0);
  
  // State for loading indicators
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isLineageLoading, setIsLineageLoading] = useState<boolean>(false);
  const [isReverseMapLoading, setIsReverseMapLoading] = useState<boolean>(false);
  
  // Create a new pipeline
  const createPipeline = async () => {
    try {
      setIsCreatingPipeline(true);
      setError(null);
      
      const request: CreatePipelineRequest = {};
      const response = await dataPipelineService.createPipeline(request);
      
      setPipelineId(response.data.pipeline_id);
    } catch (err: any) {
      console.error('Error creating pipeline:', err);
      setError(err.response?.data?.message || 'Failed to create pipeline');
    } finally {
      setIsCreatingPipeline(false);
    }
  };
  
  // Load pipeline data
  const loadPipelineData = async (pid: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const request: GetPipelineRequest = {
        pipeline_id: pid,
      };
      
      const response = await dataPipelineService.getPipeline(request);
      const data = response.data;
      
      // Set state with pipeline data
      setDataSources(data.data_sources || {});
      setIntermediateStages(data.intermediate_stages || {});
      setVectors(data.vector_representations || {});
      setTransformationRules(data.transformation_rules || {});
      
      // Build flow visualization
      buildFlowVisualization(
        data.data_sources || {},
        data.intermediate_stages || {},
        data.vector_representations || {},
        data.transformation_rules || {}
      );
    } catch (err: any) {
      console.error('Error loading pipeline data:', err);
      setError(err.response?.data?.message || 'Failed to load pipeline data');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Register a data source
  const registerDataSource = async () => {
    if (!pipelineId) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const request: RegisterDataSourceRequest = {
        pipeline_id: pipelineId,
        schema_name: schemaName,
        table_name: tableName,
        include_sample: true,
        sample_size: sampleSize,
      };
      
      const response = await dataPipelineService.registerDataSource(request);
      
      // Reset form
      setShowDataSourceForm(false);
      
      // Reload pipeline data
      await loadPipelineData(pipelineId);
    } catch (err: any) {
      console.error('Error registering data source:', err);
      setError(err.response?.data?.message || 'Failed to register data source');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Register an intermediate stage
  const registerIntermediateStage = async () => {
    if (!pipelineId) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const request: RegisterIntermediateStageRequest = {
        pipeline_id: pipelineId,
        stage_name: stageName,
        stage_description: stageDescription,
        source_id: sourceId,
        column_mapping: columnMapping,
      };
      
      const response = await dataPipelineService.registerIntermediateStage(request);
      
      // Reset form
      setShowIntermediateForm(false);
      setStageName('');
      setStageDescription('');
      setSourceId('');
      setColumnMapping({});
      
      // Reload pipeline data
      await loadPipelineData(pipelineId);
    } catch (err: any) {
      console.error('Error registering intermediate stage:', err);
      setError(err.response?.data?.message || 'Failed to register intermediate stage');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Register a vector
  const registerVector = async () => {
    if (!pipelineId) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const request: RegisterVectorRequest = {
        pipeline_id: pipelineId,
        source_id: sourceId,
        model_name: vectorModelName,
        vector_dimensions: vectorDimensions,
        original_text: originalText || undefined,
      };
      
      const response = await dataPipelineService.registerVector(request);
      
      // Reset form
      setShowVectorForm(false);
      setSourceId('');
      setVectorModelName('all-MiniLM-L6-v2');
      setVectorDimensions(384);
      setOriginalText('');
      
      // Reload pipeline data
      await loadPipelineData(pipelineId);
    } catch (err: any) {
      console.error('Error registering vector:', err);
      setError(err.response?.data?.message || 'Failed to register vector');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Register a transformation rule
  const registerTransformationRule = async () => {
    if (!pipelineId) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const request: RegisterTransformationRuleRequest = {
        pipeline_id: pipelineId,
        rule_name: ruleName,
        rule_description: ruleDescription,
        input_columns: inputColumns,
        output_columns: outputColumns,
        transformation_type: transformationType,
        transformation_params: transformationParams,
      };
      
      const response = await dataPipelineService.registerTransformationRule(request);
      
      // Reset form
      setShowRuleForm(false);
      setRuleName('');
      setRuleDescription('');
      setInputColumns([]);
      setOutputColumns([]);
      setTransformationType('join');
      setTransformationParams({});
      
      // Reload pipeline data
      await loadPipelineData(pipelineId);
    } catch (err: any) {
      console.error('Error registering transformation rule:', err);
      setError(err.response?.data?.message || 'Failed to register transformation rule');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Get data lineage for a vector
  const getDataLineage = async (vectorId: string) => {
    if (!pipelineId) return;
    
    try {
      setIsLineageLoading(true);
      setError(null);
      
      const request: GetDataLineageRequest = {
        pipeline_id: pipelineId,
        vector_id: vectorId,
      };
      
      const response = await dataPipelineService.getDataLineage(request);
      
      setLineageData(response.data);
      setActiveTab(1); // Switch to lineage tab
    } catch (err: any) {
      console.error('Error getting data lineage:', err);
      setError(err.response?.data?.message || 'Failed to get data lineage');
    } finally {
      setIsLineageLoading(false);
    }
  };
  
  // Get reverse mapping for a vector
  const getReverseMap = async (vectorId: string) => {
    if (!pipelineId) return;
    
    try {
      setIsReverseMapLoading(true);
      setError(null);
      
      const request: GetReverseMapRequest = {
        pipeline_id: pipelineId,
        vector_id: vectorId,
        similarity_threshold: 0.8,
      };
      
      const response = await dataPipelineService.getReverseMap(request);
      
      setReverseMapData(response.data);
      setActiveTab(2); // Switch to reverse map tab
    } catch (err: any) {
      console.error('Error getting reverse mapping:', err);
      setError(err.response?.data?.message || 'Failed to get reverse mapping');
    } finally {
      setIsReverseMapLoading(false);
    }
  };
  
  // Build flow visualization
  const buildFlowVisualization = (
    sources: Record<string, any>,
    stages: Record<string, any>,
    vectorReps: Record<string, any>,
    rules: Record<string, any>
  ) => {
    const newNodes: Node[] = [];
    const newEdges: Edge[] = [];
    
    // Layout configuration
    let sourceX = 50;
    let sourceY = 50;
    const sourceSpacing = 150;
    
    let intermediateX = 350;
    let intermediateY = 50;
    const intermediateSpacing = 150;
    
    let vectorX = 650;
    let vectorY = 50;
    const vectorSpacing = 150;
    
    let ruleX = 350;
    let ruleY = 400;
    const ruleSpacing = 200;
    
    // Add data source nodes
    Object.entries(sources).forEach(([id, source], index) => {
      newNodes.push({
        id: `source-${id}`,
        type: 'dataSource',
        position: { x: sourceX, y: sourceY + index * sourceSpacing },
        data: {
          label: `${source.schema_name}.${source.table_name}`,
          tableInfo: {
            schema: source.schema_name,
            table: source.table_name,
            rows: source.row_count,
          },
          onView: () => setSelectedDataSource(id),
        },
      });
    });
    
    // Add intermediate stage nodes
    Object.entries(stages).forEach(([id, stage], index) => {
      newNodes.push({
        id: `stage-${id}`,
        type: 'intermediate',
        position: { x: intermediateX, y: intermediateY + index * intermediateSpacing },
        data: {
          label: stage.stage_name,
          description: stage.stage_description,
          onView: () => setSelectedStage(id),
        },
      });
      
      // Add edge from source to stage
      newEdges.push({
        id: `edge-source-stage-${id}`,
        source: `source-${stage.source_id}`,
        target: `stage-${id}`,
        animated: true,
        style: { stroke: theme.palette.primary.main },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: theme.palette.primary.main,
        },
      });
    });
    
    // Add vector nodes
    Object.entries(vectorReps).forEach(([id, vector], index) => {
      newNodes.push({
        id: `vector-${id}`,
        type: 'vector',
        position: { x: vectorX, y: vectorY + index * vectorSpacing },
        data: {
          label: `Vector: ${vector.model_name}`,
          vectorInfo: {
            model: vector.model_name,
            dimensions: vector.vector_dimensions,
          },
          onView: () => setSelectedVector(id),
          onLineage: () => getDataLineage(id),
        },
      });
      
      // Add edge from intermediate to vector if there's a connection
      const sourceStage = Object.values(stages).find(
        (stage: any) => stage.source_id === vector.source_id
      );
      
      if (sourceStage) {
        newEdges.push({
          id: `edge-stage-vector-${id}`,
          source: `stage-${sourceStage.stage_id}`,
          target: `vector-${id}`,
          animated: true,
          style: { stroke: theme.palette.success.main },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: theme.palette.success.main,
          },
        });
      } else {
        // Direct edge from source to vector
        newEdges.push({
          id: `edge-source-vector-${id}`,
          source: `source-${vector.source_id}`,
          target: `vector-${id}`,
          animated: true,
          style: { stroke: theme.palette.success.main },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: theme.palette.success.main,
          },
        });
      }
    });
    
    // Add transformation rule nodes
    Object.entries(rules).forEach(([id, rule], index) => {
      newNodes.push({
        id: `rule-${id}`,
        type: 'rule',
        position: { x: ruleX + index * ruleSpacing, y: ruleY },
        data: {
          label: rule.rule_name,
          description: rule.rule_description,
          ruleType: rule.transformation_type,
          onView: () => setSelectedRule(id),
        },
      });
    });
    
    setNodes(newNodes);
    setEdges(newEdges);
  };
  
  // Handle tab change
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  // Initialize pipeline or load existing one
  useEffect(() => {
    if (initialPipelineId) {
      setPipelineId(initialPipelineId);
      loadPipelineData(initialPipelineId);
    }
  }, [initialPipelineId]);
  
  // Load pipeline data when pipeline ID changes
  useEffect(() => {
    if (pipelineId) {
      loadPipelineData(pipelineId);
    }
  }, [pipelineId]);
  
  // Render forms
  const renderDataSourceForm = () => (
    <Box
      component={Paper}
      sx={{
        p: 3,
        mb: 3,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <HumanText variant="h6">Register HANA Table</HumanText>
        <IconButton onClick={() => setShowDataSourceForm(false)}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Schema Name"
            value={schemaName}
            onChange={(e) => setSchemaName(e.target.value)}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Table Name"
            value={tableName}
            onChange={(e) => setTableName(e.target.value)}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Sample Size"
            type="number"
            value={sampleSize}
            onChange={(e) => setSampleSize(Number(e.target.value))}
            inputProps={{ min: 1, max: 100 }}
          />
        </Grid>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              variant="contained"
              onClick={registerDataSource}
              disabled={!schemaName || !tableName || isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : null}
            >
              Register Data Source
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
  
  // Simple mock forms for demo purposes
  const renderIntermediateForm = () => (
    <Box
      component={Paper}
      sx={{
        p: 3,
        mb: 3,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <HumanText variant="h6">Register Intermediate Stage</HumanText>
        <IconButton onClick={() => setShowIntermediateForm(false)}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Stage Name"
            value={stageName}
            onChange={(e) => setStageName(e.target.value)}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={2}
            label="Description"
            value={stageDescription}
            onChange={(e) => setStageDescription(e.target.value)}
          />
        </Grid>
        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Source</InputLabel>
            <Select
              value={sourceId}
              label="Source"
              onChange={(e) => setSourceId(e.target.value)}
            >
              {Object.entries(dataSources).map(([id, source]) => (
                <MenuItem key={id} value={id}>
                  {source.schema_name}.{source.table_name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              variant="contained"
              onClick={registerIntermediateStage}
              disabled={!stageName || !sourceId || isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : null}
            >
              Register Stage
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
  
  const renderVectorForm = () => (
    <Box
      component={Paper}
      sx={{
        p: 3,
        mb: 3,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <HumanText variant="h6">Register Vector Representation</HumanText>
        <IconButton onClick={() => setShowVectorForm(false)}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Source</InputLabel>
            <Select
              value={sourceId}
              label="Source"
              onChange={(e) => setSourceId(e.target.value)}
            >
              {Object.entries(dataSources).map(([id, source]) => (
                <MenuItem key={id} value={id}>
                  {source.schema_name}.{source.table_name}
                </MenuItem>
              ))}
              {Object.entries(intermediateStages).map(([id, stage]) => (
                <MenuItem key={id} value={stage.source_id}>
                  {stage.stage_name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Model Name"
            value={vectorModelName}
            onChange={(e) => setVectorModelName(e.target.value)}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Dimensions"
            type="number"
            value={vectorDimensions}
            onChange={(e) => setVectorDimensions(Number(e.target.value))}
            inputProps={{ min: 1 }}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Original Text (optional)"
            value={originalText}
            onChange={(e) => setOriginalText(e.target.value)}
          />
        </Grid>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              variant="contained"
              onClick={registerVector}
              disabled={!sourceId || !vectorModelName || isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : null}
            >
              Register Vector
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
  
  const renderRuleForm = () => (
    <Box
      component={Paper}
      sx={{
        p: 3,
        mb: 3,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <HumanText variant="h6">Register Transformation Rule</HumanText>
        <IconButton onClick={() => setShowRuleForm(false)}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Rule Name"
            value={ruleName}
            onChange={(e) => setRuleName(e.target.value)}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={2}
            label="Description"
            value={ruleDescription}
            onChange={(e) => setRuleDescription(e.target.value)}
          />
        </Grid>
        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Transformation Type</InputLabel>
            <Select
              value={transformationType}
              label="Transformation Type"
              onChange={(e) => setTransformationType(e.target.value)}
            >
              <MenuItem value="join">Join</MenuItem>
              <MenuItem value="filter">Filter</MenuItem>
              <MenuItem value="aggregate">Aggregate</MenuItem>
              <MenuItem value="transform">Transform</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              variant="contained"
              onClick={registerTransformationRule}
              disabled={!ruleName || !transformationType || isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : null}
            >
              Register Rule
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
  
  // Render details panels
  const renderDataSourceDetails = () => {
    if (!selectedDataSource || !dataSources[selectedDataSource]) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <HumanText variant="body2" color="text.secondary">
            Select a data source to view details
          </HumanText>
        </Box>
      );
    }
    
    const source = dataSources[selectedDataSource];
    
    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <HumanText variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
            <DatabaseIcon color="primary" sx={{ mr: 1 }} />
            Data Source Details
          </HumanText>
          <IconButton onClick={() => setSelectedDataSource(null)}>
            <CloseIcon />
          </IconButton>
        </Box>
        <Paper sx={{ p: 2, mb: 3, bgcolor: alpha('#e3f2fd', 0.5) }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Schema</HumanText>
              <HumanText variant="body2">{source.schema_name}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Table</HumanText>
              <HumanText variant="body2">{source.table_name}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Row Count</HumanText>
              <HumanText variant="body2">{source.row_count}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Created At</HumanText>
              <HumanText variant="body2">
                {new Date(source.created_at * 1000).toLocaleString()}
              </HumanText>
            </Grid>
          </Grid>
        </Paper>
        
        <HumanText variant="subtitle1" sx={{ mb: 1 }}>Column Metadata</HumanText>
        <Paper sx={{ p: 2, mb: 3 }}>
          <List dense>
            {Object.entries(source.column_metadata || {}).map(([colName, metadata]) => (
              <ListItem key={colName}>
                <ListItemText
                  primary={colName}
                  secondary={`${(metadata as any).data_type} (${(metadata as any).length}${
                    (metadata as any).scale ? `, scale ${(metadata as any).scale}` : ''
                  })`}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
        
        {source.sample_data && (
          <>
            <HumanText variant="subtitle1" sx={{ mb: 1 }}>Sample Data</HumanText>
            <Paper sx={{ p: 2, mb: 3, overflow: 'auto' }}>
              <Box sx={{ minWidth: 600 }}>
                {source.sample_data.length > 0 ? (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        {Object.keys(source.sample_data[0]).map((key) => (
                          <th key={key} style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {source.sample_data.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {Object.values(row).map((value: any, valueIndex) => (
                            <td key={valueIndex} style={{ border: '1px solid #ddd', padding: '8px' }}>
                              {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <HumanText variant="body2" color="text.secondary">
                    No sample data available
                  </HumanText>
                )}
              </Box>
            </Paper>
          </>
        )}
      </Box>
    );
  };
  
  const renderIntermediateStageDetails = () => {
    if (!selectedStage || !intermediateStages[selectedStage]) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <HumanText variant="body2" color="text.secondary">
            Select an intermediate stage to view details
          </HumanText>
        </Box>
      );
    }
    
    const stage = intermediateStages[selectedStage];
    
    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <HumanText variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
            <TransformIcon color="secondary" sx={{ mr: 1 }} />
            Intermediate Stage Details
          </HumanText>
          <IconButton onClick={() => setSelectedStage(null)}>
            <CloseIcon />
          </IconButton>
        </Box>
        <Paper sx={{ p: 2, mb: 3, bgcolor: alpha('#f3e5f5', 0.5) }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <HumanText variant="subtitle2">Name</HumanText>
              <HumanText variant="body2">{stage.stage_name}</HumanText>
            </Grid>
            <Grid item xs={12}>
              <HumanText variant="subtitle2">Description</HumanText>
              <HumanText variant="body2">{stage.stage_description}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Source ID</HumanText>
              <HumanText variant="body2">{stage.source_id}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Created At</HumanText>
              <HumanText variant="body2">
                {new Date(stage.created_at * 1000).toLocaleString()}
              </HumanText>
            </Grid>
          </Grid>
        </Paper>
        
        <HumanText variant="subtitle1" sx={{ mb: 1 }}>Column Mapping</HumanText>
        <Paper sx={{ p: 2, mb: 3 }}>
          {Object.entries(stage.column_mapping || {}).length > 0 ? (
            <List dense>
              {Object.entries(stage.column_mapping || {}).map(([outputCol, inputCols]) => (
                <ListItem key={outputCol}>
                  <ListItemText
                    primary={outputCol}
                    secondary={`Input columns: ${(inputCols as string[]).join(', ')}`}
                  />
                </ListItem>
              ))}
            </List>
          ) : (
            <HumanText variant="body2" color="text.secondary">
              No column mapping defined
            </HumanText>
          )}
        </Paper>
        
        {stage.data_sample && (
          <>
            <HumanText variant="subtitle1" sx={{ mb: 1 }}>Sample Data</HumanText>
            <Paper sx={{ p: 2, mb: 3, overflow: 'auto' }}>
              <Box sx={{ minWidth: 600 }}>
                {stage.data_sample.length > 0 ? (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        {Object.keys(stage.data_sample[0]).map((key) => (
                          <th key={key} style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {stage.data_sample.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {Object.values(row).map((value: any, valueIndex) => (
                            <td key={valueIndex} style={{ border: '1px solid #ddd', padding: '8px' }}>
                              {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <HumanText variant="body2" color="text.secondary">
                    No sample data available
                  </HumanText>
                )}
              </Box>
            </Paper>
          </>
        )}
      </Box>
    );
  };
  
  const renderVectorDetails = () => {
    if (!selectedVector || !vectors[selectedVector]) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <HumanText variant="body2" color="text.secondary">
            Select a vector to view details
          </HumanText>
        </Box>
      );
    }
    
    const vector = vectors[selectedVector];
    
    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <HumanText variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
            <ModelTrainingIcon color="success" sx={{ mr: 1 }} />
            Vector Details
          </HumanText>
          <Box>
            <Button
              size="small"
              startIcon={<ShowChartIcon />}
              onClick={() => getDataLineage(selectedVector)}
              sx={{ mr: 1 }}
            >
              Lineage
            </Button>
            <IconButton onClick={() => setSelectedVector(null)}>
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>
        <Paper sx={{ p: 2, mb: 3, bgcolor: alpha('#e8f5e9', 0.5) }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Model</HumanText>
              <HumanText variant="body2">{vector.model_name}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Dimensions</HumanText>
              <HumanText variant="body2">{vector.vector_dimensions}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Source ID</HumanText>
              <HumanText variant="body2">{vector.source_id}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Created At</HumanText>
              <HumanText variant="body2">
                {new Date(vector.created_at * 1000).toLocaleString()}
              </HumanText>
            </Grid>
          </Grid>
        </Paper>
        
        {vector.original_text && (
          <>
            <HumanText variant="subtitle1" sx={{ mb: 1 }}>Original Text</HumanText>
            <Paper sx={{ p: 2, mb: 3 }}>
              <HumanText variant="body2">{vector.original_text}</HumanText>
            </Paper>
          </>
        )}
        
        {vector.vector_sample && (
          <>
            <HumanText variant="subtitle1" sx={{ mb: 1 }}>Vector Sample</HumanText>
            <Paper sx={{ p: 2, mb: 3, overflow: 'auto' }}>
              <HumanText variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem', whiteSpace: 'pre-wrap' }}>
                [{vector.vector_sample.slice(0, 10).map((v) => v.toFixed(4)).join(', ')}
                {vector.vector_sample.length > 10 ? ', ...' : ''}]
              </HumanText>
            </Paper>
          </>
        )}
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<SearchIcon />}
              onClick={() => getReverseMap(selectedVector)}
              sx={{ height: '100%' }}
            >
              Find Similar Vectors
            </Button>
          </Grid>
          <Grid item xs={12} md={6}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<ArrowBackIcon />}
              onClick={() => {
                const source = dataSources[vector.source_id];
                if (source) {
                  setSelectedVector(null);
                  setSelectedDataSource(vector.source_id);
                }
              }}
              disabled={!dataSources[vector.source_id]}
              sx={{ height: '100%' }}
            >
              View Source Data
            </Button>
          </Grid>
        </Grid>
      </Box>
    );
  };
  
  const renderTransformationRuleDetails = () => {
    if (!selectedRule || !transformationRules[selectedRule]) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <HumanText variant="body2" color="text.secondary">
            Select a transformation rule to view details
          </HumanText>
        </Box>
      );
    }
    
    const rule = transformationRules[selectedRule];
    
    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <HumanText variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
            <CodeIcon color="warning" sx={{ mr: 1 }} />
            Transformation Rule Details
          </HumanText>
          <IconButton onClick={() => setSelectedRule(null)}>
            <CloseIcon />
          </IconButton>
        </Box>
        <Paper sx={{ p: 2, mb: 3, bgcolor: alpha('#fff8e1', 0.5) }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <HumanText variant="subtitle2">Name</HumanText>
              <HumanText variant="body2">{rule.rule_name}</HumanText>
            </Grid>
            <Grid item xs={12}>
              <HumanText variant="subtitle2">Description</HumanText>
              <HumanText variant="body2">{rule.rule_description}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Transformation Type</HumanText>
              <HumanText variant="body2">
                {rule.transformation_type.charAt(0).toUpperCase() + rule.transformation_type.slice(1)}
              </HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Created At</HumanText>
              <HumanText variant="body2">
                {new Date(rule.created_at * 1000).toLocaleString()}
              </HumanText>
            </Grid>
          </Grid>
        </Paper>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <HumanText variant="subtitle1" sx={{ mb: 1 }}>Input Columns</HumanText>
            <Paper sx={{ p: 2, mb: { xs: 3, md: 0 } }}>
              {rule.input_columns.length > 0 ? (
                <List dense>
                  {rule.input_columns.map((col) => (
                    <ListItem key={col}>
                      <ListItemText primary={col} />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <HumanText variant="body2" color="text.secondary">
                  No input columns defined
                </HumanText>
              )}
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <HumanText variant="subtitle1" sx={{ mb: 1 }}>Output Columns</HumanText>
            <Paper sx={{ p: 2 }}>
              {rule.output_columns.length > 0 ? (
                <List dense>
                  {rule.output_columns.map((col) => (
                    <ListItem key={col}>
                      <ListItemText primary={col} />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <HumanText variant="body2" color="text.secondary">
                  No output columns defined
                </HumanText>
              )}
            </Paper>
          </Grid>
        </Grid>
        
        {Object.keys(rule.transformation_params || {}).length > 0 && (
          <>
            <HumanText variant="subtitle1" sx={{ mb: 1, mt: 3 }}>Parameters</HumanText>
            <Paper sx={{ p: 2 }}>
              <pre style={{ margin: 0, overflow: 'auto' }}>
                {JSON.stringify(rule.transformation_params, null, 2)}
              </pre>
            </Paper>
          </>
        )}
      </Box>
    );
  };
  
  const renderLineageData = () => {
    if (isLineageLoading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
          <CircularProgress />
        </Box>
      );
    }
    
    if (!lineageData) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <HumanText variant="body2" color="text.secondary">
            Select a vector and click "Lineage" to view its data lineage
          </HumanText>
        </Box>
      );
    }
    
    return (
      <Box sx={{ p: 3 }}>
        <HumanText variant="h6" sx={{ mb: 2 }}>Data Lineage</HumanText>
        
        <Paper sx={{ p: 2, mb: 3, bgcolor: alpha('#e8f5e9', 0.5) }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Vector ID</HumanText>
              <HumanText variant="body2">{lineageData.vector_id}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Model</HumanText>
              <HumanText variant="body2">{lineageData.vector_data.model_name}</HumanText>
            </Grid>
          </Grid>
        </Paper>
        
        <HumanText variant="subtitle1" sx={{ mb: 2 }}>Transformation Path</HumanText>
        
        <Stepper orientation="vertical" sx={{ mb: 3 }}>
          <Step active completed>
            <StepLabel StepIconComponent={() => <DatabaseIcon color="primary" />}>
              <HumanText variant="subtitle2">Source Data</HumanText>
            </StepLabel>
            <StepContent>
              <Paper sx={{ p: 2, mb: 2, bgcolor: alpha('#e3f2fd', 0.5) }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <HumanText variant="subtitle2">Schema</HumanText>
                    <HumanText variant="body2">{lineageData.source_data.schema_name}</HumanText>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <HumanText variant="subtitle2">Table</HumanText>
                    <HumanText variant="body2">{lineageData.source_data.table_name}</HumanText>
                  </Grid>
                </Grid>
              </Paper>
            </StepContent>
          </Step>
          
          {lineageData.transformation_stages.map((stage, index) => (
            <Step active completed key={stage.stage_id}>
              <StepLabel StepIconComponent={() => <TransformIcon color="secondary" />}>
                <HumanText variant="subtitle2">{stage.stage_name}</HumanText>
              </StepLabel>
              <StepContent>
                <Paper sx={{ p: 2, mb: 2, bgcolor: alpha('#f3e5f5', 0.5) }}>
                  <HumanText variant="body2">{stage.stage_description}</HumanText>
                  {stage.processing_metadata && (
                    <Box sx={{ mt: 1 }}>
                      <HumanText variant="caption" color="text.secondary">
                        Processing time: {stage.processing_metadata.duration_ms} ms
                      </HumanText>
                    </Box>
                  )}
                </Paper>
              </StepContent>
            </Step>
          ))}
          
          <Step active completed>
            <StepLabel StepIconComponent={() => <ModelTrainingIcon color="success" />}>
              <HumanText variant="subtitle2">Vector Embedding</HumanText>
            </StepLabel>
            <StepContent>
              <Paper sx={{ p: 2, mb: 2, bgcolor: alpha('#e8f5e9', 0.5) }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <HumanText variant="subtitle2">Model</HumanText>
                    <HumanText variant="body2">{lineageData.vector_data.model_name}</HumanText>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <HumanText variant="subtitle2">Dimensions</HumanText>
                    <HumanText variant="body2">{lineageData.vector_data.vector_dimensions}</HumanText>
                  </Grid>
                </Grid>
                {lineageData.vector_data.original_text && (
                  <Box sx={{ mt: 2 }}>
                    <HumanText variant="subtitle2">Original Text</HumanText>
                    <HumanText variant="body2">{lineageData.vector_data.original_text}</HumanText>
                  </Box>
                )}
              </Paper>
            </StepContent>
          </Step>
        </Stepper>
      </Box>
    );
  };
  
  const renderReverseMapping = () => {
    if (isReverseMapLoading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
          <CircularProgress />
        </Box>
      );
    }
    
    if (!reverseMapData) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <HumanText variant="body2" color="text.secondary">
            Select a vector and click "Find Similar Vectors" to view the reverse mapping
          </HumanText>
        </Box>
      );
    }
    
    return (
      <Box sx={{ p: 3 }}>
        <HumanText variant="h6" sx={{ mb: 2 }}>Reverse Mapping</HumanText>
        
        <Paper sx={{ p: 2, mb: 3, bgcolor: alpha('#e8f5e9', 0.5) }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Vector ID</HumanText>
              <HumanText variant="body2">{reverseMapData.vector_id}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Similarity Threshold</HumanText>
              <HumanText variant="body2">{reverseMapData.threshold}</HumanText>
            </Grid>
          </Grid>
        </Paper>
        
        <HumanText variant="subtitle1" sx={{ mb: 2 }}>Source Data</HumanText>
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Schema</HumanText>
              <HumanText variant="body2">{reverseMapData.source_data.schema_name}</HumanText>
            </Grid>
            <Grid item xs={12} sm={6}>
              <HumanText variant="subtitle2">Table</HumanText>
              <HumanText variant="body2">{reverseMapData.source_data.table_name}</HumanText>
            </Grid>
          </Grid>
        </Paper>
        
        <HumanText variant="subtitle1" sx={{ mb: 2 }}>Similar Vectors</HumanText>
        {reverseMapData.similar_vectors.length > 0 ? (
          <List>
            {reverseMapData.similar_vectors.map((vec, index) => (
              <Paper
                key={index}
                sx={{ 
                  mb: 2, 
                  borderLeft: '4px solid',
                  borderColor: theme.palette.primary.main,
                  overflow: 'hidden',
                }}
              >
                <ListItem 
                  secondaryAction={
                    <Chip 
                      label={`${(vec.similarity * 100).toFixed(0)}% similar`}
                      color={
                        vec.similarity > 0.9 
                          ? 'success' 
                          : vec.similarity > 0.7 
                          ? 'primary' 
                          : 'warning'
                      }
                      size="small"
                    />
                  }
                >
                  <ListItemIcon>
                    <ModelTrainingIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary={vec.vector_data.model_name}
                    secondary={`Dimensions: ${vec.vector_data.vector_dimensions}`}
                  />
                </ListItem>
                {vec.vector_data.original_text && (
                  <Box sx={{ px: 2, pb: 2 }}>
                    <HumanText variant="caption" color="text.secondary">
                      Original text:
                    </HumanText>
                    <Paper sx={{ p: 1.5, bgcolor: alpha('#f5f5f5', 0.5) }}>
                      <HumanText variant="body2">{vec.vector_data.original_text}</HumanText>
                    </Paper>
                  </Box>
                )}
              </Paper>
            ))}
          </List>
        ) : (
          <Paper sx={{ p: 2 }}>
            <HumanText variant="body2" color="text.secondary">
              No similar vectors found above the threshold ({reverseMapData.threshold})
            </HumanText>
          </Paper>
        )}
      </Box>
    );
  };
  
  // Main render
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
              Data Pipeline Visualizer
            </HumanText>
            <HumanText variant="body2" color="text.secondary">
              Visualize the complete transformation from HANA tables to vector embeddings
            </HumanText>
          </Box>

          {pipelineId ? (
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                size="small"
                startIcon={<RefreshIcon />}
                onClick={() => loadPipelineData(pipelineId)}
                disabled={isLoading}
              >
                Refresh
              </Button>
              <Button
                variant="contained"
                size="small"
                startIcon={<AddIcon />}
                onClick={() => setShowDataSourceForm(true)}
                disabled={isLoading}
              >
                Add Data Source
              </Button>
            </Box>
          ) : (
            <Button
              variant="contained"
              onClick={createPipeline}
              disabled={isCreatingPipeline}
              startIcon={isCreatingPipeline ? <CircularProgress size={20} /> : null}
            >
              Create Pipeline
            </Button>
          )}
        </Box>

        {/* Main content */}
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Error message */}
          {error && (
            <Alert severity="error" sx={{ m: 2 }}>
              {error}
            </Alert>
          )}

          {/* Forms */}
          <Box sx={{ px: 3, pt: 3 }}>
            {showDataSourceForm && renderDataSourceForm()}
            {showIntermediateForm && renderIntermediateForm()}
            {showVectorForm && renderVectorForm()}
            {showRuleForm && renderRuleForm()}
          </Box>

          {pipelineId ? (
            <Box sx={{ flexGrow: 1, display: 'flex', overflow: 'hidden' }}>
              {/* Flow visualization */}
              <Box sx={{ width: '60%', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ px: 3, pb: 2, display: 'flex', gap: 1 }}>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<TransformIcon />}
                    onClick={() => setShowIntermediateForm(true)}
                  >
                    Add Transformation
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<ModelTrainingIcon />}
                    onClick={() => setShowVectorForm(true)}
                  >
                    Add Vector
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<CodeIcon />}
                    onClick={() => setShowRuleForm(true)}
                  >
                    Add Rule
                  </Button>
                </Box>

                <Box sx={{ flexGrow: 1, position: 'relative' }}>
                  {isLoading ? (
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        bgcolor: 'rgba(255,255,255,0.7)',
                        zIndex: 10,
                      }}
                    >
                      <CircularProgress />
                    </Box>
                  ) : null}

                  <Box sx={{ width: '100%', height: '100%' }}>
                    <ReactFlow
                      nodes={nodes}
                      edges={edges}
                      onNodesChange={onNodesChange}
                      onEdgesChange={onEdgesChange}
                      nodeTypes={nodeTypes}
                      fitView
                      attributionPosition="bottom-right"
                      connectionLineType={ConnectionLineType.SmoothStep}
                    >
                      <Controls />
                      <Background />
                    </ReactFlow>
                  </Box>
                </Box>
              </Box>

              {/* Details panel */}
              <Box
                sx={{
                  width: '40%',
                  borderLeft: '1px solid',
                  borderColor: 'divider',
                  overflow: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <Tabs
                  value={activeTab}
                  onChange={handleTabChange}
                  variant="scrollable"
                  scrollButtons="auto"
                  sx={{ borderBottom: 1, borderColor: 'divider' }}
                >
                  <Tab label="Details" />
                  <Tab label="Lineage" />
                  <Tab label="Reverse Map" />
                </Tabs>

                <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                  {activeTab === 0 && (
                    <>
                      {selectedDataSource ? renderDataSourceDetails() :
                       selectedStage ? renderIntermediateStageDetails() :
                       selectedVector ? renderVectorDetails() :
                       selectedRule ? renderTransformationRuleDetails() : (
                        <Box sx={{ p: 3, textAlign: 'center' }}>
                          <HumanText variant="body2" color="text.secondary">
                            Select a node in the visualization to view details
                          </HumanText>
                        </Box>
                      )}
                    </>
                  )}
                  {activeTab === 1 && renderLineageData()}
                  {activeTab === 2 && renderReverseMapping()}
                </Box>
              </Box>
            </Box>
          ) : (
            <Box
              sx={{
                flexGrow: 1,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                flexDirection: 'column',
                p: 3,
              }}
            >
              <Box
                sx={{
                  width: 80,
                  height: 80,
                  borderRadius: '50%',
                  backgroundColor: alpha('#e3f2fd', 0.7),
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  mb: 2,
                }}
              >
                <LayersIcon
                  sx={{
                    fontSize: 40,
                    color: theme.palette.primary.main,
                  }}
                />
              </Box>
              <HumanText variant="h6" sx={{ mb: 1, textAlign: 'center' }}>
                Create a Data Pipeline
              </HumanText>
              <HumanText
                variant="body2"
                color="text.secondary"
                sx={{ maxWidth: 500, textAlign: 'center' }}
              >
                A data pipeline allows you to visualize and track the complete transformation
                process from HANA tables to vector embeddings and back.
              </HumanText>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default DataPipelineVisualizer;