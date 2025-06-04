import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Tabs,
  Tab,
  Grid,
  Button,
  Divider,
  Paper,
  TextField,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
} from '@mui/material';
import useErrorHandler from '../hooks/useErrorHandler';
import {
  PlayArrow as RunIcon,
  Save as SaveIcon,
  Code as CodeIcon,
  AccountTree as FlowIcon,
  GitHub as GitHubIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
  PlayCircleOutline as PreviewIcon,
  Add as AddIcon,
  Settings as SettingsIcon,
  Storage as StorageIcon,
  Memory as MemoryIcon,
  Search as SearchIcon,
  DataArray as DataArrayIcon,
  CloudUpload as CloudUploadIcon,
  RestoreFromTrash as RestoreIcon,
  BubbleChart as BubbleChartIcon,
  BugReport as DebugIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import ReactFlow, { 
  Background, 
  Controls, 
  useNodesState, 
  useEdgesState,
  addEdge,
  MarkerType,
  ConnectionLineType,
  Panel,
  MiniMap,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { developerService, Flow, FlowNode, FlowEdge } from '../api/services';
import VectorVisualization from '../components/VectorVisualization';
import DebugPanel from '../components/DebugPanel';

// Custom node components
function HanaConnectionNode({ data }) {
  return (
    <NodeCard title="SAP HANA Cloud Connection" icon={<StorageIcon color="primary" />}>
      <Typography variant="body2">Host: {data.params.host}</Typography>
      <Typography variant="body2">Port: {data.params.port}</Typography>
      <Typography variant="body2">User: {data.params.user}</Typography>
      <Typography variant="body2">Schema: {data.params.schema}</Typography>
    </NodeCard>
  );
}

function EmbeddingNode({ data }) {
  return (
    <NodeCard title="Embedding Model" icon={<MemoryIcon color="primary" />}>
      <Typography variant="body2">Model: {data.params.model}</Typography>
      <Typography variant="body2">GPU: {data.params.useGPU ? 'Enabled' : 'Disabled'}</Typography>
      <Typography variant="body2">TensorRT: {data.params.useTensorRT ? 'Enabled' : 'Disabled'}</Typography>
    </NodeCard>
  );
}

function VectorStoreNode({ data }) {
  return (
    <NodeCard title="HANA Vector Store" icon={<StorageIcon color="primary" />}>
      <Typography variant="body2">Table: {data.params.tableName}</Typography>
      <Typography variant="body2">Dimension: {data.params.embeddingDimension}</Typography>
    </NodeCard>
  );
}

function QueryNode({ data }) {
  return (
    <NodeCard title="Query" icon={<SearchIcon color="primary" />}>
      <Typography variant="body2" noWrap sx={{ maxWidth: 150 }}>
        "{data.params.queryText}"
      </Typography>
      <Typography variant="body2">k: {data.params.k}</Typography>
      <Typography variant="body2">MMR: {data.params.useMMR ? 'Enabled' : 'Disabled'}</Typography>
    </NodeCard>
  );
}

function ResultsNode({ data }) {
  return (
    <NodeCard title="Results" icon={<DataArrayIcon color="primary" />}>
      {data.results?.length > 0 ? (
        <Typography variant="body2">{data.results.length} results found</Typography>
      ) : (
        <Typography variant="body2" color="text.secondary">No results yet</Typography>
      )}
    </NodeCard>
  );
}

function NodeCard({ children, title, icon }) {
  return (
    <Paper 
      elevation={2} 
      sx={{ 
        p: 1, 
        minWidth: 200,
        border: '1px solid rgba(0, 102, 179, 0.2)',
        borderRadius: 2,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        {icon}
        <Typography variant="subtitle2" sx={{ ml: 1, fontWeight: 500 }}>
          {title}
        </Typography>
      </Box>
      <Divider sx={{ mb: 1 }} />
      {children}
    </Paper>
  );
}

// Define node types
const nodeTypes = {
  hanaConnection: HanaConnectionNode,
  embedding: EmbeddingNode,
  vectorStore: VectorStoreNode,
  query: QueryNode,
  results: ResultsNode,
};

// Example initial nodes for the visual development environment
const initialNodes = [
  {
    id: '1',
    type: 'hanaConnection',
    data: { 
      label: 'SAP HANA Cloud',
      params: {
        host: 'your-hana-host.hanacloud.ondemand.com',
        port: 443,
        user: 'DBADMIN',
        schema: 'SYSTEM',
      },
    },
    position: { x: 250, y: 5 },
  },
  {
    id: '2',
    type: 'embedding',
    data: { 
      label: 'Embedding Model',
      params: {
        model: 'all-MiniLM-L6-v2',
        useGPU: true,
        useTensorRT: true,
      },
    },
    position: { x: 100, y: 100 },
  },
  {
    id: '3',
    type: 'vectorStore',
    data: { 
      label: 'HANA Vector Store',
      params: {
        tableName: 'LANGCHAIN_VECTORS',
        embeddingDimension: 384,
      },
    },
    position: { x: 400, y: 100 },
  },
  {
    id: '4',
    type: 'query',
    data: { 
      label: 'Query',
      params: {
        queryText: 'What is SAP HANA Cloud?',
        k: 4,
        useMMR: true,
      },
    },
    position: { x: 250, y: 200 },
  },
  {
    id: '5',
    type: 'results',
    data: { 
      label: 'Results',
      results: [],
    },
    position: { x: 250, y: 300 },
  },
];

// Example initial edges for the visual development environment
const initialEdges = [
  { 
    id: 'e1-2', 
    source: '1', 
    target: '2', 
    animated: true, 
    type: 'smoothstep',
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
  },
  { 
    id: 'e1-3', 
    source: '1', 
    target: '3', 
    animated: true, 
    type: 'smoothstep',
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
  },
  { 
    id: 'e2-3', 
    source: '2', 
    target: '3', 
    animated: true, 
    type: 'smoothstep',
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
  },
  { 
    id: 'e3-4', 
    source: '3', 
    target: '4', 
    animated: true, 
    type: 'smoothstep',
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
  },
  { 
    id: 'e4-5', 
    source: '4', 
    target: '5', 
    animated: true, 
    type: 'smoothstep',
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
  },
];

// Node templates for adding new nodes
const nodeTemplates = {
  hanaConnection: {
    type: 'hanaConnection',
    data: { 
      label: 'SAP HANA Cloud',
      params: {
        host: 'your-hana-host.hanacloud.ondemand.com',
        port: 443,
        user: 'DBADMIN',
        schema: 'SYSTEM',
      },
    },
  },
  embedding: {
    type: 'embedding',
    data: { 
      label: 'Embedding Model',
      params: {
        model: 'all-MiniLM-L6-v2',
        useGPU: true,
        useTensorRT: true,
      },
    },
  },
  vectorStore: {
    type: 'vectorStore',
    data: { 
      label: 'HANA Vector Store',
      params: {
        tableName: 'LANGCHAIN_VECTORS',
        embeddingDimension: 384,
      },
    },
  },
  query: {
    type: 'query',
    data: { 
      label: 'Query',
      params: {
        queryText: 'Enter your query here',
        k: 4,
        useMMR: false,
      },
    },
  },
  results: {
    type: 'results',
    data: { 
      label: 'Results',
      results: [],
    },
  },
};

const Developer: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [running, setRunning] = useState(false);
  const [generatedCode, setGeneratedCode] = useState('');
  const [copied, setCopied] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [addNodeDialogOpen, setAddNodeDialogOpen] = useState(false);
  const [newNodeType, setNewNodeType] = useState('embedding');
  const [nodeEditDialogOpen, setNodeEditDialogOpen] = useState(false);
  const [currentFlowName, setCurrentFlowName] = useState('HANA Vector Search');
  const [currentFlowDescription, setCurrentFlowDescription] = useState('A LangChain integration with SAP HANA Cloud for semantic search with GPU acceleration');
  const [savedFlows, setSavedFlows] = useState<Flow[]>([]);
  const [loadFlowDialogOpen, setLoadFlowDialogOpen] = useState(false);
  const { handleError, clearError } = useErrorHandler();
  
  // Generate Python code whenever the flow changes
  useEffect(() => {
    generateCode();
  }, [nodes, edges]);
  
  // Fetch saved flows when the component mounts
  useEffect(() => {
    fetchSavedFlows();
  }, []);
  
  // Function to fetch saved flows from the API
  const fetchSavedFlows = async () => {
    try {
      const response = await developerService.listFlows();
      setSavedFlows(response.data.flows);
    } catch (error) {
      console.error("Error fetching saved flows:", error);
      handleError(error);
    }
  };
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge({
      ...params,
      type: 'smoothstep',
      animated: true,
      markerEnd: {
        type: MarkerType.ArrowClosed,
      },
    }, eds));
  }, [setEdges]);
  
  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
    setNodeEditDialogOpen(true);
  }, []);
  
  const handleAddNode = () => {
    const newNode = {
      ...nodeTemplates[newNodeType],
      id: `${nodes.length + 1}`,
      position: { 
        x: Math.random() * 300 + 100, 
        y: Math.random() * 300 + 100 
      },
    };
    
    setNodes((nds) => [...nds, newNode]);
    setAddNodeDialogOpen(false);
  };
  
  const handleUpdateNodeParams = () => {
    if (!selectedNode) return;
    
    setNodes((nds) => 
      nds.map((node) => {
        if (node.id === selectedNode.id) {
          return {
            ...node,
            data: {
              ...node.data,
              params: { ...selectedNode.data.params },
            },
          };
        }
        return node;
      })
    );
    
    setNodeEditDialogOpen(false);
  };
  
  const handleParamChange = (key, value) => {
    if (!selectedNode) return;
    
    setSelectedNode({
      ...selectedNode,
      data: {
        ...selectedNode.data,
        params: {
          ...selectedNode.data.params,
          [key]: value,
        },
      },
    });
  };
  
  const generateCode = async () => {
    try {
      // Create a flow object
      const flow: Flow = {
        name: currentFlowName,
        description: currentFlowDescription,
        nodes: nodes,
        edges: edges,
      };
      
      // Call the API to generate code
      const response = await developerService.generateCode(flow);
      setGeneratedCode(response.data.code);
    } catch (error) {
      console.error("Error generating code:", error);
      handleError(error);
      
      // Fallback to local code generation if API fails
      const code = `"""
${currentFlowName}
${currentFlowDescription}

Auto-generated by SAP HANA LangChain Visual Developer
"""

from langchain_hana import HanaVectorStore
from langchain_hana.embeddings import GPUAcceleratedEmbeddings

# This is a fallback code generation. The API request failed.
# Please try again or contact support.
`;
      setGeneratedCode(code);
    }
  };
  
  const handleRun = async () => {
    setRunning(true);
    clearError(); // Clear any previous errors
    
    try {
      // Create a flow object
      const flow: Flow = {
        name: currentFlowName,
        description: currentFlowDescription,
        nodes: nodes,
        edges: edges,
      };
      
      // Call the API to run the flow
      const response = await developerService.runFlow(flow);
      
      if (response.data.success) {
        // Update the results node with the retrieved results
        const resultsNodeId = nodes.find(node => node.type === 'results')?.id;
        if (resultsNodeId) {
          setNodes((nds) => 
            nds.map((node) => {
              if (node.id === resultsNodeId) {
                return {
                  ...node,
                  data: {
                    ...node.data,
                    results: response.data.results,
                  },
                };
              }
              return node;
            })
          );
        }
        
        // Update the generated code
        setGeneratedCode(response.data.generated_code);
      } else {
        throw new Error("Flow execution failed");
      }
    } catch (error) {
      console.error("Error running the flow:", error);
      handleError(error);
      
      // Show error in UI
      const resultsNodeId = nodes.find(node => node.type === 'results')?.id;
      if (resultsNodeId) {
        setNodes((nds) => 
          nds.map((node) => {
            if (node.id === resultsNodeId) {
              return {
                ...node,
                data: {
                  ...node.data,
                  results: [],
                },
              };
            }
            return node;
          })
        );
      }
    } finally {
      setRunning(false);
    }
  };
  
  const handleCopyCode = () => {
    navigator.clipboard.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const handleDownloadCode = () => {
    const element = document.createElement("a");
    const file = new Blob([generatedCode], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = "langchain_hana_integration.py";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };
  
  const handleSaveFlow = async () => {
    clearError(); // Clear any previous errors
    
    try {
      // Create a flow object
      const flow: Flow = {
        name: currentFlowName,
        description: currentFlowDescription,
        nodes: nodes,
        edges: edges,
      };
      
      // Call the API to save the flow
      const response = await developerService.saveFlow(flow);
      
      if (response.data.success) {
        // Update the flow ID if it's a new flow
        if (!flow.id) {
          flow.id = response.data.flow_id;
        }
        
        // Show success message
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        console.error("Failed to save flow:", response.data.message);
        throw new Error(response.data.message || "Failed to save flow");
      }
    } catch (error) {
      console.error("Error saving flow:", error);
      handleError(error);
    }
  };
  
  // Function to load a flow from the API
  const handleLoadFlow = async (flowId: string) => {
    clearError(); // Clear any previous errors
    
    try {
      const response = await developerService.getFlow(flowId);
      const flow = response.data;
      
      // Update the current flow name and description
      setCurrentFlowName(flow.name);
      setCurrentFlowDescription(flow.description);
      
      // Update the nodes and edges
      setNodes(flow.nodes);
      setEdges(flow.edges);
      
      // Close the dialog
      setLoadFlowDialogOpen(false);
    } catch (error) {
      console.error("Error loading flow:", error);
      handleError(error);
    }
  };
  
  // Function to delete a flow
  const handleDeleteFlow = async (flowId: string) => {
    clearError(); // Clear any previous errors
    
    try {
      await developerService.deleteFlow(flowId);
      
      // Refresh the list of saved flows
      fetchSavedFlows();
    } catch (error) {
      console.error("Error deleting flow:", error);
      handleError(error);
    }
  };
  
  const renderNodeEditDialog = () => {
    if (!selectedNode) return null;
    
    const nodeType = selectedNode.type;
    const params = selectedNode.data.params;
    
    return (
      <Dialog 
        open={nodeEditDialogOpen} 
        onClose={() => setNodeEditDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Edit {selectedNode.data.label} Node
        </DialogTitle>
        <DialogContent dividers>
          {nodeType === 'hanaConnection' && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  label="Host"
                  fullWidth
                  value={params.host}
                  onChange={(e) => handleParamChange('host', e.target.value)}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Port"
                  type="number"
                  fullWidth
                  value={params.port}
                  onChange={(e) => handleParamChange('port', Number(e.target.value))}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="User"
                  fullWidth
                  value={params.user}
                  onChange={(e) => handleParamChange('user', e.target.value)}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Schema"
                  fullWidth
                  value={params.schema}
                  onChange={(e) => handleParamChange('schema', e.target.value)}
                  margin="normal"
                />
              </Grid>
            </Grid>
          )}
          
          {nodeType === 'embedding' && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Embedding Model</InputLabel>
                  <Select
                    value={params.model}
                    label="Embedding Model"
                    onChange={(e) => handleParamChange('model', e.target.value)}
                  >
                    <MenuItem value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</MenuItem>
                    <MenuItem value="all-mpnet-base-v2">all-mpnet-base-v2</MenuItem>
                    <MenuItem value="paraphrase-multilingual-MiniLM-L12-v2">paraphrase-multilingual-MiniLM-L12-v2</MenuItem>
                    <MenuItem value="bge-small-en">bge-small-en</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={params.useGPU}
                      onChange={(e) => handleParamChange('useGPU', e.target.checked)}
                    />
                  }
                  label="Use GPU Acceleration"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={params.useTensorRT}
                      onChange={(e) => handleParamChange('useTensorRT', e.target.checked)}
                      disabled={!params.useGPU}
                    />
                  }
                  label="Use TensorRT Optimization"
                />
              </Grid>
            </Grid>
          )}
          
          {nodeType === 'vectorStore' && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  label="Table Name"
                  fullWidth
                  value={params.tableName}
                  onChange={(e) => handleParamChange('tableName', e.target.value)}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Embedding Dimension"
                  type="number"
                  fullWidth
                  value={params.embeddingDimension}
                  onChange={(e) => handleParamChange('embeddingDimension', Number(e.target.value))}
                  margin="normal"
                />
              </Grid>
            </Grid>
          )}
          
          {nodeType === 'query' && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  label="Query Text"
                  fullWidth
                  value={params.queryText}
                  onChange={(e) => handleParamChange('queryText', e.target.value)}
                  margin="normal"
                  multiline
                  rows={3}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Number of Results (k)"
                  type="number"
                  fullWidth
                  value={params.k}
                  onChange={(e) => handleParamChange('k', Number(e.target.value))}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={params.useMMR}
                      onChange={(e) => handleParamChange('useMMR', e.target.checked)}
                    />
                  }
                  label="Use Maximum Marginal Relevance (MMR)"
                />
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNodeEditDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleUpdateNodeParams} 
            variant="contained" 
            color="primary"
          >
            Update
          </Button>
        </DialogActions>
      </Dialog>
    );
  };
  
  const renderLoadFlowDialog = () => {
    return (
      <Dialog 
        open={loadFlowDialogOpen} 
        onClose={() => setLoadFlowDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Load Saved Flow
        </DialogTitle>
        <DialogContent dividers>
          {savedFlows.length === 0 ? (
            <Typography variant="body1" align="center" sx={{ py: 4 }}>
              No saved flows found. Create and save a flow first.
            </Typography>
          ) : (
            <Grid container spacing={2}>
              {savedFlows.map((flow) => (
                <Grid item xs={12} sm={6} md={4} key={flow.id}>
                  <Paper 
                    variant="outlined" 
                    sx={{ 
                      p: 2, 
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      cursor: 'pointer',
                      '&:hover': {
                        boxShadow: 3,
                      },
                    }}
                    onClick={() => handleLoadFlow(flow.id)}
                  >
                    <Typography variant="h6" gutterBottom noWrap>
                      {flow.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {flow.description.length > 100 
                        ? flow.description.substring(0, 100) + '...' 
                        : flow.description}
                    </Typography>
                    <Box sx={{ mt: 'auto', display: 'flex', justifyContent: 'space-between', pt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        {flow.updated_at 
                          ? new Date(flow.updated_at).toLocaleDateString() 
                          : 'No date'}
                      </Typography>
                      <IconButton 
                        size="small" 
                        color="error"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteFlow(flow.id);
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLoadFlowDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={() => fetchSavedFlows()} 
            variant="outlined" 
            startIcon={<RefreshIcon />}
          >
            Refresh
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  return (
    <Box className="fade-in">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="500">
          Visual Developer
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Visually design and test your LangChain integration with SAP HANA Cloud
        </Typography>
      </Box>
      
      <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tab icon={<FlowIcon />} label="Visual Designer" iconPosition="start" />
        <Tab icon={<CodeIcon />} label="Generated Code" iconPosition="start" />
        <Tab icon={<PreviewIcon />} label="Preview" iconPosition="start" />
        <Tab icon={<BubbleChartIcon />} label="Vector Explorer" iconPosition="start" />
        <Tab icon={<DebugIcon />} label="Debugger" iconPosition="start" />
      </Tabs>
      
      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={9}>
            <Paper 
              variant="outlined" 
              sx={{ 
                height: 600, 
                p: 0,
                borderRadius: 2,
                overflow: 'hidden',
              }}
            >
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={onNodeClick}
                nodeTypes={nodeTypes}
                fitView
                snapToGrid
                connectionLineType={ConnectionLineType.SmoothStep}
              >
                <Controls />
                <MiniMap 
                  nodeStrokeColor="#0066B3"
                  nodeColor="#0066B3"
                  nodeBorderRadius={8}
                />
                <Background gap={12} size={1} />
                <Panel position="top-right">
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<AddIcon />}
                    onClick={() => setAddNodeDialogOpen(true)}
                    sx={{ mr: 1 }}
                  >
                    Add Node
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={running ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
                    onClick={handleRun}
                    disabled={running}
                  >
                    {running ? 'Running...' : 'Run Flow'}
                  </Button>
                </Panel>
              </ReactFlow>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Flow Controls
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Button 
                  variant="contained" 
                  color="primary" 
                  fullWidth 
                  startIcon={running ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
                  onClick={handleRun}
                  disabled={running}
                  sx={{ mb: 2 }}
                >
                  {running ? 'Running...' : 'Run Flow'}
                </Button>
                
                <Button 
                  variant="outlined" 
                  color="primary" 
                  fullWidth 
                  startIcon={<SaveIcon />}
                  onClick={handleSaveFlow}
                  sx={{ mb: 2 }}
                >
                  Save Flow
                </Button>
                
                <Button 
                  variant="outlined" 
                  color="primary" 
                  fullWidth 
                  startIcon={<CloudUploadIcon />}
                  sx={{ mb: 2 }}
                >
                  Deploy Flow
                </Button>
                
                <Button 
                  variant="outlined" 
                  color="primary" 
                  fullWidth 
                  startIcon={<GitHubIcon />}
                  sx={{ mb: 2 }}
                >
                  Export to GitHub
                </Button>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="subtitle2" gutterBottom>
                  Flow Properties
                </Typography>
                
                <TextField
                  label="Flow Name"
                  fullWidth
                  value={currentFlowName}
                  onChange={(e) => setCurrentFlowName(e.target.value)}
                  variant="outlined"
                  size="small"
                  sx={{ mb: 2 }}
                />
                
                <TextField
                  label="Description"
                  fullWidth
                  multiline
                  rows={3}
                  value={currentFlowDescription}
                  onChange={(e) => setCurrentFlowDescription(e.target.value)}
                  variant="outlined"
                  size="small"
                  sx={{ mb: 2 }}
                />
                
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<RestoreIcon />}
                    onClick={() => setLoadFlowDialogOpen(true)}
                  >
                    Load Saved Flow
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
      
      {tabValue === 1 && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Generated Python Code
              </Typography>
              <Box>
                <Tooltip title="Copy code">
                  <IconButton onClick={handleCopyCode}>
                    <CopyIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Download code">
                  <IconButton onClick={handleDownloadCode}>
                    <DownloadIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
            <Divider sx={{ mb: 2 }} />
            
            <SyntaxHighlighter
              language="python"
              style={atomDark}
              customStyle={{
                borderRadius: '8px',
                padding: '16px',
                maxHeight: '600px',
              }}
            >
              {generatedCode}
            </SyntaxHighlighter>
          </CardContent>
        </Card>
      )}
      
      {tabValue === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Flow Preview
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <Alert severity="info" sx={{ mb: 3 }}>
              This preview shows what will happen when you run this flow with the current configuration.
            </Alert>
            
            <Paper variant="outlined" sx={{ p: 3, mb: 3, borderRadius: 2 }}>
              <Typography variant="subtitle1" gutterBottom fontWeight={500}>
                Flow Summary
              </Typography>
              <Typography variant="body2" paragraph>
                This flow connects to SAP HANA Cloud, initializes a GPU-accelerated embedding model with TensorRT optimization,
                and performs a semantic search query against the HANA vector store.
              </Typography>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Components:
                </Typography>
                <ul>
                  <li>SAP HANA Cloud Connection</li>
                  <li>GPU-Accelerated Embedding Model</li>
                  <li>HANA Vector Store</li>
                  <li>Semantic Query</li>
                  <li>Results Processing</li>
                </ul>
              </Box>
            </Paper>
            
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom fontWeight={500}>
                Expected Results
              </Typography>
              
              {running ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <Box>
                  <Typography variant="body2" paragraph>
                    For the query "{nodes.find(n => n.id === '4')?.data.params.queryText}", the flow will return:
                  </Typography>
                  
                  {nodes.find(n => n.id === '5')?.data.results?.length > 0 ? (
                    nodes.find(n => n.id === '5').data.results.map((result, index) => (
                      <Paper 
                        key={index} 
                        variant="outlined" 
                        sx={{ p: 2, mb: 2, borderRadius: 2 }}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="subtitle2">
                            Result {index + 1}
                          </Typography>
                          <Typography variant="body2" color="primary.main">
                            Score: {result.score.toFixed(2)}
                          </Typography>
                        </Box>
                        <Typography variant="body2">
                          {result.content}
                        </Typography>
                      </Paper>
                    ))
                  ) : (
                    <Alert severity="warning">
                      No results available. Run the flow to see results.
                    </Alert>
                  )}
                </Box>
              )}
            </Box>
            
            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
              <Button 
                variant="contained" 
                color="primary" 
                startIcon={running ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
                onClick={handleRun}
                disabled={running}
              >
                {running ? 'Running...' : 'Run Flow'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}
      
      {tabValue === 3 && (
        <Box sx={{ height: 'calc(100vh - 250px)', minHeight: 600 }}>
          <VectorVisualization 
            tableName={nodes.find(n => n.type === 'vectorStore')?.data.params.tableName || 'EMBEDDINGS'}
          />
        </Box>
      )}
      
      {tabValue === 4 && (
        <Box sx={{ height: 'calc(100vh - 250px)', minHeight: 600 }}>
          <DebugPanel 
            flow={{
              id: flow.id,
              name: currentFlowName,
              nodes: nodes,
              edges: edges
            }}
            onNodeHighlight={(nodeId) => {
              // Highlight the node in the flow
              const node = nodes.find(n => n.id === nodeId);
              if (node) {
                // You could implement a visual highlight here
                console.log('Highlighting node:', nodeId);
              }
            }}
          />
        </Box>
      )}
      
      {/* Add Node Dialog */}
      <Dialog 
        open={addNodeDialogOpen} 
        onClose={() => setAddNodeDialogOpen(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>
          Add New Node
        </DialogTitle>
        <DialogContent dividers>
          <FormControl fullWidth margin="normal">
            <InputLabel>Node Type</InputLabel>
            <Select
              value={newNodeType}
              label="Node Type"
              onChange={(e) => setNewNodeType(e.target.value)}
            >
              <MenuItem value="hanaConnection">SAP HANA Cloud Connection</MenuItem>
              <MenuItem value="embedding">Embedding Model</MenuItem>
              <MenuItem value="vectorStore">HANA Vector Store</MenuItem>
              <MenuItem value="query">Query</MenuItem>
              <MenuItem value="results">Results</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddNodeDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddNode} variant="contained" color="primary">
            Add
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Node Edit Dialog */}
      {renderNodeEditDialog()}
      
      <Snackbar
        open={copied}
        autoHideDuration={2000}
        message={tabValue === 1 ? "Code copied to clipboard" : "Flow saved successfully"}
      />
    </Box>
  );
};

export default Developer;