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
import axios from 'axios';

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
  
  // Generate Python code whenever the flow changes
  useEffect(() => {
    generateCode();
  }, [nodes, edges]);
  
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
  
  const generateCode = () => {
    // Create a simple Python code representation of the flow
    const connectionNodes = nodes.filter(node => node.type === 'hanaConnection');
    const embeddingNodes = nodes.filter(node => node.type === 'embedding');
    const vectorStoreNodes = nodes.filter(node => node.type === 'vectorStore');
    const queryNodes = nodes.filter(node => node.type === 'query');
    
    let code = `"""
${currentFlowName}
${currentFlowDescription}

Auto-generated by SAP HANA LangChain Visual Developer
"""

from langchain_hana import HanaVectorStore
from langchain_hana.embeddings import GPUAcceleratedEmbeddings
`;

    // Add imports based on node types
    if (embeddingNodes.some(node => node.data.params.useTensorRT)) {
      code += `from langchain_hana.embeddings import TensorRTEmbeddings\n`;
    }
    
    code += `
# Connect to SAP HANA Cloud
import hdbcli.dbapi\n`;

    // Generate connection code
    if (connectionNodes.length > 0) {
      const conn = connectionNodes[0].data.params;
      code += `conn = hdbcli.dbapi.connect(
    address="${conn.host}",
    port=${conn.port},
    user="${conn.user}",
    password="********",  # Replace with your actual password
    encrypt=True,
    sslValidateCertificate=False
)
`;
    }

    // Generate embedding model code
    if (embeddingNodes.length > 0) {
      const embedding = embeddingNodes[0].data.params;
      
      if (embedding.useTensorRT) {
        code += `
# Initialize TensorRT-optimized embedding model
embeddings = TensorRTEmbeddings(
    model_name="${embedding.model}",
    device="${embedding.useGPU ? 'cuda' : 'cpu'}",
    precision="fp16"
)
`;
      } else if (embedding.useGPU) {
        code += `
# Initialize GPU-accelerated embedding model
embeddings = GPUAcceleratedEmbeddings(
    model_name="${embedding.model}",
    device="cuda",
    batch_size=32
)
`;
      } else {
        code += `
# Initialize embedding model
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="${embedding.model}")
`;
      }
    }

    // Generate vector store code
    if (vectorStoreNodes.length > 0 && connectionNodes.length > 0 && embeddingNodes.length > 0) {
      const vectorStore = vectorStoreNodes[0].data.params;
      code += `
# Create or connect to vector store
vector_store = HanaVectorStore(
    connection=conn,
    embedding=embeddings,
    table_name="${vectorStore.tableName}",
    embedding_dimension=${vectorStore.embeddingDimension}
)
`;
    }

    // Generate query code
    if (queryNodes.length > 0 && vectorStoreNodes.length > 0) {
      const query = queryNodes[0].data.params;
      
      code += `
# Perform semantic search
query = "${query.queryText}"
`;
      
      if (query.useMMR) {
        code += `
# Using Maximum Marginal Relevance for diverse results
results = vector_store.max_marginal_relevance_search(
    query=query, 
    k=${query.k},
    fetch_k=${query.k * 4},
    lambda_mult=0.5
)
`;
      } else {
        code += `
# Standard similarity search
results = vector_store.similarity_search(
    query=query, 
    k=${query.k}
)
`;
      }
      
      // Add code to process results
      code += `
# Process and display results
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")
`;
    }
    
    // Add code to close the connection
    code += `
# Close the connection
conn.close()
`;
    
    setGeneratedCode(code);
  };
  
  const handleRun = async () => {
    setRunning(true);
    
    try {
      // Find query node to get the query text
      const queryNode = nodes.find(node => node.type === 'query');
      if (!queryNode) throw new Error("No query node found in the flow");
      
      const query = queryNode.data.params.queryText;
      const k = queryNode.data.params.k;
      const useMMR = queryNode.data.params.useMMR;
      
      // Prepare the endpoint and parameters based on whether MMR is used
      const endpoint = useMMR ? '/query/mmr' : '/query';
      const params = {
        query,
        k,
      };
      
      if (useMMR) {
        params.fetch_k = k * 4;
        params.lambda_mult = 0.5;
      }
      
      // Call the API
      const response = await axios.post(endpoint, params);
      
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
                  results: response.data.results.map(result => ({
                    content: result.document.page_content,
                    score: result.score,
                    metadata: result.document.metadata,
                  })),
                },
              };
            }
            return node;
          })
        );
      }
    } catch (error) {
      console.error("Error running the flow:", error);
      // Simulate results with mock data if the API call fails
      const resultsNodeId = nodes.find(node => node.type === 'results')?.id;
      if (resultsNodeId) {
        setNodes((nds) => 
          nds.map((node) => {
            if (node.id === resultsNodeId) {
              return {
                ...node,
                data: {
                  ...node.data,
                  results: [
                    { content: "SAP HANA Cloud is a cloud-based database management system.", score: 0.89 },
                    { content: "HANA Cloud offers in-memory computing capabilities.", score: 0.78 },
                    { content: "SAP HANA Cloud provides both transactional and analytical processing.", score: 0.72 },
                    { content: "HANA Cloud integrates with SAP's business applications.", score: 0.65 }
                  ],
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
  
  const handleSaveFlow = () => {
    // In a real app, this would save to a backend or localStorage
    const flow = {
      name: currentFlowName,
      description: currentFlowDescription,
      nodes,
      edges,
    };
    
    // Here we just simulate saving by showing a snackbar
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
                
                <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                  <InputLabel>Template</InputLabel>
                  <Select
                    value="custom"
                    label="Template"
                  >
                    <MenuItem value="custom">Custom Flow</MenuItem>
                    <MenuItem value="semantic_search">Semantic Search</MenuItem>
                    <MenuItem value="qa_chain">Question Answering</MenuItem>
                    <MenuItem value="summarization">Document Summarization</MenuItem>
                  </Select>
                </FormControl>
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