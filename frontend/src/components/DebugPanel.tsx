import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Divider,
  Button,
  IconButton,
  Tooltip,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Collapse,
  Chip,
  CircularProgress,
  Alert,
  Stack,
  Card,
  CardContent,
  Tabs,
  Tab,
  styled,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  SkipNext as StepIcon,
  FastForward as ContinueIcon,
  Refresh as ResetIcon,
  Check as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  PauseCircle as BreakpointIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  BugReport as DebugIcon,
  Code as CodeIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { 
  DebugSession, 
  DebugNodeData,
  DebugBreakpoint,
  DebugStepType,
  FlowNode, 
  developerService 
} from '../api/services';

interface DebugPanelProps {
  flow: {
    id?: string;
    name: string;
    nodes: FlowNode[];
    edges: any[];
  };
  onNodeHighlight?: (nodeId: string) => void;
}

// Import the withAnimation HOC from the shared package
import { withAnimation } from '@finsightdev/ui-animations';

// Styled components
const BaseDebugButton = styled(Button)(({ theme }) => ({
  minWidth: 40,
  margin: theme.spacing(0, 0.5),
}));

// Enhanced button with animations
const DebugButton = withAnimation(BaseDebugButton, {
  animationType: 'scale',
  enableHover: true,
  enableSound: true,
  soundType: 'tap',
  tension: 350,
  friction: 18
});

const NodeStatusChip = ({ status }: { status: string }) => {
  let color: 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' = 'default';
  let icon = null;

  switch (status) {
    case 'not_executed':
      color = 'default';
      break;
    case 'executing':
      color = 'info';
      icon = <InfoIcon />;
      break;
    case 'completed':
      color = 'success';
      icon = <CheckIcon />;
      break;
    case 'error':
      color = 'error';
      icon = <ErrorIcon />;
      break;
    default:
      color = 'default';
  }

  return (
    <Chip 
      size="small" 
      color={color} 
      label={status.replace('_', ' ')} 
      icon={icon}
    />
  );
};

const DebugPanel: React.FC<DebugPanelProps> = ({ flow, onNodeHighlight }) => {
  // State
  const [debugSession, setDebugSession] = useState<DebugSession | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [tabValue, setTabValue] = useState<number>(0);
  const [selectedVariable, setSelectedVariable] = useState<string | null>(null);

  // Initialize a debug session
  const initDebugSession = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await developerService.createDebugSession(flow);
      
      if (response.data.session_id) {
        const sessionData = await developerService.getDebugSession(response.data.session_id);
        setDebugSession(sessionData.data);
      }
    } catch (err) {
      console.error('Error initializing debug session:', err);
      setError('Failed to initialize debug session. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Step through the debug session
  const stepDebugSession = async (stepType: DebugStepType) => {
    if (!debugSession) return;
    
    setLoading(true);
    
    try {
      const response = await developerService.stepDebugSession(
        debugSession.session_id, 
        stepType
      );
      
      setDebugSession(response.data.session);
      
      // Highlight the current node if callback provided
      if (response.data.session.current_node_id && onNodeHighlight) {
        onNodeHighlight(response.data.session.current_node_id);
      }
    } catch (err) {
      console.error(`Error stepping debug session (${stepType}):`, err);
      setError(`Failed to execute ${stepType} operation. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  // Set a breakpoint
  const toggleBreakpoint = async (nodeId: string) => {
    if (!debugSession) return;
    
    // Find if a breakpoint already exists for this node
    const existingBreakpoint = debugSession.breakpoints.find(bp => bp.node_id === nodeId);
    
    const newBreakpoint: DebugBreakpoint = {
      node_id: nodeId,
      enabled: existingBreakpoint ? !existingBreakpoint.enabled : true
    };
    
    try {
      await developerService.setBreakpoint(debugSession.session_id, newBreakpoint);
      
      // Refresh the session
      const sessionData = await developerService.getDebugSession(debugSession.session_id);
      setDebugSession(sessionData.data);
    } catch (err) {
      console.error('Error toggling breakpoint:', err);
      setError('Failed to toggle breakpoint. Please try again.');
    }
  };

  // Toggle expanded state for a node
  const toggleNodeExpanded = (nodeId: string) => {
    setExpandedNodes(prev => ({
      ...prev,
      [nodeId]: !prev[nodeId]
    }));
  };

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Get node label by ID
  const getNodeLabel = (nodeId: string): string => {
    const node = flow.nodes.find(n => n.id === nodeId);
    return node?.data.label || `Node ${nodeId}`;
  };

  // Get node type by ID
  const getNodeType = (nodeId: string): string => {
    const node = flow.nodes.find(n => n.id === nodeId);
    return node?.type || 'unknown';
  };

  // Format execution time
  const formatExecutionTime = (time?: number): string => {
    if (!time) return '0 ms';
    return `${time.toFixed(2)} ms`;
  };

  // Reset debug session on flow change
  useEffect(() => {
    if (debugSession) {
      // Clean up old session
      developerService.deleteDebugSession(debugSession.session_id)
        .catch(err => console.error('Error deleting debug session:', err));
      
      setDebugSession(null);
    }
  }, [flow]);

  // Render nodes list
  const renderNodesList = () => {
    if (!debugSession) return null;
    
    return (
      <List dense component={Paper} variant="outlined" sx={{ mb: 2 }}>
        {flow.nodes.map(node => {
          const nodeData = debugSession.node_data[node.id];
          const hasBreakpoint = debugSession.breakpoints.some(bp => bp.node_id === node.id && bp.enabled);
          const isExpanded = expandedNodes[node.id] || false;
          
          return (
            <React.Fragment key={node.id}>
              <ListItem
                button
                onClick={() => toggleNodeExpanded(node.id)}
                selected={debugSession.current_node_id === node.id}
                secondaryAction={
                  <IconButton 
                    edge="end" 
                    size="small" 
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleBreakpoint(node.id);
                    }}
                    color={hasBreakpoint ? "error" : "default"}
                  >
                    <BreakpointIcon />
                  </IconButton>
                }
              >
                <ListItemIcon>
                  {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </ListItemIcon>
                <ListItemText 
                  primary={getNodeLabel(node.id)}
                  secondary={`Type: ${getNodeType(node.id)}`}
                />
                {nodeData && <NodeStatusChip status={nodeData.status} />}
              </ListItem>
              
              <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                <Box sx={{ pl: 4, pr: 2, py: 1, bgcolor: 'background.paper' }}>
                  {nodeData && (
                    <>
                      <Typography variant="body2" color="textSecondary">
                        Status: <NodeStatusChip status={nodeData.status} />
                      </Typography>
                      
                      {nodeData.execution_time && (
                        <Typography variant="body2" color="textSecondary">
                          Execution time: {formatExecutionTime(nodeData.execution_time)}
                        </Typography>
                      )}
                      
                      {nodeData.error && (
                        <Alert severity="error" sx={{ mt: 1, mb: 1 }}>
                          {nodeData.error}
                        </Alert>
                      )}
                      
                      {nodeData.output_data && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="body2" color="textSecondary">
                            Output:
                          </Typography>
                          <Paper variant="outlined" sx={{ p: 1, mt: 0.5, maxHeight: 200, overflow: 'auto' }}>
                            <pre style={{ margin: 0, fontSize: '0.75rem' }}>
                              {JSON.stringify(nodeData.output_data, null, 2)}
                            </pre>
                          </Paper>
                        </Box>
                      )}
                    </>
                  )}
                </Box>
              </Collapse>
              <Divider />
            </React.Fragment>
          );
        })}
      </List>
    );
  };

  // Render variables tab
  const renderVariablesTab = () => {
    if (!debugSession) return null;
    
    const variables = Object.keys(debugSession.variables);
    
    if (variables.length === 0) {
      return (
        <Alert severity="info" sx={{ mt: 2 }}>
          No variables available yet. Run the flow to see variables.
        </Alert>
      );
    }
    
    return (
      <Box sx={{ mt: 2 }}>
        <Paper variant="outlined" sx={{ mb: 2 }}>
          <List dense>
            {variables.map(varName => (
              <ListItem 
                key={varName} 
                button 
                selected={selectedVariable === varName}
                onClick={() => setSelectedVariable(varName)}
              >
                <ListItemIcon>
                  <MemoryIcon />
                </ListItemIcon>
                <ListItemText primary={varName} />
              </ListItem>
            ))}
          </List>
        </Paper>
        
        {selectedVariable && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              {selectedVariable}
            </Typography>
            <Divider sx={{ mb: 1 }} />
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              <pre style={{ margin: 0, fontSize: '0.75rem' }}>
                {JSON.stringify(debugSession.variables[selectedVariable], null, 2)}
              </pre>
            </Box>
          </Paper>
        )}
      </Box>
    );
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ p: 2, flexGrow: 0 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
            <DebugIcon sx={{ mr: 1 }} />
            Flow Debugger
          </Typography>
          
          {!debugSession ? (
            <Button
              variant="contained"
              color="primary"
              startIcon={loading ? <CircularProgress size={20} /> : <DebugIcon />}
              onClick={initDebugSession}
              disabled={loading}
            >
              Start Debugging
            </Button>
          ) : (
            <Stack direction="row" spacing={1}>
              <Tooltip title="Step">
                <DebugButton
                  variant="outlined"
                  size="small"
                  onClick={() => stepDebugSession('step')}
                  disabled={loading || debugSession.status === 'completed' || debugSession.status === 'error'}
                >
                  <StepIcon />
                </DebugButton>
              </Tooltip>
              <Tooltip title="Continue until next breakpoint">
                <DebugButton
                  variant="outlined"
                  size="small"
                  onClick={() => stepDebugSession('continue')}
                  disabled={loading || debugSession.status === 'completed' || debugSession.status === 'error'}
                >
                  <ContinueIcon />
                </DebugButton>
              </Tooltip>
              <Tooltip title="Reset">
                <DebugButton
                  variant="outlined"
                  size="small"
                  onClick={() => stepDebugSession('reset')}
                  disabled={loading}
                >
                  <ResetIcon />
                </DebugButton>
              </Tooltip>
            </Stack>
          )}
        </Box>
        
        {debugSession && (
          <Box sx={{ mb: 2 }}>
            <Chip 
              label={`Status: ${debugSession.status}`}
              color={
                debugSession.status === 'completed' ? 'success' :
                debugSession.status === 'error' ? 'error' :
                debugSession.status === 'paused' ? 'warning' :
                'primary'
              }
              sx={{ mr: 1 }}
            />
            {debugSession.current_node_id && (
              <Chip 
                label={`Current: ${getNodeLabel(debugSession.current_node_id)}`}
                color="info"
              />
            )}
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        {loading && !debugSession && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}
      </CardContent>
      
      {debugSession && (
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={tabValue} 
              onChange={handleTabChange}
              variant="fullWidth"
            >
              <Tab icon={<CodeIcon />} label="Nodes" />
              <Tab icon={<MemoryIcon />} label="Variables" />
            </Tabs>
          </Box>
          
          <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
            {tabValue === 0 && renderNodesList()}
            {tabValue === 1 && renderVariablesTab()}
          </Box>
        </Box>
      )}
    </Card>
  );
};

export default DebugPanel;