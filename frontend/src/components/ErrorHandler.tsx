import React, { useState } from 'react';
import { 
  Alert, 
  AlertTitle, 
  Collapse, 
  Paper, 
  Typography, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  IconButton,
  Box,
  Divider,
  Chip
} from '@mui/material';
import { 
  ErrorOutline as ErrorIcon,
  Close as CloseIcon,
  ExpandMore as ExpandMoreIcon,
  LightbulbOutlined as TipIcon,
  BugReport as BugIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import HumanText from './HumanText';
import { humanize } from '../utils/humanLanguage';

export interface ErrorDetail {
  message: string;
  operation?: string;
  suggestions?: string[];
  common_issues?: string[];
  original_error?: string;
  query_info?: Record<string, any>;
  flow_info?: Record<string, any>;
  insertion_info?: Record<string, any>;
  visualization_params?: Record<string, any>;
}

export interface ApiError {
  status: number;
  statusText: string;
  detail: ErrorDetail;
}

interface ErrorHandlerProps {
  error: ApiError | null;
  onClose?: () => void;
}

const ErrorHandler: React.FC<ErrorHandlerProps> = ({ error, onClose }) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!error) return null;

  const { status, statusText, detail } = error;
  const { 
    message, 
    operation, 
    suggestions, 
    common_issues, 
    original_error,
    query_info,
    flow_info,
    insertion_info,
    visualization_params
  } = detail || {};

  // Determine severity based on status code
  const getSeverity = (status: number) => {
    if (status >= 500) return 'error';
    if (status >= 400) return 'warning';
    return 'info';
  };

  const severity = getSeverity(status);

  // Show contextual info based on operation type
  const getContextInfo = () => {
    if (query_info) return { label: 'Query Info', data: query_info };
    if (flow_info) return { label: 'Flow Info', data: flow_info };
    if (insertion_info) return { label: 'Insertion Info', data: insertion_info };
    if (visualization_params) return { label: 'Visualization Parameters', data: visualization_params };
    return null;
  };

  const contextInfo = getContextInfo();

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        mb: 3, 
        overflow: 'hidden',
        border: `1px solid ${severity === 'error' ? '#d32f2f' : severity === 'warning' ? '#ed6c02' : '#0288d1'}`
      }}
    >
      <Alert 
        severity={severity as "error" | "warning" | "info"} 
        variant="filled"
        icon={<ErrorIcon />}
        action={
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {onClose && (
              <IconButton
                color="inherit"
                size="small"
                onClick={onClose}
                aria-label="close"
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            )}
          </Box>
        }
        sx={{ borderRadius: 0 }}
      >
        <AlertTitle>
          {operation ? `Error in ${operation}` : `${status} ${statusText}`}
          {operation && <Chip size="small" label={`Status: ${status}`} sx={{ ml: 1, fontSize: '0.7rem' }} />}
        </AlertTitle>
        <HumanText variant="body1">{message || 'An unexpected error occurred.'}</HumanText>
      </Alert>

      <Box sx={{ p: 2 }}>
        {suggestions && suggestions.length > 0 && (
          <>
            <HumanText variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <TipIcon fontSize="small" sx={{ mr: 1 }} />
              Suggested Actions:
            </HumanText>
            <List dense sx={{ pl: 2 }}>
              {suggestions.map((suggestion, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemText primary={suggestion} />
                </ListItem>
              ))}
            </List>
          </>
        )}

        {common_issues && common_issues.length > 0 && (
          <>
            <Divider sx={{ my: 1.5 }} />
            <HumanText variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <BugIcon fontSize="small" sx={{ mr: 1 }} />
              Common Issues:
            </HumanText>
            <List dense sx={{ pl: 2 }}>
              {common_issues.map((issue, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemText primary={issue} />
                </ListItem>
              ))}
            </List>
          </>
        )}

        {contextInfo && (
          <>
            <Divider sx={{ my: 1.5 }} />
            <HumanText variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <InfoIcon fontSize="small" sx={{ mr: 1 }} />
              {contextInfo.label}:
            </HumanText>
            <Box 
              sx={{ 
                backgroundColor: 'rgba(0, 0, 0, 0.04)', 
                p: 1.5, 
                borderRadius: 1,
                maxHeight: '120px',
                overflow: 'auto',
                '& pre': { margin: 0 }
              }}
            >
              <pre>{JSON.stringify(contextInfo.data, null, 2)}</pre>
            </Box>
          </>
        )}

        {original_error && (
          <>
            <Divider sx={{ my: 1.5 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <HumanText 
                variant="subtitle2" 
                sx={{ 
                  color: 'text.secondary',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center'
                }}
                onClick={() => setShowDetails(!showDetails)}
              >
                Technical Details
                <ExpandMoreIcon 
                  sx={{ 
                    ml: 0.5,
                    transform: showDetails ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.3s'
                  }} 
                />
              </HumanText>
            </Box>
            <Collapse in={showDetails}>
              <Box 
                sx={{ 
                  mt: 1,
                  p: 1.5, 
                  backgroundColor: 'rgba(0, 0, 0, 0.04)', 
                  borderRadius: 1,
                  maxHeight: '100px',
                  overflow: 'auto',
                  fontSize: '0.75rem',
                  fontFamily: 'monospace'
                }}
              >
                {original_error}
              </Box>
            </Collapse>
          </>
        )}
      </Box>
    </Paper>
  );
};

export default ErrorHandler;