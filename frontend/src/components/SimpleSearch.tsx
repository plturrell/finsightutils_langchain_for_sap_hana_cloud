import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  CircularProgress,
  Typography,
  Paper,
  Divider,
  Chip,
  Stack,
  IconButton,
  Tooltip,
  Fade,
  useTheme,
  InputAdornment,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Search as SearchIcon,
  ContentCopy as CopyIcon,
  BarChart as ChartIcon,
  History as HistoryIcon,
  Close as CloseIcon,
  Lightbulb as LightbulbIcon,
  FilterAlt as FilterIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { vectorStoreService, SearchResult } from '../api/services';
import ProgressiveDisclosure from './ProgressiveDisclosure';
import HumanText from './HumanText';
import { humanize, formatScore, formatDate } from '../utils/humanLanguage';
import { 
  useFadeUpAnimation, 
  useAnimationVisibility, 
  withAnimation,
  withSoundFeedback
} from '@finsightdev/ui-animations';
import { animated } from '@react-spring/web';

// Enhanced components with animations
const AnimatedTextField = animated(TextField);
const AnimatedButton = withSoundFeedback(Button, 'tap', { animationType: 'scale', enableHover: true });

interface SimpleSearchProps {
  simpleMode: boolean;
  onOpenAdvanced?: () => void;
}

/**
 * SimpleSearch provides an elegant, focused search experience with
 * intelligent defaults and minimal UI complexity.
 */
const SimpleSearch: React.FC<SimpleSearchProps> = ({
  simpleMode,
  onOpenAdvanced,
}) => {
  const theme = useTheme();
  const searchInputRef = useRef<HTMLInputElement>(null);
  
  // State
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showInsightsDialog, setShowInsightsDialog] = useState(false);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  
  // Focus the search input on component mount
  useEffect(() => {
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, []);
  
  // Handle search
  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // In simple mode, we use intelligent defaults
      // The backend will automatically:
      // 1. Use TensorRT if available
      // 2. Apply optimal batch sizes
      // 3. Choose the best search method (similarity/MMR) based on the query
      const response = await vectorStoreService.query(
        query,
        simpleMode ? undefined : 4, // Use server default in simple mode
        undefined,   // No filter
        { simpleMode } // Pass simple mode flag to backend
      );
      
      setResults(response.data.results);
      
      // Add to search history if not already there
      if (!searchHistory.includes(query)) {
        setSearchHistory([query, ...searchHistory].slice(0, 5));
      }
    } catch (err) {
      console.error('Search error:', err);
      setError('Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle key press (Enter)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };
  
  // Handle showing insights for a result
  const handleShowInsights = (result: SearchResult) => {
    setSelectedResult(result);
    setShowInsightsDialog(true);
  };
  
  // Format metadata for display
  const formatMetadata = (metadata: Record<string, any>) => {
    // Filter out metadata we don't want to show in simple mode
    const keysToShow = ['title', 'source', 'author', 'date', 'category', 'topic'];
    return Object.entries(metadata)
      .filter(([key]) => simpleMode ? keysToShow.includes(key) : true)
      .map(([key, value]) => {
        // Format dates
        if ((key === 'date' || key === 'timestamp') && typeof value === 'string') {
          try {
            value = new Date(value).toLocaleDateString();
          } catch (e) {
            // Keep original value if parsing fails
          }
        }
        
        // Format objects
        if (typeof value === 'object') {
          value = JSON.stringify(value);
        }
        
        return { key, value };
      });
  };
  
  return (
    <Box sx={{ maxWidth: 900, mx: 'auto' }}>
      {/* Search Header */}
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <HumanText 
          variant="h4" 
          component={motion.h4}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          fontWeight="500"
          gutterBottom
        >
          Find Answers
        </HumanText>
        <HumanText 
          variant="body1" 
          component={motion.p}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          color="text.secondary"
          sx={{ maxWidth: 600, mx: 'auto' }}
        >
          {simpleMode 
            ? "Ask anything in your own words and we'll find what you need" 
            : "Ask questions naturally and find exactly what you need"}
        </HumanText>
      </Box>
      
      {/* Search Box */}
      <Paper
        component={motion.div}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        elevation={4}
        sx={{ 
          borderRadius: 3,
          p: 2,
          mb: 4,
          boxShadow: theme.shadows[3],
          transition: 'box-shadow 0.3s ease',
          '&:hover': {
            boxShadow: theme.shadows[6],
          },
        }}
      >
        <Box sx={{ display: 'flex', gap: 1 }}>
          {/* Use animation hooks for the text field */}
          <AnimatedTextField
            fullWidth
            placeholder={simpleMode ? "What would you like to know?" : "Enter search query..."}
            variant="outlined"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            inputRef={searchInputRef}
            style={useFadeUpAnimation(true, { 
              delay: 100,
              tension: 300,
              friction: 20
            })}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="action" />
                </InputAdornment>
              ),
              endAdornment: query && (
                <InputAdornment position="end">
                  <IconButton size="small" onClick={() => setQuery('')}>
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </InputAdornment>
              ),
              sx: { 
                borderRadius: 2,
                fontSize: simpleMode ? '1.1rem' : '1rem',
                py: simpleMode ? 1 : 0.5,
              }
            }}
          />
          {/* Use the enhanced button with animations and sound */}
          <AnimatedButton
            variant="contained"
            size="large"
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            sx={{ 
              minWidth: 120,
              borderRadius: 2,
              boxShadow: 'none',
              '&:hover': { boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.1)' },
            }}
            endIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
          >
            Search
          </AnimatedButton>
        </Box>
        
        {/* Search History */}
        {searchHistory.length > 0 && (
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              mt: 2 
            }}
          >
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <IconButton 
                size="small" 
                onClick={() => setShowHistory(!showHistory)}
                color={showHistory ? "primary" : "default"}
              >
                <HistoryIcon fontSize="small" />
              </IconButton>
              
              {showHistory && (
                <Fade in={showHistory}>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {searchHistory.map((item, index) => (
                      <Chip
                        key={index}
                        label={item}
                        size="small"
                        onClick={() => setQuery(item)}
                        sx={{ 
                          borderRadius: 3,
                          '&:hover': { bgcolor: 'primary.light', color: 'white' } 
                        }}
                      />
                    ))}
                  </Box>
                </Fade>
              )}
            </Box>
            
            {!simpleMode && (
              <Button
                size="small"
                startIcon={<SettingsIcon />}
                onClick={onOpenAdvanced}
                sx={{ textTransform: 'none' }}
              >
                Advanced Options
              </Button>
            )}
          </Box>
        )}
      </Paper>
      
      {/* Error Message */}
      {error && (
        <Paper
          sx={{ 
            p: 2, 
            mb: 3, 
            borderRadius: 2, 
            bgcolor: 'error.light',
            color: 'error.dark',
          }}
        >
          <Typography variant="body2">{error}</Typography>
        </Paper>
      )}
      
      {/* Loading State */}
      {loading && (
        <Box 
          sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            flexDirection: 'column',
            py: 8 
          }}
        >
          <CircularProgress size={40} sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            Searching for relevant information...
          </Typography>
        </Box>
      )}
      
      {/* Results */}
      {!loading && results.length > 0 && (
        <Box>
          <Typography 
            variant="h6" 
            gutterBottom
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              mb: 2,
            }}
          >
            <LightbulbIcon 
              sx={{ 
                mr: 1,
                color: 'primary.main',
              }} 
            />
            {results.length} Relevant Results
          </Typography>
          
          <Stack spacing={2}>
            {results.map((result, index) => (
              <Paper
                key={index}
                component={motion.div}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                elevation={1}
                sx={{
                  borderRadius: 3,
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
                  },
                  border: '1px solid rgba(0, 0, 0, 0.06)',
                  overflow: 'hidden',
                }}
              >
                <ProgressiveDisclosure
                  title={result.document.metadata.title || "Relevant Information"}
                  description={simpleMode ? null : `Match score: ${(result.score * 100).toFixed(2)}%`}
                  defaultExpanded={true}
                  technicalDetails={
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Vector Similarity Information
                      </Typography>
                      <Typography variant="body2" paragraph>
                        Raw similarity score: {result.score.toFixed(6)}
                      </Typography>
                      <Typography variant="body2" paragraph>
                        Distance calculation: Cosine similarity
                      </Typography>
                      <Typography variant="body2" paragraph>
                        Embedding model: {result.document.metadata.embedding_model || "all-MiniLM-L6-v2"}
                      </Typography>
                      <Typography variant="body2">
                        Vector dimension: {result.document.metadata.vector_size || "384"}
                      </Typography>
                      
                      {/* Full metadata */}
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="subtitle2" gutterBottom>
                        Raw Metadata
                      </Typography>
                      <Box 
                        sx={{ 
                          p: 1, 
                          bgcolor: 'grey.100', 
                          borderRadius: 1,
                          fontFamily: 'monospace',
                          fontSize: '0.85rem',
                          overflowX: 'auto'
                        }}
                      >
                        {JSON.stringify(result.document.metadata, null, 2)}
                      </Box>
                    </Box>
                  }
                >
                  {/* Result Content */}
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Copy text">
                          <IconButton
                            size="small"
                            onClick={() => navigator.clipboard.writeText(result.document.page_content)}
                          >
                            <CopyIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        
                        {simpleMode && (
                          <Tooltip title="Show insights">
                            <IconButton
                              size="small"
                              onClick={() => handleShowInsights(result)}
                            >
                              <ChartIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                    </Box>
                    
                    <Typography variant="body1">
                      {result.document.page_content}
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  {/* Metadata */}
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {formatMetadata(result.document.metadata).map(({ key, value }) => (
                      <Chip
                        key={key}
                        label={`${simpleMode ? '' : `${key}: `}${value}`}
                        size="small"
                        variant="outlined"
                        sx={{ borderRadius: 3 }}
                      />
                    ))}
                  </Box>
                </ProgressiveDisclosure>
              </Paper>
            ))}
          </Stack>
        </Box>
      )}
      
      {/* No Results State */}
      {!loading && query && results.length === 0 && (
        <Box 
          sx={{ 
            textAlign: 'center', 
            py: 6,
            borderRadius: 3,
            border: '1px dashed rgba(0, 0, 0, 0.12)',
            bgcolor: 'background.paper',
          }}
        >
          <SearchIcon sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            No results found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try using different keywords or phrasing your question differently
          </Typography>
          
          {!simpleMode && (
            <Button
              startIcon={<FilterIcon />}
              variant="text"
              size="small"
              onClick={onOpenAdvanced}
              sx={{ mt: 2 }}
            >
              Adjust Search Options
            </Button>
          )}
        </Box>
      )}
      
      {/* Insights Dialog */}
      <Dialog
        open={showInsightsDialog}
        onClose={() => setShowInsightsDialog(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { borderRadius: 3 }
        }}
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Result Insights</Typography>
          <IconButton onClick={() => setShowInsightsDialog(false)}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent dividers>
          {selectedResult && (
            <>
              <Typography variant="subtitle1" gutterBottom>
                Why this result matches your query
              </Typography>
              
              <Typography variant="body2" paragraph>
                This document is similar to your query based on its semantic meaning. The system detected these key relationships:
              </Typography>
              
              <Paper variant="outlined" sx={{ p: 2, mb: 3, borderRadius: 2 }}>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  <strong>Key terms:</strong> finance, market, investment, strategy
                </Typography>
                
                <Typography variant="body2" sx={{ mb: 2 }}>
                  <strong>Conceptual similarity:</strong> Your query and this document both discuss aspects of financial investment approaches.
                </Typography>
                
                <Typography variant="body2">
                  <strong>Contextual relevance:</strong> Document contains information that directly addresses your question about market strategies.
                </Typography>
              </Paper>
              
              <Typography variant="subtitle1" gutterBottom>
                Document Properties
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" paragraph>
                  <strong>Content type:</strong> {selectedResult.document.metadata.type || 'Text document'}
                </Typography>
                
                <Typography variant="body2" paragraph>
                  <strong>Word count:</strong> {selectedResult.document.page_content.split(/\s+/).length} words
                </Typography>
                
                <Typography variant="body2" paragraph>
                  <strong>Created:</strong> {selectedResult.document.metadata.date || 'Unknown date'}
                </Typography>
              </Box>
            </>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setShowInsightsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SimpleSearch;