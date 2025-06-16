import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Paper,
  Chip,
  Divider,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Collapse,
  Tooltip,
  InputAdornment,
  Tab,
  Tabs,
  alpha,
  useTheme,
  Stack,
  Autocomplete,
} from '@mui/material';
import {
  Search as SearchIcon,
  TravelExplore as TravelExploreIcon,
  Hub as HubIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayArrowIcon,
  RestartAlt as RestartAltIcon,
  Tune as TuneIcon,
  FilterAlt as FilterAltIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Insights as InsightsIcon,
  Download as DownloadIcon,
  OpenInNew as OpenInNewIcon,
  ContentCopy as ContentCopyIcon,
  Bookmark as BookmarkIcon,
  BookmarkBorder as BookmarkBorderIcon,
} from '@mui/icons-material';
import VectorVisualization from './VectorVisualization';
import HumanText from './HumanText';
import { vectorStoreService, SearchResponse, SearchResult, vectorOperationsService } from '../api/services';

interface VectorExplorerProps {
  vectorTable?: string;
  initialQuery?: string;
  onVectorSelected?: (vectorId: string, metadata: Record<string, any>) => void;
}

// TabPanel component for the tabbed interface
function TabPanel(props: { children?: React.ReactNode; index: number; value: number }) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`vector-explorer-tabpanel-${index}`}
      aria-labelledby={`vector-explorer-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ height: '100%', py: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const VectorExplorer: React.FC<VectorExplorerProps> = ({
  vectorTable,
  initialQuery = '',
  onVectorSelected,
}) => {
  const theme = useTheme();
  
  // State
  const [query, setQuery] = useState<string>(initialQuery);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [k, setK] = useState<number>(10);
  const [useMMR, setUseMMR] = useState<boolean>(false);
  const [fetchK, setFetchK] = useState<number>(20);
  const [lambdaMultiplier, setLambdaMultiplier] = useState<number>(0.5);
  const [filter, setFilter] = useState<Record<string, any>>({});
  const [filterText, setFilterText] = useState<string>('');
  const [savedQueries, setSavedQueries] = useState<{ query: string; filter?: Record<string, any> }[]>([]);
  const [showMetadata, setShowMetadata] = useState<boolean>(true);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [availableMetadataFields, setAvailableMetadataFields] = useState<string[]>([]);
  const [metadataFilters, setMetadataFilters] = useState<Record<string, any>>({});
  const [embeddingType, setEmbeddingType] = useState<string>('DOCUMENT');
  const [batchVectorSearch, setBatchVectorSearch] = useState<boolean>(false);
  const [batchTexts, setBatchTexts] = useState<string[]>([]);
  const [batchInputText, setBatchInputText] = useState<string>('');
  
  // Refs
  const resultsPanelRef = useRef<HTMLDivElement>(null);
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Extract metadata fields from search results
  useEffect(() => {
    if (searchResults.length > 0) {
      const fields = new Set<string>();
      searchResults.forEach(result => {
        if (result.document && result.document.metadata) {
          Object.keys(result.document.metadata).forEach(key => fields.add(key));
        }
      });
      setAvailableMetadataFields(Array.from(fields));
    }
  }, [searchResults]);
  
  // Perform search
  const handleSearch = async () => {
    if (!query.trim() && !batchVectorSearch) {
      setError('Please enter a search query');
      return;
    }
    
    if (batchVectorSearch && batchTexts.length === 0) {
      setError('Please add at least one text for batch search');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSearchResults([]);
    
    try {
      let response: SearchResponse;
      
      // Combine filters
      const combinedFilter = { ...filter, ...metadataFilters };
      
      if (!batchVectorSearch) {
        // Regular semantic search
        if (useMMR) {
          // MMR search for diversity
          response = await vectorStoreService.mmrQuery(
            query,
            k,
            fetchK,
            lambdaMultiplier,
            combinedFilter
          ).then(res => res.data);
        } else {
          // Standard similarity search
          response = await vectorStoreService.query(
            query,
            k,
            combinedFilter
          ).then(res => res.data);
        }
      } else {
        // Create embeddings and then search by vector
        const embeddingsResponse = await vectorOperationsService.batchEmbed({
          texts: batchTexts,
          embedding_type: embeddingType,
        }).then(res => res.data);
        
        // Calculate average embedding for batch search
        const embeddingSum = new Array(embeddingsResponse.dimensions).fill(0);
        embeddingsResponse.embeddings.forEach(embedding => {
          embedding.forEach((val, i) => {
            embeddingSum[i] += val;
          });
        });
        
        const avgEmbedding = embeddingSum.map(sum => sum / embeddingsResponse.embeddings.length);
        
        // Search by vector
        response = await vectorStoreService.queryByVector(
          avgEmbedding,
          k,
          combinedFilter
        ).then(res => res.data);
      }
      
      setSearchResults(response.results);
      
      // Scroll to results
      if (resultsPanelRef.current) {
        resultsPanelRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    } catch (err) {
      console.error('Search error:', err);
      setError('Failed to perform search. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Add text to batch search
  const addToBatchSearch = () => {
    if (batchInputText.trim()) {
      setBatchTexts([...batchTexts, batchInputText.trim()]);
      setBatchInputText('');
    }
  };
  
  // Remove text from batch search
  const removeFromBatchSearch = (index: number) => {
    const newTexts = [...batchTexts];
    newTexts.splice(index, 1);
    setBatchTexts(newTexts);
  };
  
  // Clear batch search
  const clearBatchSearch = () => {
    setBatchTexts([]);
    setBatchInputText('');
  };
  
  // Save current query
  const saveQuery = () => {
    if (query.trim()) {
      setSavedQueries([...savedQueries, { query, filter }]);
    }
  };
  
  // Load saved query
  const loadSavedQuery = (index: number) => {
    const savedQuery = savedQueries[index];
    setQuery(savedQuery.query);
    if (savedQuery.filter) {
      setFilter(savedQuery.filter);
    }
  };
  
  // Apply metadata filter
  const applyMetadataFilter = (field: string, value: any) => {
    setMetadataFilters({
      ...metadataFilters,
      [field]: value
    });
  };
  
  // Remove metadata filter
  const removeMetadataFilter = (field: string) => {
    const newFilters = { ...metadataFilters };
    delete newFilters[field];
    setMetadataFilters(newFilters);
  };
  
  // Handle filter input
  const handleFilterChange = () => {
    try {
      if (filterText.trim()) {
        // Try to parse as JSON
        try {
          const filterObj = JSON.parse(filterText);
          setFilter(filterObj);
        } catch (e) {
          // If not valid JSON, check if it's a key:value format
          if (filterText.includes(':')) {
            const [key, value] = filterText.split(':').map(part => part.trim());
            setFilter({ [key]: value });
          } else {
            // Treat as a plain text filter
            setFilter({ content: filterText });
          }
        }
      } else {
        setFilter({});
      }
    } catch (err) {
      console.error('Filter error:', err);
      setError('Invalid filter format. Please check your input.');
    }
  };
  
  // Select a result for detailed view
  const handleResultSelect = (result: SearchResult) => {
    setSelectedResult(result);
    if (onVectorSelected && result.document && result.document.metadata) {
      onVectorSelected(
        result.document.metadata.id || 'unknown',
        result.document.metadata
      );
    }
  };
  
  // Render the search interface
  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <CardContent sx={{ p: 0, flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            aria-label="vector explorer tabs"
            sx={{ 
              px: 2, 
              '& .MuiTab-root': { 
                minHeight: '48px',
                fontSize: '0.875rem',
                fontWeight: 500,
              } 
            }}
          >
            <Tab 
              icon={<SearchIcon sx={{ fontSize: '1.1rem', mr: 1 }} />} 
              iconPosition="start" 
              label="Semantic Search" 
              id="vector-explorer-tab-0" 
              aria-controls="vector-explorer-tabpanel-0" 
            />
            <Tab 
              icon={<TravelExploreIcon sx={{ fontSize: '1.1rem', mr: 1 }} />} 
              iconPosition="start" 
              label="Vector Space" 
              id="vector-explorer-tab-1" 
              aria-controls="vector-explorer-tabpanel-1" 
            />
            <Tab 
              icon={<HubIcon sx={{ fontSize: '1.1rem', mr: 1 }} />} 
              iconPosition="start" 
              label="Connections" 
              id="vector-explorer-tab-2" 
              aria-controls="vector-explorer-tabpanel-2" 
            />
          </Tabs>
        </Box>
        
        <Box sx={{ flexGrow: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          {/* Semantic Search Tab */}
          <TabPanel value={tabValue} index={0}>
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              {/* Search Input Area */}
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  mb: 2,
                  border: '1px solid',
                  borderColor: alpha(theme.palette.divider, 0.8),
                  borderRadius: 2,
                }}
              >
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <HumanText variant="h6" sx={{ fontWeight: 500 }}>
                        Vector Search
                      </HumanText>
                      <Box>
                        <Tooltip title="Search Settings">
                          <IconButton 
                            onClick={() => setShowSettings(!showSettings)}
                            color={showSettings ? 'primary' : 'default'}
                          >
                            <SettingsIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Batch Search">
                          <IconButton 
                            onClick={() => setBatchVectorSearch(!batchVectorSearch)}
                            color={batchVectorSearch ? 'primary' : 'default'}
                          >
                            <TuneIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                  </Grid>
                  
                  {!batchVectorSearch ? (
                    // Standard search input
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        variant="outlined"
                        label="Search Query"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Enter your semantic search query..."
                        InputProps={{
                          endAdornment: (
                            <InputAdornment position="end">
                              <Tooltip title="Save Query">
                                <IconButton
                                  onClick={saveQuery}
                                  disabled={!query.trim()}
                                  edge="end"
                                >
                                  <BookmarkBorderIcon />
                                </IconButton>
                              </Tooltip>
                            </InputAdornment>
                          ),
                        }}
                      />
                    </Grid>
                  ) : (
                    // Batch search input
                    <Grid item xs={12}>
                      <HumanText variant="subtitle2" sx={{ mb: 1 }}>
                        Batch Search
                      </HumanText>
                      <Alert severity="info" sx={{ mb: 2 }}>
                        Add multiple texts to find vectors similar to the average embedding of all texts.
                      </Alert>
                      <Box sx={{ mb: 2 }}>
                        <TextField
                          fullWidth
                          variant="outlined"
                          label="Add Text"
                          value={batchInputText}
                          onChange={(e) => setBatchInputText(e.target.value)}
                          placeholder="Enter a text to add to the batch..."
                          InputProps={{
                            endAdornment: (
                              <InputAdornment position="end">
                                <Button
                                  variant="contained"
                                  onClick={addToBatchSearch}
                                  disabled={!batchInputText.trim()}
                                  size="small"
                                >
                                  Add
                                </Button>
                              </InputAdornment>
                            ),
                          }}
                        />
                      </Box>
                      
                      {/* Batch texts list */}
                      {batchTexts.length > 0 && (
                        <Box 
                          sx={{ 
                            maxHeight: '150px', 
                            overflowY: 'auto', 
                            border: '1px solid',
                            borderColor: alpha(theme.palette.divider, 0.5),
                            borderRadius: 1,
                            p: 1,
                            mb: 2,
                          }}
                        >
                          <List dense disablePadding>
                            {batchTexts.map((text, index) => (
                              <ListItem
                                key={index}
                                secondaryAction={
                                  <IconButton
                                    edge="end"
                                    aria-label="delete"
                                    onClick={() => removeFromBatchSearch(index)}
                                    size="small"
                                  >
                                    <ContentCopyIcon fontSize="small" />
                                  </IconButton>
                                }
                                sx={{ 
                                  py: 0.5, 
                                  borderBottom: index < batchTexts.length - 1 ? `1px solid ${alpha(theme.palette.divider, 0.2)}` : 'none',
                                }}
                              >
                                <ListItemText
                                  primary={
                                    <Typography variant="body2" noWrap>
                                      {text.length > 60 ? `${text.substring(0, 60)}...` : text}
                                    </Typography>
                                  }
                                />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <FormControl size="small" sx={{ minWidth: 150 }}>
                          <InputLabel id="embedding-type-label">Embedding Type</InputLabel>
                          <Select
                            labelId="embedding-type-label"
                            value={embeddingType}
                            label="Embedding Type"
                            onChange={(e) => setEmbeddingType(e.target.value)}
                          >
                            <MenuItem value="DOCUMENT">DOCUMENT</MenuItem>
                            <MenuItem value="QUERY">QUERY</MenuItem>
                            <MenuItem value="CODE">CODE</MenuItem>
                          </Select>
                        </FormControl>
                        
                        <Button
                          variant="outlined"
                          color="secondary"
                          onClick={clearBatchSearch}
                          disabled={batchTexts.length === 0}
                          startIcon={<RestartAltIcon />}
                        >
                          Clear All
                        </Button>
                      </Box>
                    </Grid>
                  )}
                  
                  {/* Search button */}
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <HumanText variant="caption" color="text.secondary">
                        {batchVectorSearch 
                          ? `${batchTexts.length} text(s) in batch` 
                          : `Searching in ${vectorTable || 'vector store'}`}
                      </HumanText>
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={handleSearch}
                        disabled={loading || (!query.trim() && !batchVectorSearch) || (batchVectorSearch && batchTexts.length === 0)}
                        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
                      >
                        {loading ? 'Searching...' : 'Search'}
                      </Button>
                    </Box>
                  </Grid>
                  
                  {/* Search settings */}
                  <Grid item xs={12}>
                    <Collapse in={showSettings}>
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          mt: 1,
                          bgcolor: alpha(theme.palette.background.default, 0.5),
                          border: '1px solid',
                          borderColor: alpha(theme.palette.divider, 0.3),
                          borderRadius: 1,
                        }}
                      >
                        <Grid container spacing={2}>
                          <Grid item xs={12} sm={6}>
                            <FormControl fullWidth size="small">
                              <HumanText gutterBottom>
                                Results: {k}
                              </HumanText>
                              <Slider
                                value={k}
                                onChange={(_, newValue) => setK(newValue as number)}
                                min={1}
                                max={50}
                                step={1}
                                marks={[
                                  { value: 1, label: '1' },
                                  { value: 10, label: '10' },
                                  { value: 20, label: '20' },
                                  { value: 50, label: '50' },
                                ]}
                                valueLabelDisplay="auto"
                              />
                            </FormControl>
                          </Grid>
                          
                          <Grid item xs={12} sm={6}>
                            <FormControlLabel
                              control={
                                <Switch
                                  checked={useMMR}
                                  onChange={(e) => setUseMMR(e.target.checked)}
                                  color="primary"
                                />
                              }
                              label={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                  <HumanText>Use MMR for Diversity</HumanText>
                                  <Tooltip title="Maximal Marginal Relevance ensures more diverse results">
                                    <InfoIcon fontSize="small" color="action" />
                                  </Tooltip>
                                </Box>
                              }
                            />
                          </Grid>
                          
                          {useMMR && (
                            <>
                              <Grid item xs={12} sm={6}>
                                <FormControl fullWidth size="small">
                                  <HumanText gutterBottom>
                                    Fetch K: {fetchK}
                                  </HumanText>
                                  <Slider
                                    value={fetchK}
                                    onChange={(_, newValue) => setFetchK(newValue as number)}
                                    min={k}
                                    max={100}
                                    step={5}
                                    marks={[
                                      { value: k, label: `${k}` },
                                      { value: 50, label: '50' },
                                      { value: 100, label: '100' },
                                    ]}
                                    valueLabelDisplay="auto"
                                  />
                                </FormControl>
                              </Grid>
                              
                              <Grid item xs={12} sm={6}>
                                <FormControl fullWidth size="small">
                                  <HumanText gutterBottom>
                                    Diversity (λ): {lambdaMultiplier}
                                  </HumanText>
                                  <Slider
                                    value={lambdaMultiplier}
                                    onChange={(_, newValue) => setLambdaMultiplier(newValue as number)}
                                    min={0}
                                    max={1}
                                    step={0.1}
                                    marks={[
                                      { value: 0, label: 'Diverse' },
                                      { value: 0.5, label: 'Balanced' },
                                      { value: 1, label: 'Relevant' },
                                    ]}
                                    valueLabelDisplay="auto"
                                  />
                                </FormControl>
                              </Grid>
                            </>
                          )}
                          
                          <Grid item xs={12}>
                            <TextField
                              fullWidth
                              size="small"
                              label="Filter"
                              value={filterText}
                              onChange={(e) => setFilterText(e.target.value)}
                              placeholder="Filter by metadata (e.g. 'category:finance' or JSON)"
                              InputProps={{
                                endAdornment: (
                                  <InputAdornment position="end">
                                    <Button
                                      variant="text"
                                      onClick={handleFilterChange}
                                      size="small"
                                    >
                                      Apply
                                    </Button>
                                  </InputAdornment>
                                ),
                              }}
                            />
                          </Grid>
                          
                          {/* Active filters */}
                          {(Object.keys(filter).length > 0 || Object.keys(metadataFilters).length > 0) && (
                            <Grid item xs={12}>
                              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
                                {Object.entries(filter).map(([key, value]) => (
                                  <Chip
                                    key={key}
                                    label={`${key}: ${value}`}
                                    onDelete={() => {
                                      const newFilter = { ...filter };
                                      delete newFilter[key];
                                      setFilter(newFilter);
                                    }}
                                    size="small"
                                    color="primary"
                                    variant="outlined"
                                  />
                                ))}
                                
                                {Object.entries(metadataFilters).map(([key, value]) => (
                                  <Chip
                                    key={key}
                                    label={`${key}: ${value}`}
                                    onDelete={() => removeMetadataFilter(key)}
                                    size="small"
                                    color="secondary"
                                    variant="outlined"
                                  />
                                ))}
                              </Box>
                            </Grid>
                          )}
                          
                          {/* Saved queries */}
                          {savedQueries.length > 0 && (
                            <Grid item xs={12}>
                              <HumanText variant="subtitle2" gutterBottom>
                                Saved Queries
                              </HumanText>
                              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                {savedQueries.map((saved, index) => (
                                  <Chip
                                    key={index}
                                    label={saved.query}
                                    onClick={() => loadSavedQuery(index)}
                                    icon={<BookmarkIcon fontSize="small" />}
                                    color="default"
                                    size="small"
                                  />
                                ))}
                              </Box>
                            </Grid>
                          )}
                        </Grid>
                      </Paper>
                    </Collapse>
                  </Grid>
                </Grid>
              </Paper>
              
              {/* Search Results */}
              <Box
                ref={resultsPanelRef}
                sx={{ 
                  flexGrow: 1, 
                  overflow: 'auto', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  mt: 2 
                }}
              >
                {error && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                  </Alert>
                )}
                
                {searchResults.length > 0 ? (
                  <Box sx={{ display: 'flex', height: '100%' }}>
                    {/* Results list */}
                    <Box sx={{ width: '50%', pr: 2, overflow: 'auto' }}>
                      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <HumanText variant="subtitle1" fontWeight={500}>
                          Search Results ({searchResults.length})
                        </HumanText>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={showMetadata}
                              onChange={(e) => setShowMetadata(e.target.checked)}
                              size="small"
                            />
                          }
                          label={
                            <HumanText variant="body2">
                              Show Metadata
                            </HumanText>
                          }
                        />
                      </Box>
                      
                      <Stack spacing={2}>
                        {searchResults.map((result, index) => (
                          <Paper
                            key={index}
                            elevation={0}
                            sx={{
                              p: 2,
                              border: '1px solid',
                              borderColor: selectedResult === result
                                ? theme.palette.primary.main
                                : alpha(theme.palette.divider, 0.5),
                              borderRadius: 2,
                              cursor: 'pointer',
                              transition: 'all 0.2s',
                              '&:hover': {
                                borderColor: alpha(theme.palette.primary.main, 0.5),
                                bgcolor: alpha(theme.palette.primary.main, 0.02),
                                transform: 'translateY(-2px)',
                                boxShadow: '0 4px 8px rgba(0, 0, 0, 0.05)',
                              },
                            }}
                            onClick={() => handleResultSelect(result)}
                          >
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Chip
                                label={`Score: ${result.score.toFixed(3)}`}
                                size="small"
                                color={
                                  result.score > 0.9 ? 'success' :
                                  result.score > 0.7 ? 'primary' :
                                  result.score > 0.5 ? 'info' : 'default'
                                }
                                variant="outlined"
                              />
                              <HumanText variant="caption" color="text.secondary">
                                #{index + 1}
                              </HumanText>
                            </Box>
                            
                            <HumanText 
                              variant="body2" 
                              sx={{ 
                                mb: 1,
                                maxHeight: '100px',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                display: '-webkit-box',
                                WebkitLineClamp: 4,
                                WebkitBoxOrient: 'vertical',
                              }}
                            >
                              {result.document.page_content}
                            </HumanText>
                            
                            {showMetadata && result.document.metadata && Object.keys(result.document.metadata).length > 0 && (
                              <Box 
                                sx={{ 
                                  mt: 1, 
                                  pt: 1, 
                                  borderTop: '1px dashed',
                                  borderColor: alpha(theme.palette.divider, 0.5),
                                }}
                              >
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                  {Object.entries(result.document.metadata)
                                    .filter(([key, _]) => key !== 'id' && key !== 'vector')
                                    .slice(0, 3)
                                    .map(([key, value]) => (
                                      <Chip
                                        key={key}
                                        label={`${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`}
                                        size="small"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          applyMetadataFilter(key, value);
                                        }}
                                        sx={{ 
                                          fontSize: '0.7rem',
                                          height: 20,
                                          '& .MuiChip-label': { px: 1 },
                                        }}
                                      />
                                    ))}
                                  
                                  {Object.keys(result.document.metadata).length > 3 && (
                                    <Chip
                                      label={`+${Object.keys(result.document.metadata).length - 3} more`}
                                      size="small"
                                      sx={{ 
                                        fontSize: '0.7rem',
                                        height: 20,
                                        '& .MuiChip-label': { px: 1 },
                                      }}
                                    />
                                  )}
                                </Box>
                              </Box>
                            )}
                          </Paper>
                        ))}
                      </Stack>
                    </Box>
                    
                    {/* Detail view */}
                    <Box sx={{ width: '50%', borderLeft: '1px solid', borderColor: 'divider', pl: 2 }}>
                      {selectedResult ? (
                        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                          <HumanText variant="subtitle1" fontWeight={500} gutterBottom>
                            Document Details
                          </HumanText>
                          
                          <Paper
                            elevation={0}
                            sx={{
                              p: 2,
                              border: '1px solid',
                              borderColor: alpha(theme.palette.divider, 0.5),
                              borderRadius: 2,
                              flexGrow: 1,
                              overflowY: 'auto',
                            }}
                          >
                            <Box sx={{ mb: 2 }}>
                              <HumanText variant="subtitle2" gutterBottom>
                                Content
                              </HumanText>
                              <Box 
                                sx={{ 
                                  p: 2, 
                                  bgcolor: alpha(theme.palette.background.default, 0.5),
                                  border: '1px solid',
                                  borderColor: alpha(theme.palette.divider, 0.2),
                                  borderRadius: 1,
                                  whiteSpace: 'pre-wrap',
                                  overflowWrap: 'break-word',
                                }}
                              >
                                <HumanText variant="body2">
                                  {selectedResult.document.page_content}
                                </HumanText>
                              </Box>
                            </Box>
                            
                            <Divider sx={{ my: 2 }} />
                            
                            <Box>
                              <HumanText variant="subtitle2" gutterBottom>
                                Metadata
                              </HumanText>
                              
                              {selectedResult.document.metadata && Object.keys(selectedResult.document.metadata).length > 0 ? (
                                <Box 
                                  sx={{ 
                                    borderRadius: 1,
                                    border: '1px solid',
                                    borderColor: alpha(theme.palette.divider, 0.2),
                                    overflow: 'hidden',
                                  }}
                                >
                                  {Object.entries(selectedResult.document.metadata)
                                    .filter(([key, _]) => key !== 'vector')
                                    .map(([key, value], index, arr) => (
                                      <Box 
                                        key={key}
                                        sx={{ 
                                          display: 'flex',
                                          borderBottom: index < arr.length - 1 ? `1px solid ${alpha(theme.palette.divider, 0.1)}` : 'none',
                                        }}
                                      >
                                        <Box 
                                          sx={{ 
                                            width: '30%', 
                                            p: 1.5,
                                            bgcolor: alpha(theme.palette.background.default, 0.5),
                                          }}
                                        >
                                          <HumanText variant="caption" fontWeight={500}>
                                            {key}
                                          </HumanText>
                                        </Box>
                                        <Box 
                                          sx={{ 
                                            width: '70%', 
                                            p: 1.5,
                                            whiteSpace: 'pre-wrap',
                                            overflowWrap: 'break-word',
                                          }}
                                        >
                                          <HumanText variant="caption">
                                            {typeof value === 'object' 
                                              ? JSON.stringify(value, null, 2) 
                                              : String(value)
                                            }
                                          </HumanText>
                                        </Box>
                                      </Box>
                                    ))}
                                </Box>
                              ) : (
                                <Alert severity="info" variant="outlined">
                                  No metadata available
                                </Alert>
                              )}
                            </Box>
                            
                            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
                              <Button
                                variant="outlined"
                                size="small"
                                startIcon={<ContentCopyIcon />}
                                onClick={() => {
                                  navigator.clipboard.writeText(selectedResult.document.page_content);
                                }}
                              >
                                Copy Content
                              </Button>
                              
                              <Button
                                variant="outlined"
                                size="small"
                                startIcon={<InsightsIcon />}
                                onClick={() => {
                                  // Handle showing similar vectors
                                }}
                              >
                                Find Similar
                              </Button>
                            </Box>
                          </Paper>
                        </Box>
                      ) : (
                        <Box 
                          sx={{ 
                            height: '100%', 
                            display: 'flex', 
                            flexDirection: 'column', 
                            justifyContent: 'center', 
                            alignItems: 'center',
                            color: 'text.secondary',
                          }}
                        >
                          <TravelExploreIcon sx={{ fontSize: 48, opacity: 0.3, mb: 2 }} />
                          <HumanText variant="body2">
                            Select a result to view details
                          </HumanText>
                        </Box>
                      )}
                    </Box>
                  </Box>
                ) : !loading && (
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      flexDirection: 'column', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      height: '100%',
                      opacity: 0.7,
                    }}
                  >
                    <SearchIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <HumanText variant="body1" color="text.secondary">
                      {query.trim() ? 'No results found.' : 'Enter a query to search.'}
                    </HumanText>
                    <HumanText variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      {query.trim() ? 'Try a different query or adjust your filters.' : 'You can search by text or by metadata filters.'}
                    </HumanText>
                  </Box>
                )}
              </Box>
            </Box>
          </TabPanel>
          
          {/* Vector Space Tab */}
          <TabPanel value={tabValue} index={1}>
            <Box sx={{ height: '100%' }}>
              <VectorVisualization
                tableName={vectorTable}
                maxPoints={500}
                filter={filter}
              />
            </Box>
          </TabPanel>
          
          {/* Connections Tab */}
          <TabPanel value={tabValue} index={2}>
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Alert severity="info" sx={{ mb: 3 }}>
                This tab shows the connections between vectors and their source data, allowing you to trace lineage.
              </Alert>
              
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  flexGrow: 1,
                  border: '1px solid',
                  borderColor: alpha(theme.palette.divider, 0.5),
                  borderRadius: 2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Box sx={{ textAlign: 'center' }}>
                  <HubIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <HumanText variant="h6" color="text.secondary" gutterBottom>
                    Vector Connections
                  </HumanText>
                  <HumanText variant="body2" color="text.secondary">
                    This feature will be implemented in the next phase to show vector-to-source mappings.
                  </HumanText>
                </Box>
              </Paper>
            </Box>
          </TabPanel>
        </Box>
      </CardContent>
    </Card>
  );
};

export default VectorExplorer;