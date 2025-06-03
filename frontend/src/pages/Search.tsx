import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  IconButton,
  CircularProgress,
  Chip,
  Paper,
  Stack,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  Grid,
  Collapse,
  Tooltip,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Search as SearchIcon,
  ExpandMore,
  ExpandLess,
  Tune as TuneIcon,
  FilterAlt as FilterIcon,
  Settings as SettingsIcon,
  DeleteOutline as ClearIcon,
  Launch as LaunchIcon,
  ContentCopy as CopyIcon,
} from '@mui/icons-material';
import { vectorStoreService, SearchResult, SearchResponse } from '../api/services';

interface
  document: {
    page_content: string;
    metadata: Record<string, any>;
  };
  score: number;
}

// Using SearchResponse from api/services.ts

const Search: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchMethod, setSearchMethod] = useState('similarity');
  const [resultCount, setResultCount] = useState(4);
  const [mmrLambda, setMmrLambda] = useState(0.5);
  const [mmrFetchK, setMmrFetchK] = useState(20);
  const [filter, setFilter] = useState<Record<string, any>>({});
  const [filterKey, setFilterKey] = useState('');
  const [filterValue, setFilterValue] = useState('');
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [useTensorRT, setUseTensorRT] = useState(true);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      let response;
      const filter = Object.keys(filter).length > 0 ? filter : undefined;
      
      if (searchMethod === 'similarity') {
        response = await vectorStoreService.query(
          query,
          resultCount,
          filter
        );
      } else {
        // MMR search
        response = await vectorStoreService.mmrQuery(
          query,
          resultCount,
          mmrFetchK,
          mmrLambda,
          filter
        );
      }
      
      setResults(response.data.results);
      
      // Add to search history if not already there
      if (!searchHistory.includes(query)) {
        setSearchHistory([query, ...searchHistory].slice(0, 5));
      }
    } catch (err) {
      console.error('Search error:', err);
      setError('Failed to perform search. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleAddFilter = () => {
    if (filterKey && filterValue) {
      setFilter({
        ...filter,
        [filterKey]: filterValue,
      });
      setFilterKey('');
      setFilterValue('');
    }
  };

  const handleRemoveFilter = (key: string) => {
    const newFilter = { ...filter };
    delete newFilter[key];
    setFilter(newFilter);
  };

  const handleClearAll = () => {
    setQuery('');
    setResults([]);
    setFilter({});
    setFilterKey('');
    setFilterValue('');
    setSearchMethod('similarity');
    setResultCount(4);
    setMmrLambda(0.5);
    setMmrFetchK(20);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch (e) {
      return dateString;
    }
  };

  return (
    <Box className="fade-in">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="500">
          Vector Search
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Search your data using semantic similarity with GPU acceleration
        </Typography>
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              placeholder="Enter your search query..."
              variant="outlined"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              InputProps={{
                startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />,
                sx: { borderRadius: 2 }
              }}
            />
            <Button
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
            </Button>
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {searchHistory.length > 0 && (
                <>
                  <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
                    Recent:
                  </Typography>
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
                </>
              )}
            </Box>
            <Button
              startIcon={showAdvanced ? <ExpandLess /> : <ExpandMore />}
              onClick={() => setShowAdvanced(!showAdvanced)}
              size="small"
              color="primary"
              sx={{ textTransform: 'none' }}
            >
              {showAdvanced ? 'Hide' : 'Show'} Advanced Options
            </Button>
          </Box>

          <Collapse in={showAdvanced}>
            <Paper variant="outlined" sx={{ mt: 2, p: 2, borderRadius: 2 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth variant="outlined" size="small">
                    <InputLabel>Search Method</InputLabel>
                    <Select
                      value={searchMethod}
                      onChange={(e) => setSearchMethod(e.target.value)}
                      label="Search Method"
                    >
                      <MenuItem value="similarity">Similarity Search</MenuItem>
                      <MenuItem value="mmr">Maximum Marginal Relevance</MenuItem>
                    </Select>
                  </FormControl>
                  
                  {searchMethod === 'mmr' && (
                    <Box sx={{ mt: 2 }}>
                      <Typography gutterBottom variant="body2">
                        Diversity Factor (λ = {mmrLambda})
                      </Typography>
                      <Slider
                        value={mmrLambda}
                        onChange={(_, value) => setMmrLambda(value as number)}
                        min={0}
                        max={1}
                        step={0.1}
                        valueLabelDisplay="auto"
                        marks={[
                          { value: 0, label: 'Diverse' },
                          { value: 1, label: 'Relevant' }
                        ]}
                      />
                      
                      <Typography gutterBottom variant="body2" sx={{ mt: 2 }}>
                        Fetch K: {mmrFetchK}
                      </Typography>
                      <Slider
                        value={mmrFetchK}
                        onChange={(_, value) => setMmrFetchK(value as number)}
                        min={resultCount}
                        max={50}
                        step={5}
                        valueLabelDisplay="auto"
                      />
                    </Box>
                  )}
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Typography gutterBottom variant="body2">
                    Number of Results: {resultCount}
                  </Typography>
                  <Slider
                    value={resultCount}
                    onChange={(_, value) => setResultCount(value as number)}
                    min={1}
                    max={20}
                    step={1}
                    valueLabelDisplay="auto"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={useTensorRT}
                        onChange={(e) => setUseTensorRT(e.target.checked)}
                        color="primary"
                      />
                    }
                    label={
                      <Typography variant="body2">
                        Use TensorRT Acceleration
                      </Typography>
                    }
                    sx={{ mt: 1 }}
                  />
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Typography gutterBottom variant="body2">
                    Metadata Filters
                  </Typography>
                  
                  <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                    <TextField
                      placeholder="Key"
                      size="small"
                      value={filterKey}
                      onChange={(e) => setFilterKey(e.target.value)}
                      sx={{ flex: 1 }}
                    />
                    <TextField
                      placeholder="Value"
                      size="small"
                      value={filterValue}
                      onChange={(e) => setFilterValue(e.target.value)}
                      sx={{ flex: 1 }}
                    />
                    <IconButton
                      color="primary"
                      onClick={handleAddFilter}
                      disabled={!filterKey || !filterValue}
                    >
                      <FilterIcon />
                    </IconButton>
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {Object.entries(filter).map(([key, value]) => (
                      <Chip
                        key={key}
                        label={`${key}: ${value}`}
                        onDelete={() => handleRemoveFilter(key)}
                        size="small"
                      />
                    ))}
                  </Box>
                  
                  {Object.keys(filter).length > 0 && (
                    <Button
                      startIcon={<ClearIcon />}
                      variant="text"
                      size="small"
                      onClick={() => setFilter({})}
                      sx={{ mt: 1 }}
                    >
                      Clear Filters
                    </Button>
                  )}
                </Grid>
              </Grid>
              
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                <Button
                  variant="outlined"
                  startIcon={<SettingsIcon />}
                  onClick={() => {}}
                  sx={{ mr: 1 }}
                >
                  Save as Default
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<ClearIcon />}
                  onClick={handleClearAll}
                >
                  Clear All
                </Button>
              </Box>
            </Paper>
          </Collapse>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {!loading && results.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            {results.length} Results
          </Typography>

          <Stack spacing={2}>
            {results.map((result, index) => (
              <Paper
                key={index}
                elevation={0}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  border: '1px solid rgba(0, 0, 0, 0.08)',
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
                    borderColor: 'rgba(0, 0, 0, 0.15)',
                  },
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Chip
                    label={`Score: ${(result.score * 100).toFixed(2)}%`}
                    color="primary"
                    size="small"
                    sx={{ borderRadius: 3 }}
                  />
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title="Copy text">
                      <IconButton
                        size="small"
                        onClick={() => navigator.clipboard.writeText(result.document.page_content)}
                      >
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    {result.document.metadata.source && (
                      <Tooltip title="Open source">
                        <IconButton size="small">
                          <LaunchIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Box>
                </Box>

                <Typography variant="body1" gutterBottom>
                  {result.document.page_content}
                </Typography>

                <Divider sx={{ my: 2 }} />

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(result.document.metadata).map(([key, value]) => (
                    <Chip
                      key={key}
                      label={`${key}: ${
                        key === 'date' || key === 'timestamp'
                          ? formatDate(value as string)
                          : typeof value === 'object'
                          ? JSON.stringify(value)
                          : value
                      }`}
                      size="small"
                      variant="outlined"
                      onClick={() => {
                        setFilter({ ...filter, [key]: value });
                        if (showAdvanced === false) setShowAdvanced(true);
                      }}
                      sx={{ borderRadius: 3 }}
                    />
                  ))}
                </Box>
              </Paper>
            ))}
          </Stack>
        </Box>
      )}

      {!loading && query && results.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 6 }}>
          <SearchIcon sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            No results found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try using different keywords or removing filters
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default Search;