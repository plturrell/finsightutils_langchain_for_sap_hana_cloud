import React, { useState, useEffect, useRef } from 'react';
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
  Container,
  InputAdornment,
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
  History as HistoryIcon,
} from '@mui/icons-material';
import { useSpring, useTrail, useChain, animated, useSpringRef, config } from 'react-spring';
import { vectorStoreService, SearchResult, SearchResponse } from '../api/services';
import ExperienceManager from '../components/ExperienceManager';
import { EnhancedSearchResults } from '../components/enhanced/EnhancedSearchResults';

// Animated components
const AnimatedBox = animated(Box);
const AnimatedCard = animated(Card);
const AnimatedPaper = animated(Paper);
const AnimatedTypography = animated(Typography);
const AnimatedGrid = animated(Grid);
const AnimatedContainer = animated(Container);
const AnimatedButton = animated(Button);
const AnimatedAlert = animated(Alert);
const AnimatedDivider = animated(Divider);
const AnimatedTextField = animated(TextField);

interface SearchResult {
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
  const [showSearchHistory, setShowSearchHistory] = useState(false);
  const [animationsVisible, setAnimationsVisible] = useState<boolean>(false);
  
  // Animation spring refs for sequence chaining
  const headerSpringRef = useSpringRef();
  const searchBoxSpringRef = useSpringRef();
  const advancedOptionsSpringRef = useSpringRef();
  const resultsSpringRef = useSpringRef();
  
  // Animation springs
  const headerAnimation = useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const searchBoxAnimation = useSpring({
    ref: searchBoxSpringRef,
    from: { opacity: 0, transform: 'translateY(-10px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const advancedOptionsAnimation = useSpring({
    ref: advancedOptionsSpringRef,
    from: { opacity: 0, height: 0 },
    to: { 
      opacity: animationsVisible && showAdvanced ? 1 : 0, 
      height: animationsVisible && showAdvanced ? 'auto' : 0 
    },
    config: { tension: 280, friction: 60 }
  });
  
  const resultsAnimation = useSpring({
    ref: resultsSpringRef,
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Result cards animation trail
  const resultTrail = useTrail(results.length, {
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
    config: { mass: 1, tension: 280, friction: 60 }
  });
  
  // Chain animations sequence
  useChain(
    animationsVisible 
      ? [headerSpringRef, searchBoxSpringRef, advancedOptionsSpringRef, resultsSpringRef] 
      : [resultsSpringRef, advancedOptionsSpringRef, searchBoxSpringRef, headerSpringRef],
    animationsVisible 
      ? [0, 0.1, 0.2, 0.3] 
      : [0, 0.1, 0.2, 0.3]
  );
  
  // Trigger animations on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);

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
    <AnimatedBox>
      <ExperienceManager 
        currentComponent="search"
        onOpenAdvanced={() => setShowAdvanced(true)}
      />
      
      {/* Header Section */}
      <AnimatedBox sx={{ mb: 4 }} style={headerAnimation}>
        <AnimatedTypography 
          variant="h4" 
          fontWeight="600"
          sx={{
            background: `linear-gradient(90deg, #0066B3, #2a8fd8)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            letterSpacing: '0.02em',
          }}
        >
          Semantic Search
        </AnimatedTypography>
        <AnimatedTypography 
          variant="body1" 
          color="text.secondary"
          sx={{
            maxWidth: '800px',
            lineHeight: 1.6,
          }}
        >
          Search your vector database using natural language and find semantically similar content
        </AnimatedTypography>
      </AnimatedBox>
      
      {/* Search Box */}
      <AnimatedCard 
        style={searchBoxAnimation}
        sx={{ 
          mb: 3,
          borderRadius: '12px',
          boxShadow: '0 6px 20px rgba(0,0,0,0.05)',
          transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
          border: '1px solid rgba(0, 102, 179, 0.1)',
          '&:hover': {
            boxShadow: '0 12px 28px rgba(0,0,0,0.1), 0 8px 10px rgba(0,0,0,0.08)',
          }
        }}
      >
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
            <AnimatedTextField
              fullWidth
              placeholder="Search for anything..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              variant="outlined"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon 
                      color="primary" 
                      sx={{ 
                        animation: loading ? 'pulse 1.5s infinite' : 'none',
                        '@keyframes pulse': {
                          '0%': { opacity: 0.6 },
                          '50%': { opacity: 1 },
                          '100%': { opacity: 0.6 }
                        }
                      }} 
                    />
                  </InputAdornment>
                ),
                endAdornment: query ? (
                  <InputAdornment position="end">
                    <IconButton onClick={() => setQuery('')} size="small">
                      <ClearIcon fontSize="small" />
                    </IconButton>
                  </InputAdornment>
                ) : null,
                sx: {
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(0, 102, 179, 0.2)',
                    transition: 'all 0.3s ease',
                  },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(0, 102, 179, 0.5)',
                  },
                  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                    borderColor: '#0066B3',
                    borderWidth: 2,
                    boxShadow: '0 0 0 3px rgba(0, 102, 179, 0.1)',
                  }
                }
              }}
            />
            
            <Box sx={{ display: 'flex', ml: 1 }}>
              <AnimatedButton
                variant="contained"
                color="primary"
                onClick={handleSearch}
                disabled={loading || !query.trim()}
                sx={{
                  ml: 1,
                  position: 'relative',
                  overflow: 'hidden',
                  background: loading ? '#0066B3' : 'linear-gradient(90deg, #0066B3, #2a8fd8)',
                  boxShadow: '0 4px 12px rgba(0, 102, 179, 0.2)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: '0 6px 16px rgba(0, 102, 179, 0.3)',
                  },
                  '&:after': {
                    content: '""',
                    position: 'absolute',
                    width: '100%',
                    height: '100%',
                    top: 0,
                    left: 0,
                    pointerEvents: 'none',
                    background: 'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 60%)',
                    transform: 'scale(0, 0)',
                    opacity: 0,
                    transition: 'transform 0.4s, opacity 0.3s',
                  },
                  '&:active:after': {
                    transform: 'scale(2, 2)',
                    opacity: 0,
                    transition: '0s',
                  },
                }}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Search'}
              </AnimatedButton>
              
              <Tooltip title="Advanced Options">
                <IconButton 
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  color="primary"
                  sx={{
                    ml: 1,
                    transition: 'all 0.3s ease',
                    transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)',
                    '&:hover': {
                      backgroundColor: 'rgba(0, 102, 179, 0.1)',
                    }
                  }}
                >
                  {showAdvanced ? <ExpandLess /> : <ExpandMore />}
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Search History">
                <IconButton 
                  onClick={() => setShowSearchHistory(!showSearchHistory)}
                  color="primary"
                  sx={{
                    ml: 1,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      backgroundColor: 'rgba(0, 102, 179, 0.1)',
                    }
                  }}
                >
                  <HistoryIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
          
          {/* Search History Dropdown */}
          <Collapse in={showSearchHistory && searchHistory.length > 0}>
            <AnimatedPaper
              variant="outlined"
              sx={{ 
                p: 2, 
                mb: 2,
                borderRadius: '8px',
                borderColor: 'rgba(0, 102, 179, 0.2)',
              }}
            >
              <Typography variant="subtitle2" gutterBottom>
                Recent Searches
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {searchHistory.map((item, index) => (
                  <Chip
                    key={index}
                    label={item}
                    onClick={() => setQuery(item)}
                    sx={{ 
                      m: 0.5, 
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        backgroundColor: 'rgba(0, 102, 179, 0.1)',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 2px 8px rgba(0, 102, 179, 0.1)',
                      }
                    }}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Stack>
            </AnimatedPaper>
          </Collapse>
          
          {/* Advanced Options */}
          <animated.div style={advancedOptionsAnimation}>
            <Collapse in={showAdvanced}>
              <AnimatedPaper
                variant="outlined"
                sx={{ 
                  p: 2, 
                  mb: 2,
                  borderRadius: '8px',
                  borderColor: 'rgba(0, 102, 179, 0.2)',
                }}
              >
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Search Method
                    </Typography>
                    <FormControl fullWidth variant="outlined" size="small">
                      <Select
                        value={searchMethod}
                        onChange={(e) => setSearchMethod(e.target.value as string)}
                        sx={{ 
                          borderRadius: '8px',
                          '& .MuiOutlinedInput-notchedOutline': {
                            borderColor: 'rgba(0, 102, 179, 0.2)',
                          },
                          '&:hover .MuiOutlinedInput-notchedOutline': {
                            borderColor: 'rgba(0, 102, 179, 0.5)',
                          },
                        }}
                      >
                        <MenuItem value="similarity">Similarity Search</MenuItem>
                        <MenuItem value="mmr">Maximum Marginal Relevance (MMR)</MenuItem>
                      </Select>
                    </FormControl>
                    
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Number of Results: {resultCount}
                      </Typography>
                      <Slider
                        value={resultCount}
                        onChange={(_, value) => setResultCount(value as number)}
                        min={1}
                        max={20}
                        step={1}
                        valueLabelDisplay="auto"
                        sx={{
                          color: '#0066B3',
                          '& .MuiSlider-thumb': {
                            '&:hover, &.Mui-focusVisible': {
                              boxShadow: '0 0 0 8px rgba(0, 102, 179, 0.1)',
                            },
                          },
                        }}
                      />
                    </Box>
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={useTensorRT}
                          onChange={(e) => setUseTensorRT(e.target.checked)}
                          color="primary"
                        />
                      }
                      label="Use TensorRT Acceleration"
                      sx={{ mt: 2 }}
                    />
                  </Grid>
                  
                  {searchMethod === 'mmr' && (
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" gutterBottom>
                        Diversity (MMR Lambda): {mmrLambda.toFixed(2)}
                      </Typography>
                      <Slider
                        value={mmrLambda}
                        onChange={(_, value) => setMmrLambda(value as number)}
                        min={0}
                        max={1}
                        step={0.01}
                        valueLabelDisplay="auto"
                        sx={{
                          color: '#0066B3',
                          '& .MuiSlider-thumb': {
                            '&:hover, &.Mui-focusVisible': {
                              boxShadow: '0 0 0 8px rgba(0, 102, 179, 0.1)',
                            },
                          },
                        }}
                      />
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Lower values increase diversity, higher values increase relevance.
                      </Typography>
                      
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          MMR Fetch K: {mmrFetchK}
                        </Typography>
                        <Slider
                          value={mmrFetchK}
                          onChange={(_, value) => setMmrFetchK(value as number)}
                          min={5}
                          max={100}
                          step={1}
                          valueLabelDisplay="auto"
                          sx={{
                            color: '#0066B3',
                            '& .MuiSlider-thumb': {
                              '&:hover, &.Mui-focusVisible': {
                                boxShadow: '0 0 0 8px rgba(0, 102, 179, 0.1)',
                              },
                            },
                          }}
                        />
                      </Box>
                    </Grid>
                  )}
                  
                  {searchMethod === 'similarity' && (
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" gutterBottom>
                        Filter Results
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <TextField
                          label="Key"
                          value={filterKey}
                          onChange={(e) => setFilterKey(e.target.value)}
                          variant="outlined"
                          size="small"
                          sx={{ mr: 1, flex: 1 }}
                        />
                        <TextField
                          label="Value"
                          value={filterValue}
                          onChange={(e) => setFilterValue(e.target.value)}
                          variant="outlined"
                          size="small"
                          sx={{ mr: 1, flex: 1 }}
                        />
                        <Button
                          onClick={handleAddFilter}
                          variant="outlined"
                          color="primary"
                          disabled={!filterKey || !filterValue}
                          sx={{ 
                            transition: 'all 0.3s ease',
                            '&:hover': {
                              transform: 'translateY(-2px)',
                              boxShadow: '0 2px 8px rgba(0, 102, 179, 0.1)',
                            }
                          }}
                        >
                          Add
                        </Button>
                      </Box>
                      
                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        {Object.entries(filter).map(([key, value], index) => (
                          <Chip
                            key={index}
                            label={`${key}: ${value}`}
                            onDelete={() => handleRemoveFilter(key)}
                            sx={{ 
                              m: 0.5, 
                              transition: 'all 0.3s ease',
                              '&:hover': {
                                transform: 'translateY(-2px)',
                                boxShadow: '0 2px 8px rgba(0, 102, 179, 0.1)',
                              }
                            }}
                          />
                        ))}
                      </Stack>
                    </Grid>
                  )}
                </Grid>
                
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={handleClearAll}
                    startIcon={<ClearIcon />}
                    sx={{ 
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: '0 2px 8px rgba(0, 102, 179, 0.1)',
                      }
                    }}
                  >
                    Clear All
                  </Button>
                </Box>
              </AnimatedPaper>
            </Collapse>
          </animated.div>
        </CardContent>
      </AnimatedCard>
      
      {/* Error Alert */}
      {error && (
        <AnimatedAlert 
          severity="error" 
          sx={{ 
            mb: 3,
            borderRadius: '8px',
            position: 'relative',
            overflow: 'hidden',
            '&::after': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '4px',
              background: 'linear-gradient(90deg, #f44336, #ff9800)',
              animation: 'shimmer 2s infinite linear',
            },
            '@keyframes shimmer': {
              '0%': {
                backgroundPosition: '-100% 0',
              },
              '100%': {
                backgroundPosition: '100% 0',
              },
            },
          }}
          style={resultsAnimation}
        >
          {error}
        </AnimatedAlert>
      )}
      
      {/* Results */}
      {results.length > 0 && (
        <AnimatedBox style={resultsAnimation}>
          <EnhancedSearchResults 
            results={results}
            animationsVisible={animationsVisible}
            baseDelay={300}
            resultsLabel="Search Results"
            animateHeader={true}
            enableHover={true}
            onCopy={(result) => {
              // Provide haptic feedback when copying
              if ('navigator' in window && 'vibrate' in navigator) {
                navigator.vibrate(5); // Subtle vibration for 5ms
              }
            }}
          />
        </AnimatedBox>
      )}
      
      {/* Loading indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress 
            size={60} 
            sx={{
              color: '#0066B3',
              filter: 'drop-shadow(0 0 4px rgba(0, 102, 179, 0.3))'
            }}
          />
        </Box>
      )}
    </AnimatedBox>
  );
};

export default Search;