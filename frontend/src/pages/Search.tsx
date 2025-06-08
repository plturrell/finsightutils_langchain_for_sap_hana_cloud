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
import ExperienceManager from '../components/ExperienceManager';

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
      <ExperienceManager 
        currentComponent="search"
        onOpenAdvanced={() => setShowAdvanced(true)}
      />
      
      {/* The original interface is kept but will be shown only when 
          simple mode is disabled through the ExperienceManager */}
    </Box>
  );
};

export default Search;