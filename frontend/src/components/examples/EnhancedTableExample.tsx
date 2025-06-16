import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  IconButton,
  Chip,
  Link,
  Tooltip
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  FilterList as FilterIcon,
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon
} from '@mui/icons-material';
import {
  EnhancedAnimatedTable,
  EnhancedTableSortHeader,
  EnhancedExpandableTableRow,
  EnhancedAnimatedTableRow,
  EnhancedAnimatedTableCell,
  EnhancedPaper,
  EnhancedBox,
  EnhancedTypography,
  EnhancedGradientTypography,
  EnhancedButton
} from '../enhanced';
import { soundEffects } from '../../utils/soundEffects';
import { useAnimationContext } from '../../context/AnimationContext';

/**
 * Sample data for the tables
 */
const FINANCIAL_DATA = [
  {
    id: '1',
    metric: 'Revenue',
    value: '$1,245,000',
    change: '+5.3%',
    status: 'positive',
    date: '2025-06-15',
    category: 'Income'
  },
  {
    id: '2',
    metric: 'Expenses',
    value: '$780,500',
    change: '-2.1%',
    status: 'negative',
    date: '2025-06-15',
    category: 'Expense'
  },
  {
    id: '3',
    metric: 'Profit',
    value: '$464,500',
    change: '+12.7%',
    status: 'positive',
    date: '2025-06-15',
    category: 'Income'
  },
  {
    id: '4',
    metric: 'Cash Flow',
    value: '$320,000',
    change: '+3.8%',
    status: 'positive',
    date: '2025-06-15',
    category: 'Flow'
  },
  {
    id: '5',
    metric: 'Debt',
    value: '$550,000',
    change: '-5.2%',
    status: 'negative',
    date: '2025-06-15',
    category: 'Liability'
  }
];

/**
 * Sample expanded content data
 */
const EXPANDED_DATA = {
  '1': {
    details: 'Revenue increased due to successful product launches and expanded market share.',
    breakdown: [
      { source: 'Product Sales', amount: '$950,000' },
      { source: 'Services', amount: '$295,000' }
    ],
    chart: 'Revenue Trend Chart'
  },
  '2': {
    details: 'Expenses decreased due to operational efficiencies and cost-cutting measures.',
    breakdown: [
      { source: 'Operations', amount: '$480,000' },
      { source: 'Marketing', amount: '$210,500' },
      { source: 'R&D', amount: '$90,000' }
    ],
    chart: 'Expense Breakdown Chart'
  },
  '3': {
    details: 'Profit increased significantly due to higher revenue and lower expenses.',
    breakdown: [
      { source: 'Gross Profit', amount: '$540,000' },
      { source: 'Taxes', amount: '-$75,500' }
    ],
    chart: 'Profit Margin Chart'
  },
  '4': {
    details: 'Cash flow improved with better collection practices and inventory management.',
    breakdown: [
      { source: 'Operating Activities', amount: '$280,000' },
      { source: 'Investing Activities', amount: '-$50,000' },
      { source: 'Financing Activities', amount: '$90,000' }
    ],
    chart: 'Cash Flow Statement'
  },
  '5': {
    details: 'Debt reduced through strategic repayment of high-interest loans.',
    breakdown: [
      { source: 'Long-term Debt', amount: '$400,000' },
      { source: 'Short-term Debt', amount: '$150,000' }
    ],
    chart: 'Debt Ratio Chart'
  }
};

/**
 * Example component to showcase enhanced table features
 */
export const EnhancedTableExample: React.FC = () => {
  const { animationsEnabled } = useAnimationContext();
  const [data, setData] = useState(FINANCIAL_DATA);
  const [animationsVisible, setAnimationsVisible] = useState(false);
  const [orderBy, setOrderBy] = useState('metric');
  const [order, setOrder] = useState<'asc' | 'desc'>('asc');
  
  // Show animations after a short delay
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, 300);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Sort data
  const handleRequestSort = (property: string) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
    
    // Sort the data
    const sortedData = [...data].sort((a, b) => {
      const aValue = a[property as keyof typeof a];
      const bValue = b[property as keyof typeof b];
      
      if (aValue < bValue) {
        return order === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return order === 'asc' ? 1 : -1;
      }
      return 0;
    });
    
    setData(sortedData);
  };
  
  // Render a cell with appropriate formatting
  const renderCell = (row: any, header: string, index: number) => {
    switch (header) {
      case 'Metric':
        return <Typography fontWeight={600}>{row.metric}</Typography>;
        
      case 'Value':
        return <Typography>{row.value}</Typography>;
        
      case 'Change':
        return (
          <Chip
            label={row.change}
            color={row.status === 'positive' ? 'success' : 'error'}
            size="small"
            variant="outlined"
            sx={{
              fontWeight: 600,
              borderWidth: 2,
              '& .MuiChip-label': {
                px: 1,
              }
            }}
          />
        );
        
      case 'Category':
        return (
          <Chip
            label={row.category}
            size="small"
            sx={{
              backgroundColor: getCategoryColor(row.category),
              color: '#fff',
              fontWeight: 600
            }}
          />
        );
        
      case 'Date':
        return <Typography>{formatDate(row.date)}</Typography>;
        
      case 'Actions':
        return (
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Edit">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  if (animationsEnabled) {
                    soundEffects.tap();
                  }
                }}
                sx={{
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    transform: 'scale(1.1)',
                    backgroundColor: 'rgba(0, 102, 179, 0.1)'
                  }
                }}
              >
                <EditIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Delete">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  if (animationsEnabled) {
                    soundEffects.tap();
                  }
                }}
                sx={{
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    transform: 'scale(1.1)',
                    backgroundColor: 'rgba(211, 47, 47, 0.1)',
                    color: '#d32f2f'
                  }
                }}
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        );
        
      default:
        return <Typography>{row[header.toLowerCase()]}</Typography>;
    }
  };
  
  // Helper function to get color for category
  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Income':
        return '#4caf50';
      case 'Expense':
        return '#f44336';
      case 'Flow':
        return '#2196f3';
      case 'Liability':
        return '#ff9800';
      default:
        return '#9e9e9e';
    }
  };
  
  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(date);
  };
  
  // Handle refresh
  const handleRefresh = () => {
    setAnimationsVisible(false);
    setTimeout(() => {
      setAnimationsVisible(true);
      // Here you would typically fetch fresh data
    }, 300);
    
    if (animationsEnabled) {
      soundEffects.tap();
    }
  };
  
  // Row click handler
  const handleRowClick = (row: any) => {
    console.log('Row clicked:', row);
    if (animationsEnabled) {
      soundEffects.tap();
    }
  };
  
  // Render expanded content
  const renderExpandedContent = (id: string) => {
    const expandedData = EXPANDED_DATA[id as keyof typeof EXPANDED_DATA];
    
    if (!expandedData) return null;
    
    return (
      <Box>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          Details
        </Typography>
        <Typography variant="body2" paragraph>
          {expandedData.details}
        </Typography>
        
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          Breakdown
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {expandedData.breakdown.map((item, index) => (
            <Box 
              key={index}
              sx={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                p: 1,
                borderRadius: 1,
                bgcolor: 'rgba(0, 102, 179, 0.05)',
                '&:hover': {
                  bgcolor: 'rgba(0, 102, 179, 0.1)',
                }
              }}
            >
              <Typography variant="body2">{item.source}</Typography>
              <Typography variant="body2" fontWeight={600}>{item.amount}</Typography>
            </Box>
          ))}
        </Box>
        
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            size="small"
            startIcon={<InfoIcon />}
            sx={{
              textTransform: 'none',
              transition: 'all 0.2s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
              }
            }}
          >
            View {expandedData.chart}
          </Button>
        </Box>
      </Box>
    );
  };
  
  return (
    <Box>
      {/* Basic Table Example */}
      <EnhancedGradientTypography variant="h5" sx={{ mb: 3 }}>
        Enhanced Table Examples
      </EnhancedGradientTypography>
      
      <EnhancedBox sx={{ mb: 5 }}>
        <EnhancedTypography variant="h6" gutterBottom>
          1. Basic Table with Batch Animations
        </EnhancedTypography>
        
        <EnhancedTypography variant="body2" color="text.secondary" paragraph>
          This table demonstrates basic animation features with batch animations for performance.
        </EnhancedTypography>
        
        <EnhancedAnimatedTable
          headers={['Metric', 'Value', 'Change', 'Category', 'Date']}
          data={data}
          getRowKey={(row) => row.id}
          renderCell={renderCell}
          onRowClick={handleRowClick}
          animationsVisible={animationsVisible}
          title="Financial Metrics"
          onRefresh={handleRefresh}
        />
      </EnhancedBox>
      
      {/* Sortable Table Example */}
      <EnhancedBox sx={{ mb: 5 }}>
        <EnhancedTypography variant="h6" gutterBottom>
          2. Sortable Table with Custom Headers
        </EnhancedTypography>
        
        <EnhancedTypography variant="body2" color="text.secondary" paragraph>
          This example shows a sortable table with animated sort headers.
        </EnhancedTypography>
        
        <EnhancedPaper sx={{ p: 0, borderRadius: 2, overflow: 'hidden' }}>
          <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(0, 102, 179, 0.1)' }}>
            <Typography variant="h6" fontWeight={600}>
              Sortable Financial Metrics
            </Typography>
            <Tooltip title="Refresh">
              <IconButton
                onClick={handleRefresh}
                size="small"
              >
                <RefreshIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Box sx={{ overflow: 'auto' }}>
            <Table>
              <TableHead>
                <TableRow>
                  <EnhancedTableSortHeader
                    id="metric"
                    label="Metric"
                    orderBy={orderBy}
                    order={order}
                    onRequestSort={handleRequestSort}
                    animationsVisible={animationsVisible}
                  />
                  <EnhancedTableSortHeader
                    id="value"
                    label="Value"
                    orderBy={orderBy}
                    order={order}
                    onRequestSort={handleRequestSort}
                    animationsVisible={animationsVisible}
                  />
                  <EnhancedTableSortHeader
                    id="change"
                    label="Change"
                    orderBy={orderBy}
                    order={order}
                    onRequestSort={handleRequestSort}
                    animationsVisible={animationsVisible}
                  />
                  <EnhancedTableSortHeader
                    id="category"
                    label="Category"
                    orderBy={orderBy}
                    order={order}
                    onRequestSort={handleRequestSort}
                    animationsVisible={animationsVisible}
                  />
                  <EnhancedTableSortHeader
                    id="date"
                    label="Date"
                    orderBy={orderBy}
                    order={order}
                    onRequestSort={handleRequestSort}
                    animationsVisible={animationsVisible}
                  />
                  <EnhancedAnimatedTableCell>
                    <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                      <Tooltip title="Filter list">
                        <IconButton size="small">
                          <FilterIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </EnhancedAnimatedTableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((row, index) => (
                  <EnhancedAnimatedTableRow
                    key={row.id}
                    animationsVisible={animationsVisible}
                    animationDelay={200}
                    index={index}
                    onClick={() => handleRowClick(row)}
                  >
                    <EnhancedAnimatedTableCell>{row.metric}</EnhancedAnimatedTableCell>
                    <EnhancedAnimatedTableCell>{row.value}</EnhancedAnimatedTableCell>
                    <EnhancedAnimatedTableCell>
                      <Chip
                        label={row.change}
                        color={row.status === 'positive' ? 'success' : 'error'}
                        size="small"
                        variant="outlined"
                        sx={{
                          fontWeight: 600,
                          borderWidth: 2,
                        }}
                      />
                    </EnhancedAnimatedTableCell>
                    <EnhancedAnimatedTableCell>
                      <Chip
                        label={row.category}
                        size="small"
                        sx={{
                          backgroundColor: getCategoryColor(row.category),
                          color: '#fff',
                          fontWeight: 600
                        }}
                      />
                    </EnhancedAnimatedTableCell>
                    <EnhancedAnimatedTableCell>{formatDate(row.date)}</EnhancedAnimatedTableCell>
                    <EnhancedAnimatedTableCell>
                      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
                        <IconButton size="small" onClick={(e) => e.stopPropagation()}>
                          <EditIcon fontSize="small" />
                        </IconButton>
                        <IconButton size="small" onClick={(e) => e.stopPropagation()}>
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </EnhancedAnimatedTableCell>
                  </EnhancedAnimatedTableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </EnhancedPaper>
      </EnhancedBox>
      
      {/* Expandable Rows Table Example */}
      <EnhancedBox sx={{ mb: 5 }}>
        <EnhancedTypography variant="h6" gutterBottom>
          3. Expandable Rows Table
        </EnhancedTypography>
        
        <EnhancedTypography variant="body2" color="text.secondary" paragraph>
          This example demonstrates tables with expandable rows for additional details.
        </EnhancedTypography>
        
        <EnhancedPaper sx={{ p: 0, borderRadius: 2, overflow: 'hidden' }}>
          <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(0, 102, 179, 0.1)' }}>
            <Typography variant="h6" fontWeight={600}>
              Financial Metrics with Details
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Click a row to expand
            </Typography>
          </Box>
          
          <Box sx={{ overflow: 'auto' }}>
            <Table>
              <TableHead>
                <TableRow>
                  <EnhancedAnimatedTableCell>Metric</EnhancedAnimatedTableCell>
                  <EnhancedAnimatedTableCell>Value</EnhancedAnimatedTableCell>
                  <EnhancedAnimatedTableCell>Change</EnhancedAnimatedTableCell>
                  <EnhancedAnimatedTableCell>Category</EnhancedAnimatedTableCell>
                  <EnhancedAnimatedTableCell padding="checkbox"></EnhancedAnimatedTableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((row, index) => (
                  <EnhancedExpandableTableRow
                    key={row.id}
                    animationsVisible={animationsVisible}
                    animationDelay={200}
                    index={index}
                    row={
                      <>
                        <EnhancedAnimatedTableCell>{row.metric}</EnhancedAnimatedTableCell>
                        <EnhancedAnimatedTableCell>{row.value}</EnhancedAnimatedTableCell>
                        <EnhancedAnimatedTableCell>
                          <Chip
                            label={row.change}
                            color={row.status === 'positive' ? 'success' : 'error'}
                            size="small"
                            variant="outlined"
                            sx={{
                              fontWeight: 600,
                              borderWidth: 2,
                            }}
                          />
                        </EnhancedAnimatedTableCell>
                        <EnhancedAnimatedTableCell>
                          <Chip
                            label={row.category}
                            size="small"
                            sx={{
                              backgroundColor: getCategoryColor(row.category),
                              color: '#fff',
                              fontWeight: 600
                            }}
                          />
                        </EnhancedAnimatedTableCell>
                      </>
                    }
                    expandedContent={renderExpandedContent(row.id)}
                  />
                ))}
              </TableBody>
            </Table>
          </Box>
        </EnhancedPaper>
      </EnhancedBox>
      
      <EnhancedButton
        variant="outlined"
        onClick={handleRefresh}
        sx={{ mb: 3 }}
      >
        Reset Animations
      </EnhancedButton>
      
      <EnhancedTypography variant="body2" color="text.secondary">
        Tables use batch animations for optimal performance with Apple-inspired physics.
      </EnhancedTypography>
    </Box>
  );
};

export default EnhancedTableExample;