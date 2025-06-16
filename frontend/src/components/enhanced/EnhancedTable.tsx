import React, { useState } from 'react';
import {
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableSortLabel,
  Paper,
  Typography,
  Box,
  Tooltip,
  IconButton,
  useTheme,
  alpha
} from '@mui/material';
import {
  KeyboardArrowDown as ExpandIcon,
  KeyboardArrowUp as CollapseIcon,
  Sync as RefreshIcon
} from '@mui/icons-material';
import { animated, useSpring } from '@react-spring/web';
import { useAnimationContext } from '../../context/AnimationContext';
import { useBatchAnimations } from '../../hooks/useAnimations';
import { soundEffects } from '../../utils/soundEffects';

// Animated versions of MUI table components
const AnimatedTableContainer = animated(TableContainer);
const AnimatedTable = animated(Table);
const AnimatedTableHead = animated(TableHead);
const AnimatedTableBody = animated(TableBody);
const AnimatedTableRow = animated(TableRow);
const AnimatedTableCell = animated(TableCell);
const AnimatedPaper = animated(Paper);

/**
 * Props for the EnhancedTableRow component
 */
export interface EnhancedTableRowProps {
  /** Row children */
  children: React.ReactNode;
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Animation delay in milliseconds */
  animationDelay?: number;
  /** Row index for staggered animations */
  index?: number;
  /** Whether to enable hover effect */
  enableHover?: boolean;
  /** Function called when the row is clicked */
  onClick?: () => void;
  /** Additional styles to apply */
  sx?: any;
}

/**
 * Enhanced Table Row with Apple-inspired animations
 * Uses batch animations for improved performance
 */
export const EnhancedTableRow: React.FC<EnhancedTableRowProps> = ({
  children,
  animationsVisible = true,
  animationDelay = 0,
  index = 0,
  enableHover = true,
  onClick,
  sx = {}
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Calculate staggered delay based on index
  const totalDelay = animationDelay + (index * 20); // Shorter delay for tables
  
  // Use batch animations for row elements
  const animations = useBatchAnimations(
    animationsVisible,
    [
      {
        key: 'row',
        from: { opacity: 0, transform: 'translateX(-10px)' },
        to: { 
          opacity: animationsEnabled && animationsVisible ? 1 : 0, 
          transform: animationsEnabled && animationsVisible ? 'translateX(0)' : 'translateX(-10px)'
        },
        delay: totalDelay,
        config: { tension: 350, friction: 26, mass: 1 }
      },
      {
        key: 'hover',
        to: enableHover && { 
          backgroundColor: isHovered && animationsEnabled 
            ? alpha(theme.palette.primary.main, 0.07)
            : 'transparent'
        },
        config: { tension: 350, friction: 20 } // Apple-like immediate response
      }
    ]
  );
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
    if (animationsEnabled && enableHover) {
      soundEffects.hover();
    }
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  // Handle click
  const handleClick = () => {
    if (onClick) {
      onClick();
      if (animationsEnabled) {
        soundEffects.tap();
      }
    }
  };
  
  return (
    <AnimatedTableRow
      style={{
        ...animations.row,
        ...animations.hover
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      sx={{
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        willChange: 'background-color, opacity, transform',
        '&:hover': {
          backgroundColor: enableHover ? alpha(theme.palette.primary.main, 0.07) : undefined
        },
        ...sx
      }}
    >
      {children}
    </AnimatedTableRow>
  );
};

/**
 * Props for the EnhancedTableCell component
 */
export interface EnhancedTableCellProps {
  /** Cell children */
  children: React.ReactNode;
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Animation delay in milliseconds */
  animationDelay?: number;
  /** Additional styles to apply */
  sx?: any;
}

/**
 * Enhanced Table Cell with subtle animations
 */
export const EnhancedTableCell: React.FC<EnhancedTableCellProps> = ({
  children,
  animationsVisible = true,
  animationDelay = 0,
  sx = {}
}) => {
  const { animationsEnabled } = useAnimationContext();
  
  // Cell animation
  const animation = useSpring({
    from: { opacity: 0 },
    to: { opacity: animationsEnabled && animationsVisible ? 1 : 0 },
    delay: animationDelay,
    config: { tension: 350, friction: 26 }
  });
  
  return (
    <AnimatedTableCell
      style={animation}
      sx={{
        transition: 'all 0.2s ease',
        ...sx
      }}
    >
      {children}
    </AnimatedTableCell>
  );
};

/**
 * Props for the EnhancedTableHeader component
 */
export interface EnhancedTableHeaderProps {
  /** Header children */
  children: React.ReactNode;
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Additional styles to apply */
  sx?: any;
}

/**
 * Enhanced Table Header with animations
 */
export const EnhancedTableHeader: React.FC<EnhancedTableHeaderProps> = ({
  children,
  animationsVisible = true,
  sx = {}
}) => {
  const { animationsEnabled } = useAnimationContext();
  
  // Header animation
  const animation = useSpring({
    from: { opacity: 0, transform: 'translateY(-10px)' },
    to: { 
      opacity: animationsEnabled && animationsVisible ? 1 : 0,
      transform: animationsEnabled && animationsVisible ? 'translateY(0)' : 'translateY(-10px)'
    },
    config: { tension: 350, friction: 26 }
  });
  
  return (
    <AnimatedTableHead
      style={animation}
      sx={{
        borderBottom: '2px solid rgba(0, 102, 179, 0.2)',
        background: 'linear-gradient(180deg, rgba(0,102,179,0.03) 0%, rgba(0,102,179,0) 100%)',
        ...sx
      }}
    >
      {children}
    </AnimatedTableHead>
  );
};

/**
 * Props for the EnhancedTableComponent
 */
export interface EnhancedTableProps {
  /** Table headers */
  headers: string[];
  /** Data to display */
  data: any[];
  /** Function to render a cell */
  renderCell: (row: any, header: string, index: number) => React.ReactNode;
  /** Function to get a unique key for each row */
  getRowKey: (row: any) => string;
  /** Function called when a row is clicked */
  onRowClick?: (row: any) => void;
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Base animation delay */
  baseDelay?: number;
  /** Whether to enable row hover effects */
  enableHover?: boolean;
  /** Optional table title */
  title?: string;
  /** Optional refresh callback */
  onRefresh?: () => void;
  /** Optional custom empty state */
  emptyState?: React.ReactNode;
  /** Optional table caption */
  caption?: string;
}

/**
 * Enhanced Table Component with Apple-inspired animations
 * Uses batch animations for improved performance
 */
export const EnhancedTableComponent: React.FC<EnhancedTableProps> = ({
  headers,
  data,
  renderCell,
  getRowKey,
  onRowClick,
  animationsVisible = true,
  baseDelay = 0,
  enableHover = true,
  title,
  onRefresh,
  emptyState,
  caption
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  
  // Container animation
  const containerAnimation = useSpring({
    from: { opacity: 0, transform: 'scale(0.98)' },
    to: { 
      opacity: animationsEnabled && animationsVisible ? 1 : 0,
      transform: animationsEnabled && animationsVisible ? 'scale(1)' : 'scale(0.98)'
    },
    delay: baseDelay,
    config: { tension: 300, friction: 26 }
  });
  
  // Title animation
  const titleAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(-10px)' },
    to: { 
      opacity: animationsEnabled && animationsVisible ? 1 : 0,
      transform: animationsEnabled && animationsVisible ? 'translateY(0)' : 'translateY(-10px)'
    },
    delay: baseDelay + 100,
    config: { tension: 350, friction: 26 }
  });
  
  // Handle refresh
  const handleRefresh = () => {
    if (onRefresh) {
      if (animationsEnabled) {
        soundEffects.tap();
      }
      onRefresh();
    }
  };
  
  return (
    <Box>
      {/* Table title and actions */}
      {title && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <animated.div style={titleAnimation}>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 600,
                background: `linear-gradient(90deg, #0066B3, #2a8fd8)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textFillColor: 'transparent',
              }}
            >
              {title}
            </Typography>
          </animated.div>
          
          {onRefresh && (
            <animated.div style={titleAnimation}>
              <Tooltip title="Refresh">
                <IconButton
                  onClick={handleRefresh}
                  size="small"
                  sx={{
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'rotate(30deg)',
                      backgroundColor: alpha(theme.palette.primary.main, 0.1)
                    }
                  }}
                >
                  <RefreshIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </animated.div>
          )}
        </Box>
      )}
      
      {/* Main table */}
      <AnimatedTableContainer 
        style={containerAnimation}
        component={Paper}
        sx={{
          borderRadius: 2,
          overflow: 'hidden',
          boxShadow: '0 6px 20px rgba(0,0,0,0.05)',
          border: '1px solid rgba(0, 102, 179, 0.1)',
          willChange: 'opacity, transform',
        }}
      >
        {data.length > 0 ? (
          <>
            <AnimatedTable>
              {caption && <caption style={{ padding: '8px' }}>{caption}</caption>}
              
              <EnhancedTableHeader animationsVisible={animationsVisible}>
                <TableRow>
                  {headers.map((header) => (
                    <TableCell key={header}>
                      <Typography 
                        variant="subtitle2"
                        sx={{ 
                          fontWeight: 600,
                          color: '#0066B3',
                        }}
                      >
                        {header}
                      </Typography>
                    </TableCell>
                  ))}
                </TableRow>
              </EnhancedTableHeader>
              
              <AnimatedTableBody>
                {data.map((row, index) => (
                  <EnhancedTableRow
                    key={getRowKey(row)}
                    animationsVisible={animationsVisible}
                    animationDelay={baseDelay + 200}
                    index={index}
                    enableHover={enableHover}
                    onClick={onRowClick ? () => onRowClick(row) : undefined}
                  >
                    {headers.map((header, cellIndex) => (
                      <EnhancedTableCell
                        key={`${getRowKey(row)}-${cellIndex}`}
                        animationsVisible={animationsVisible}
                        animationDelay={baseDelay + 300 + (index * 20) + (cellIndex * 10)}
                      >
                        {renderCell(row, header, cellIndex)}
                      </EnhancedTableCell>
                    ))}
                  </EnhancedTableRow>
                ))}
              </AnimatedTableBody>
            </AnimatedTable>
          </>
        ) : (
          emptyState || (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="body1" color="text.secondary">
                No data to display
              </Typography>
            </Box>
          )
        )}
      </AnimatedTableContainer>
    </Box>
  );
};

/**
 * Props for the EnhancedTableSortHeader component
 */
export interface EnhancedTableSortHeaderProps {
  /** Column ID */
  id: string;
  /** Column label */
  label: string;
  /** Current sort field */
  orderBy: string;
  /** Current sort direction */
  order: 'asc' | 'desc';
  /** Function called when sort changes */
  onRequestSort: (property: string) => void;
  /** Whether animations are visible */
  animationsVisible?: boolean;
}

/**
 * Enhanced Table Sort Header with animations
 */
export const EnhancedTableSortHeader: React.FC<EnhancedTableSortHeaderProps> = ({
  id,
  label,
  orderBy,
  order,
  onRequestSort,
  animationsVisible = true
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  
  // Animation
  const animation = useSpring({
    from: { opacity: 0, transform: 'translateY(-5px)' },
    to: { 
      opacity: animationsEnabled && animationsVisible ? 1 : 0,
      transform: animationsEnabled && animationsVisible ? 'translateY(0)' : 'translateY(-5px)'
    },
    config: { tension: 350, friction: 26 }
  });
  
  const handleClick = () => {
    onRequestSort(id);
    if (animationsEnabled) {
      soundEffects.tap();
    }
  };
  
  return (
    <AnimatedTableCell 
      style={animation}
      sortDirection={orderBy === id ? order : false}
      sx={{
        transition: 'background-color 0.2s ease',
        '&:hover': {
          backgroundColor: alpha(theme.palette.primary.main, 0.04)
        }
      }}
    >
      <TableSortLabel
        active={orderBy === id}
        direction={orderBy === id ? order : 'asc'}
        onClick={handleClick}
        sx={{
          fontWeight: 600,
          color: '#0066B3 !important',
          '&.MuiTableSortLabel-active': {
            color: '#0066B3 !important',
          },
          '&:hover': {
            color: '#2a8fd8 !important',
          },
          '& .MuiTableSortLabel-icon': {
            transition: 'transform 0.2s ease',
          }
        }}
      >
        {label}
      </TableSortLabel>
    </AnimatedTableCell>
  );
};

/**
 * Props for the ExpandableTableRow component
 */
export interface ExpandableTableRowProps {
  /** Main row content */
  row: React.ReactNode;
  /** Expanded content */
  expandedContent: React.ReactNode;
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Animation delay in milliseconds */
  animationDelay?: number;
  /** Row index for staggered animations */
  index?: number;
}

/**
 * Expandable Table Row with animations
 */
export const ExpandableTableRow: React.FC<ExpandableTableRowProps> = ({
  row,
  expandedContent,
  animationsVisible = true,
  animationDelay = 0,
  index = 0
}) => {
  const [expanded, setExpanded] = useState(false);
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  
  // Calculate staggered delay based on index
  const totalDelay = animationDelay + (index * 20);
  
  // Row animation
  const rowAnimation = useSpring({
    from: { opacity: 0, transform: 'translateX(-10px)' },
    to: { 
      opacity: animationsEnabled && animationsVisible ? 1 : 0, 
      transform: animationsEnabled && animationsVisible ? 'translateX(0)' : 'translateX(-10px)'
    },
    delay: totalDelay,
    config: { tension: 350, friction: 26 }
  });
  
  // Expanded content animation
  const expandedAnimation = useSpring({
    from: { opacity: 0, height: 0, transform: 'translateY(-20px)' },
    to: { 
      opacity: expanded && animationsEnabled ? 1 : 0,
      height: expanded && animationsEnabled ? 'auto' : 0,
      transform: expanded && animationsEnabled ? 'translateY(0)' : 'translateY(-20px)'
    },
    config: { tension: 350, friction: 26 }
  });
  
  // Toggle expanded state
  const toggleExpanded = () => {
    setExpanded(!expanded);
    if (animationsEnabled) {
      soundEffects.switch();
    }
  };
  
  return (
    <>
      <AnimatedTableRow
        style={rowAnimation}
        onClick={toggleExpanded}
        sx={{
          cursor: 'pointer',
          backgroundColor: expanded ? alpha(theme.palette.primary.main, 0.04) : 'transparent',
          transition: 'background-color 0.2s ease',
          '&:hover': {
            backgroundColor: expanded 
              ? alpha(theme.palette.primary.main, 0.07)
              : alpha(theme.palette.primary.main, 0.04)
          }
        }}
      >
        {row}
        <TableCell padding="checkbox">
          <IconButton
            size="small"
            sx={{ 
              transition: 'transform 0.3s ease',
              transform: expanded ? 'rotate(-180deg)' : 'rotate(0deg)'
            }}
          >
            <ExpandIcon />
          </IconButton>
        </TableCell>
      </AnimatedTableRow>
      
      <TableRow>
        <TableCell 
          colSpan={100}
          padding="0"
          sx={{ 
            border: expanded ? undefined : 0,
            overflow: 'hidden'
          }}
        >
          <animated.div 
            style={expandedAnimation}
            className="expanded-content"
          >
            <Box sx={{ p: 2, backgroundColor: alpha(theme.palette.primary.main, 0.02) }}>
              {expandedContent}
            </Box>
          </animated.div>
        </TableCell>
      </TableRow>
    </>
  );
};

export default {
  TableRow: EnhancedTableRow,
  TableCell: EnhancedTableCell,
  TableHeader: EnhancedTableHeader,
  Table: EnhancedTableComponent,
  SortHeader: EnhancedTableSortHeader,
  ExpandableRow: ExpandableTableRow
};