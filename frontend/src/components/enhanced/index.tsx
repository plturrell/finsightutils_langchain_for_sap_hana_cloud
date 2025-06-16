import React from 'react';
import {
  Button,
  IconButton,
  TextField,
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemButton,
  TableContainer,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Grid,
  Divider,
  CircularProgress,
  Switch,
  Checkbox,
  Radio,
  Slider,
  Alert,
  Tabs,
  Tab,
  Badge,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  FormControlLabel,
  Autocomplete
} from '@mui/material';
import { animated } from '@react-spring/web';
import { 
  withFadeUp, 
  withFadeDown, 
  withHoverEffect, 
  withSoundFeedback,
  withGradientText,
  withCompleteAnimation
} from '../../hoc/withAnimation';

// Import enhanced components
import EnhancedInputs from './EnhancedInputs';
import EnhancedListComponents from './EnhancedList';
import EnhancedDashboardCardComponents from './EnhancedDashboardCards';
import EnhancedSearchResultComponents from './EnhancedSearchResults';
import EnhancedTableComponents from './EnhancedTable';

/**
 * Enhanced versions of MUI components with animations
 * All components have Apple-like animations and feedback
 */

// Button components
export const EnhancedButton = withCompleteAnimation(Button);
export const EnhancedIconButton = withCompleteAnimation(IconButton);

// Text components
export const EnhancedTypography = withFadeUp(Typography);
export const EnhancedGradientTypography = withGradientText(Typography);

// Import enhanced input components (Apple-like animations)
export const {
  TextField: EnhancedTextField,
  Select: EnhancedSelect,
  Checkbox: EnhancedCheckbox,
  Radio: EnhancedRadio,
  Switch: EnhancedSwitch,
  Slider: EnhancedSlider,
  FormControlLabel: EnhancedFormControlLabel,
  Autocomplete: EnhancedAutocomplete,
  InputGroup: EnhancedInputGroup
} = EnhancedInputs;

// Container components
export const EnhancedBox = withFadeUp(Box);
export const EnhancedPaper = withFadeUp(
  withHoverEffect(Paper),
  { delay: 150 }
);
export const EnhancedCard = withFadeUp(
  withHoverEffect(Card),
  { delay: 200 }
);
export const EnhancedCardContent = withFadeUp(CardContent, { delay: 250 });
export const EnhancedCardHeader = withFadeUp(CardHeader, { delay: 220 });
export const EnhancedGrid = withFadeUp(Grid);
export const EnhancedDivider = withFadeUp(Divider);

// Import enhanced table components
export const {
  TableRow: EnhancedAnimatedTableRow,
  TableCell: EnhancedAnimatedTableCell,
  TableHeader: EnhancedAnimatedTableHeader,
  Table: EnhancedAnimatedTable,
  SortHeader: EnhancedTableSortHeader,
  ExpandableRow: EnhancedExpandableTableRow
} = EnhancedTableComponents;

// Legacy table components (to maintain backward compatibility)
export const EnhancedTableContainer = withFadeUp(TableContainer, { delay: 300 });
export const EnhancedTable = withFadeUp(Table);
export const EnhancedTableHead = withFadeUp(TableHead);
export const EnhancedTableBody = withFadeUp(TableBody, { delay: 350 });
export const EnhancedTableRow = withHoverEffect(TableRow);
export const EnhancedTableCell = withFadeUp(TableCell);

// Import enhanced list components
export const {
  List: AnimatedEnhancedList,
  ListItem: AnimatedEnhancedListItem,
  ListItemButton: AnimatedEnhancedListItemButton,
  NestedList: EnhancedNestedList
} = EnhancedListComponents;

// Import enhanced dashboard card components
export const {
  Card: EnhancedDashboardCard,
  CardGrid: EnhancedDashboardCardGrid
} = EnhancedDashboardCardComponents;

// Import enhanced search result components
export const {
  SearchResultCard: EnhancedSearchResultCard,
  SearchResults: EnhancedSearchResults
} = EnhancedSearchResultComponents;

// Legacy list components (to maintain backward compatibility)
export const EnhancedList = AnimatedEnhancedList;
export const EnhancedListItem = AnimatedEnhancedListItem;
export const EnhancedListItemButton = AnimatedEnhancedListItemButton;
export const EnhancedListItemText = withFadeUp(ListItemText);
export const EnhancedListItemIcon = withFadeUp(ListItemIcon);

// Feedback components
export const EnhancedAlert = withFadeUp(
  withSoundFeedback(Alert, 'error'),
  { delay: 100 }
);
export const EnhancedChip = withCompleteAnimation(Chip);
export const EnhancedBadge = withFadeUp(Badge);
export const EnhancedCircularProgress = withFadeUp(CircularProgress, { delay: 200 });

// Navigation components
export const EnhancedTabs = withFadeUp(Tabs);
export const EnhancedTab = withCompleteAnimation(Tab);

// Dialog components
export const EnhancedDialog = withFadeUp(Dialog);
export const EnhancedDialogTitle = withFadeUp(DialogTitle, { delay: 100 });
export const EnhancedDialogContent = withFadeUp(DialogContent, { delay: 150 });
export const EnhancedDialogActions = withFadeUp(DialogActions, { delay: 200 });

/**
 * Creates a grid of cards with staggered animations
 */
interface EnhancedCardGridProps {
  /** Array of child elements to render in cards */
  children: React.ReactNode[];
  /** Number of columns on extra-large screens */
  xlColumns?: number;
  /** Number of columns on large screens */
  lgColumns?: number;
  /** Number of columns on medium screens */
  mdColumns?: number;
  /** Number of columns on small screens */
  smColumns?: number;
  /** Number of columns on extra-small screens */
  xsColumns?: number;
  /** Spacing between cards */
  spacing?: number;
}

export const EnhancedCardGrid: React.FC<EnhancedCardGridProps> = ({
  children,
  xlColumns = 4,
  lgColumns = 3,
  mdColumns = 2,
  smColumns = 2,
  xsColumns = 1,
  spacing = 3,
}) => {
  return (
    <EnhancedGrid container spacing={spacing}>
      {children.map((child, index) => (
        <EnhancedGrid 
          item 
          xs={12 / xsColumns} 
          sm={12 / smColumns} 
          md={12 / mdColumns}
          lg={12 / lgColumns}
          xl={12 / xlColumns}
          key={index}
        >
          <EnhancedCard sx={{ height: '100%' }} style={{ delay: 100 + (index * 50) }}>
            <EnhancedCardContent>
              {child}
            </EnhancedCardContent>
          </EnhancedCard>
        </EnhancedGrid>
      ))}
    </EnhancedGrid>
  );
};

/**
 * Creates an enhanced data table with animations
 */
interface EnhancedDataTableProps<T> {
  /** Table headers */
  headers: string[];
  /** Data to display */
  data: T[];
  /** Function to get a unique key for each row */
  getRowKey: (row: T) => string;
  /** Function to render a cell */
  renderCell: (row: T, header: string, index: number) => React.ReactNode;
  /** Function called when a row is clicked */
  onRowClick?: (row: T) => void;
}

export function EnhancedDataTable<T>({
  headers,
  data,
  getRowKey,
  renderCell,
  onRowClick
}: EnhancedDataTableProps<T>) {
  return (
    <EnhancedTableContainer component={EnhancedPaper}>
      <EnhancedTable>
        <EnhancedTableHead>
          <TableRow>
            {headers.map((header) => (
              <EnhancedTableCell key={header}>
                <EnhancedTypography variant="subtitle2">
                  {header}
                </EnhancedTypography>
              </EnhancedTableCell>
            ))}
          </TableRow>
        </EnhancedTableHead>
        <EnhancedTableBody>
          {data.map((row, rowIndex) => (
            <EnhancedTableRow 
              key={getRowKey(row)}
              onClick={() => onRowClick && onRowClick(row)}
              sx={{ cursor: onRowClick ? 'pointer' : 'default' }}
            >
              {headers.map((header, cellIndex) => (
                <EnhancedTableCell key={`${getRowKey(row)}-${cellIndex}`}>
                  {renderCell(row, header, cellIndex)}
                </EnhancedTableCell>
              ))}
            </EnhancedTableRow>
          ))}
        </EnhancedTableBody>
      </EnhancedTable>
    </EnhancedTableContainer>
  );
}

/**
 * Enhanced form component with animations
 */
interface EnhancedFormProps {
  /** Form title */
  title: string;
  /** Form description */
  description?: string;
  /** Form children */
  children: React.ReactNode;
  /** Submit button text */
  submitText?: string;
  /** Cancel button text */
  cancelText?: string;
  /** Function called on form submission */
  onSubmit?: (e: React.FormEvent) => void;
  /** Function called when cancel is clicked */
  onCancel?: () => void;
}

export const EnhancedForm: React.FC<EnhancedFormProps> = ({
  title,
  description,
  children,
  submitText = 'Submit',
  cancelText = 'Cancel',
  onSubmit,
  onCancel
}) => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onSubmit) {
      onSubmit(e);
    }
  };
  
  return (
    <EnhancedPaper sx={{ p: 3 }}>
      <form onSubmit={handleSubmit}>
        <EnhancedGradientTypography variant="h5" sx={{ mb: 1 }}>
          {title}
        </EnhancedGradientTypography>
        
        {description && (
          <EnhancedTypography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            {description}
          </EnhancedTypography>
        )}
        
        <EnhancedBox sx={{ mb: 3 }}>
          {children}
        </EnhancedBox>
        
        <EnhancedBox sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
          {onCancel && (
            <EnhancedButton variant="outlined" onClick={onCancel}>
              {cancelText}
            </EnhancedButton>
          )}
          
          <EnhancedButton 
            variant="contained" 
            type="submit"
            sx={{
              background: 'linear-gradient(90deg, #0066B3, #19B5FE)',
            }}
          >
            {submitText}
          </EnhancedButton>
        </EnhancedBox>
      </form>
    </EnhancedPaper>
  );
};

export default {
  Button: EnhancedButton,
  IconButton: EnhancedIconButton,
  Typography: EnhancedTypography,
  GradientTypography: EnhancedGradientTypography,
  
  // Enhanced form inputs
  TextField: EnhancedTextField,
  Select: EnhancedSelect,
  Checkbox: EnhancedCheckbox,
  Radio: EnhancedRadio,
  Switch: EnhancedSwitch,
  Slider: EnhancedSlider,
  FormControlLabel: EnhancedFormControlLabel,
  Autocomplete: EnhancedAutocomplete,
  InputGroup: EnhancedInputGroup,
  
  Box: EnhancedBox,
  Paper: EnhancedPaper,
  Card: EnhancedCard,
  CardContent: EnhancedCardContent,
  CardHeader: EnhancedCardHeader,
  Grid: EnhancedGrid,
  Divider: EnhancedDivider,
  
  // Table components
  TableContainer: EnhancedTableContainer,
  Table: EnhancedAnimatedTable,
  TableHead: EnhancedAnimatedTableHeader,
  TableBody: EnhancedTableBody,
  TableRow: EnhancedAnimatedTableRow,
  TableCell: EnhancedAnimatedTableCell,
  TableSortHeader: EnhancedTableSortHeader,
  ExpandableTableRow: EnhancedExpandableTableRow,
  DataTable: EnhancedAnimatedTable,
  
  // List components
  List: AnimatedEnhancedList,
  ListItem: AnimatedEnhancedListItem,
  ListItemButton: AnimatedEnhancedListItemButton,
  ListItemText: EnhancedListItemText,
  ListItemIcon: EnhancedListItemIcon,
  NestedList: EnhancedNestedList,
  
  // Dashboard components
  DashboardCard: EnhancedDashboardCard,
  DashboardCardGrid: EnhancedDashboardCardGrid,
  
  // Search components
  SearchResultCard: EnhancedSearchResultCard,
  SearchResults: EnhancedSearchResults,
  
  // Feedback components
  Alert: EnhancedAlert,
  Chip: EnhancedChip,
  Badge: EnhancedBadge,
  CircularProgress: EnhancedCircularProgress,
  
  // Navigation components
  Tabs: EnhancedTabs,
  Tab: EnhancedTab,
  
  // Dialog components
  Dialog: EnhancedDialog,
  DialogTitle: EnhancedDialogTitle,
  DialogContent: EnhancedDialogContent,
  DialogActions: EnhancedDialogActions,
  
  // Complex components
  CardGrid: EnhancedCardGrid,
  Form: EnhancedForm
};