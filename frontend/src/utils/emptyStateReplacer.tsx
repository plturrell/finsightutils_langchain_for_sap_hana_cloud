import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import EnhancedEmptyState from '../components/EnhancedEmptyState';
import {
  Storage as StorageIcon,
  Search as SearchIcon,
  Insights as InsightsIcon,
  Warning as WarningIcon,
  ErrorOutline as ErrorIcon,
  DataArray as DataArrayIcon,
  Psychology as PsychologyIcon,
  Lightbulb as LightbulbIcon,
  Settings as SettingsIcon,
  Code as CodeIcon,
  Dataset as DatasetIcon,
} from '@mui/icons-material';

/**
 * Types of empty states
 */
export type EmptyStateType = 
  | 'noData' 
  | 'noResults' 
  | 'noConnection'
  | 'noAccess'
  | 'error'
  | 'loading'
  | 'search'
  | 'firstUse'
  | 'vector'
  | 'reasoning'
  | 'settings'
  | 'developer';

/**
 * Configuration for each empty state type
 */
const emptyStateConfig: Record<
  EmptyStateType, 
  { icon: React.ReactNode; title: string; description: string; }
> = {
  noData: {
    icon: <DataArrayIcon fontSize="inherit" />,
    title: "No Data Available",
    description: "There is no data to display at this time. Check back later or try a different selection."
  },
  noResults: {
    icon: <SearchIcon fontSize="inherit" />,
    title: "No Results Found",
    description: "Your search didn't return any results. Try using different keywords or filters."
  },
  noConnection: {
    icon: <StorageIcon fontSize="inherit" />,
    title: "Connection Error",
    description: "We couldn't connect to the database. Please check your connection and try again."
  },
  noAccess: {
    icon: <WarningIcon fontSize="inherit" />,
    title: "Access Denied",
    description: "You don't have permission to access this resource. Contact your administrator for help."
  },
  error: {
    icon: <ErrorIcon fontSize="inherit" />,
    title: "Something Went Wrong",
    description: "An error occurred while processing your request. Please try again later."
  },
  loading: {
    icon: <CircularProgress size={60} />,
    title: "Loading Data",
    description: "We're retrieving your data. This should only take a moment..."
  },
  search: {
    icon: <SearchIcon fontSize="inherit" />,
    title: "Start Searching",
    description: "Enter your search terms to find information across your data sources."
  },
  firstUse: {
    icon: <LightbulbIcon fontSize="inherit" />,
    title: "Welcome to SAP HANA Explorer",
    description: "This is your first time here. Start by exploring your data or creating a vector store."
  },
  vector: {
    icon: <DatasetIcon fontSize="inherit" />,
    title: "No Vector Data",
    description: "Create vector embeddings to enable semantic search and exploration."
  },
  reasoning: {
    icon: <PsychologyIcon fontSize="inherit" />,
    title: "Ready for Reasoning",
    description: "Ask a question to see AI reasoning capabilities in action."
  },
  settings: {
    icon: <SettingsIcon fontSize="inherit" />,
    title: "Configure Settings",
    description: "Customize your experience by adjusting settings to match your preferences."
  },
  developer: {
    icon: <CodeIcon fontSize="inherit" />,
    title: "Developer Mode",
    description: "Build custom integrations or modify existing functionality using our developer tools."
  }
};

/**
 * Interface for empty state replacer props
 */
interface EmptyStateReplacerProps {
  children: React.ReactNode;
  /**
   * Function to determine if a component is an empty state
   * Return the type of empty state if it is, null otherwise
   */
  isEmptyState: (component: React.ReactElement) => EmptyStateType | null;
  /** Whether to apply animations */
  animate?: boolean;
}

/**
 * A component that replaces empty states with enhanced versions
 */
export const EmptyStateReplacer: React.FC<EmptyStateReplacerProps> = ({
  children,
  isEmptyState,
  animate = true
}) => {
  // Check if the children should be replaced with an enhanced empty state
  const processChildren = (childrenToProcess: React.ReactNode): React.ReactNode => {
    return React.Children.map(childrenToProcess, child => {
      if (!React.isValidElement(child)) {
        return child;
      }
      
      // Check if this component is an empty state
      const emptyStateType = isEmptyState(child);
      if (emptyStateType) {
        const config = emptyStateConfig[emptyStateType];
        
        // Replace with enhanced empty state
        return (
          <EnhancedEmptyState
            icon={config.icon}
            title={config.title}
            description={config.description}
            buttonText={emptyStateType === 'loading' ? undefined : "Get Started"}
            actionButton={emptyStateType !== 'loading' && emptyStateType !== 'error'}
          />
        );
      }
      
      // Recursively process children
      if (child.props && child.props.children) {
        const newChildren = processChildren(child.props.children);
        return React.cloneElement(child, { ...child.props, children: newChildren });
      }
      
      // Return unchanged
      return child;
    });
  };
  
  return <>{processChildren(children)}</>;
};

/**
 * Default empty state detector
 * Looks for common patterns that indicate empty states
 * @param component Component to check
 * @returns Empty state type if detected, null otherwise
 */
export const defaultEmptyStateDetector = (component: React.ReactElement): EmptyStateType | null => {
  // If not an element, it's not an empty state
  if (!React.isValidElement(component)) {
    return null;
  }
  
  // Check if it's a Box or div with a message and icon
  if (
    (component.type === Box || component.type === 'div' || component.type === 'section') &&
    component.props.children
  ) {
    const children = component.props.children;
    
    // Check if it has Typography children with specific text
    const hasEmptyStateText = React.Children.toArray(children).some(child => {
      if (!React.isValidElement(child)) return false;
      
      if (child.type === Typography || child.type === 'p' || child.type === 'h1' || child.type === 'h2' || child.type === 'h3') {
        const text = child.props.children;
        if (typeof text === 'string') {
          const lowerText = text.toLowerCase();
          
          // Common empty state text patterns
          if (
            lowerText.includes('no data') || 
            lowerText.includes('empty') || 
            lowerText.includes('no results') ||
            lowerText.includes('not found') ||
            lowerText.includes('no items') ||
            lowerText.includes('start by') ||
            lowerText.includes('get started')
          ) {
            // Determine the type based on text
            if (lowerText.includes('search') || lowerText.includes('results')) {
              return 'noResults';
            } else if (lowerText.includes('connect')) {
              return 'noConnection';
            } else {
              return 'noData';
            }
          }
        }
      }
      return false;
    });
    
    if (hasEmptyStateText) {
      return 'noData';
    }
  }
  
  // Check for loading states
  if (component.props.children) {
    const hasLoadingIndicator = React.Children.toArray(component.props.children).some(child => {
      if (!React.isValidElement(child)) return false;
      return child.type === CircularProgress;
    });
    
    if (hasLoadingIndicator) {
      return 'loading';
    }
  }
  
  return null;
};

/**
 * HOC that adds enhanced empty states to a component
 * @param WrappedComponent Component to enhance
 * @param options Options for empty state detection
 * @returns Enhanced component with improved empty states
 */
export function withEnhancedEmptyStates<P extends {}>(
  WrappedComponent: React.ComponentType<P>,
  emptyStateDetector = defaultEmptyStateDetector
): React.FC<P> {
  const EnhancedComponent: React.FC<P> = (props) => {
    return (
      <EmptyStateReplacer isEmptyState={emptyStateDetector}>
        <WrappedComponent {...props} />
      </EmptyStateReplacer>
    );
  };
  
  // Set display name for debugging
  const displayName = WrappedComponent.displayName || WrappedComponent.name || 'Component';
  EnhancedComponent.displayName = `withEnhancedEmptyStates(${displayName})`;
  
  return EnhancedComponent;
}