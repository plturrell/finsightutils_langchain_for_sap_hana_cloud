import React from 'react';
import { Button, IconButton } from '@mui/material';
import * as Enhanced from '../components/enhanced';
import { soundEffects } from './soundEffects';

/**
 * Utility functions to apply enhancements to existing components
 */

/**
 * Recursively replaces all Button components with EnhancedButton
 * @param children React children to process
 * @returns Enhanced children with replaced buttons
 */
export const enhanceButtons = (children: React.ReactNode): React.ReactNode => {
  return React.Children.map(children, child => {
    // If child is not an element (text, etc.), return it as-is
    if (!React.isValidElement(child)) {
      return child;
    }
    
    // Check if child is a Button or IconButton
    if (
      child.type === Button || 
      child.type === IconButton ||
      (typeof child.type === 'string' && 
        (child.type === 'button' || child.type.includes('Button')))
    ) {
      // Determine the component type
      const EnhancedComponent = child.type === IconButton 
        ? Enhanced.EnhancedIconButton 
        : Enhanced.EnhancedButton;
      
      // Copy all props
      const props = { ...child.props };
      
      // Add sound effect if it doesn't already have an onClick
      if (!props.onClick) {
        props.onClick = () => soundEffects.tap();
      }
      
      // Return enhanced button with all props and children
      return <EnhancedComponent {...props} />;
    }
    
    // Recursively process children
    if (child.props && child.props.children) {
      const newChildren = enhanceButtons(child.props.children);
      return React.cloneElement(child, { ...child.props, children: newChildren });
    }
    
    // Return the element as is
    return child;
  });
};

/**
 * Enhances a component by replacing all standard components with enhanced versions
 * @param Component The component to enhance
 * @returns Enhanced component
 */
export function enhanceComponent<P extends {}>(Component: React.ComponentType<P>): React.FC<P> {
  const EnhancedComponent: React.FC<P> = (props) => {
    const renderedComponent = <Component {...props} />;
    
    // Process and enhance the rendered output
    return enhanceButtons(renderedComponent) as React.ReactElement;
  };
  
  // Set display name for debugging
  const displayName = Component.displayName || Component.name || 'Component';
  EnhancedComponent.displayName = `Enhanced(${displayName})`;
  
  return EnhancedComponent;
}

/**
 * Higher-order component that enhances all child components with animations
 * @param WrappedComponent Component to enhance
 * @returns Enhanced component
 */
export function withEnhancedComponents<P extends {}>(WrappedComponent: React.ComponentType<P>): React.FC<P> {
  const EnhancedComponent: React.FC<P> = (props) => {
    return (
      <WrappedComponent
        {...props}
        components={{
          Button: Enhanced.EnhancedButton,
          IconButton: Enhanced.EnhancedIconButton,
          TextField: Enhanced.EnhancedTextField,
          Switch: Enhanced.EnhancedSwitch,
          Checkbox: Enhanced.EnhancedCheckbox,
          Radio: Enhanced.EnhancedRadio,
          Slider: Enhanced.EnhancedSlider,
          Card: Enhanced.EnhancedCard,
          CardContent: Enhanced.EnhancedCardContent,
          CardHeader: Enhanced.EnhancedCardHeader,
          List: Enhanced.EnhancedList,
          ListItem: Enhanced.EnhancedListItem,
          ListItemButton: Enhanced.EnhancedListItemButton,
          Table: Enhanced.EnhancedTable,
          TableRow: Enhanced.EnhancedTableRow,
          TableCell: Enhanced.EnhancedTableCell,
          ...props.components
        }}
      />
    );
  };
  
  // Set display name for debugging
  const displayName = WrappedComponent.displayName || WrappedComponent.name || 'Component';
  EnhancedComponent.displayName = `withEnhancedComponents(${displayName})`;
  
  return EnhancedComponent;
}

/**
 * Replace standard MUI components with enhanced versions in a component tree
 * @param children Children to enhance
 * @returns Enhanced children with all components replaced with enhanced versions
 */
export const enhanceAllComponents = (children: React.ReactNode): React.ReactNode => {
  return React.Children.map(children, child => {
    if (!React.isValidElement(child)) {
      return child;
    }
    
    // Map of components to their enhanced versions
    const componentMap: Record<any, React.ElementType> = {
      Button: Enhanced.EnhancedButton,
      IconButton: Enhanced.EnhancedIconButton,
      Typography: Enhanced.EnhancedTypography,
      TextField: Enhanced.EnhancedTextField,
      Switch: Enhanced.EnhancedSwitch,
      Checkbox: Enhanced.EnhancedCheckbox,
      Radio: Enhanced.EnhancedRadio,
      Slider: Enhanced.EnhancedSlider,
      Box: Enhanced.EnhancedBox,
      Paper: Enhanced.EnhancedPaper,
      Card: Enhanced.EnhancedCard,
      CardContent: Enhanced.EnhancedCardContent,
      CardHeader: Enhanced.EnhancedCardHeader,
      Grid: Enhanced.EnhancedGrid,
      Divider: Enhanced.EnhancedDivider,
      TableContainer: Enhanced.EnhancedTableContainer,
      Table: Enhanced.EnhancedTable,
      TableHead: Enhanced.EnhancedTableHead,
      TableBody: Enhanced.EnhancedTableBody,
      TableRow: Enhanced.EnhancedTableRow,
      TableCell: Enhanced.EnhancedTableCell,
      List: Enhanced.EnhancedList,
      ListItem: Enhanced.EnhancedListItem,
      ListItemButton: Enhanced.EnhancedListItemButton,
      ListItemText: Enhanced.EnhancedListItemText,
      ListItemIcon: Enhanced.EnhancedListItemIcon,
      Alert: Enhanced.EnhancedAlert,
      Chip: Enhanced.EnhancedChip,
      Badge: Enhanced.EnhancedBadge,
      CircularProgress: Enhanced.EnhancedCircularProgress,
      Tabs: Enhanced.EnhancedTabs,
      Tab: Enhanced.EnhancedTab,
      Dialog: Enhanced.EnhancedDialog,
      DialogTitle: Enhanced.EnhancedDialogTitle,
      DialogContent: Enhanced.EnhancedDialogContent,
      DialogActions: Enhanced.EnhancedDialogActions,
    };
    
    // Check if the component should be replaced
    const EnhancedComponent = componentMap[child.type as any];
    if (EnhancedComponent) {
      // Recursively process children
      const newChildren = child.props.children 
        ? enhanceAllComponents(child.props.children)
        : child.props.children;
      
      // Return enhanced component with all props and processed children
      return <EnhancedComponent {...child.props}>{newChildren}</EnhancedComponent>;
    }
    
    // Recursively process children
    if (child.props && child.props.children) {
      const newChildren = enhanceAllComponents(child.props.children);
      return React.cloneElement(child, { ...child.props, children: newChildren });
    }
    
    // Return the element as is
    return child;
  });
};

/**
 * A component that enhances all children with animations
 */
export const EnhancedComponentProvider: React.FC<{ children: React.ReactNode }> = ({ 
  children 
}) => {
  return <>{enhanceAllComponents(children)}</>;
};