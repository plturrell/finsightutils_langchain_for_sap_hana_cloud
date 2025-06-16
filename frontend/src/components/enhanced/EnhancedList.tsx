import React, { useState, useEffect } from 'react';
import {
  List,
  ListProps,
  ListItem,
  ListItemProps,
  ListItemButton,
  ListItemButtonProps,
  ListItemText,
  ListItemIcon,
  Collapse,
  alpha,
  useTheme
} from '@mui/material';
import { useSpring, animated, useTrail, config } from '@react-spring/web';
import { useAnimationContext } from '../../context/AnimationContext';
import { soundEffects } from '../../utils/soundEffects';

// Create animated versions of MUI components
const AnimatedList = animated(List);
const AnimatedListItem = animated(ListItem);
const AnimatedListItemButton = animated(ListItemButton);
const AnimatedListItemText = animated(ListItemText);
const AnimatedListItemIcon = animated(ListItemIcon);
const AnimatedCollapse = animated(Collapse);

/**
 * Enhanced List component with Apple-like animations
 */
export const AnimatedEnhancedList: React.FC<ListProps & { delayMultiplier?: number }> = ({ 
  children, 
  delayMultiplier = 50,
  ...props 
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [visible, setVisible] = useState(false);
  
  // Set visible after mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Create trail animation for list items
  const childArray = React.Children.toArray(children);
  const trail = useTrail(childArray.length, {
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0, 
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(20px)' 
    },
    config: { tension: 280, friction: 60 },
    immediate: !animationsEnabled
  });
  
  return (
    <AnimatedList
      style={useSpring({
        from: { opacity: 0 },
        to: { opacity: visible && animationsEnabled ? 1 : 0 },
        config: { tension: 280, friction: 60 },
        immediate: !animationsEnabled
      })}
      {...props}
    >
      {trail.map((style, index) => {
        const child = childArray[index];
        
        // Skip non-element children
        if (!React.isValidElement(child)) {
          return child;
        }
        
        // Apply animation style to list item
        return React.cloneElement(child, {
          ...child.props,
          style: {
            ...child.props.style,
            ...style,
            // Add delay based on index
            delay: delayMultiplier * index
          }
        });
      })}
    </AnimatedList>
  );
};

/**
 * Enhanced ListItem with Apple-like animations and interactions
 */
export const AnimatedEnhancedListItem: React.FC<ListItemProps & { 
  soundType?: 'tap' | 'hover';
  hoverEffect?: boolean;
}> = ({ 
  children, 
  soundType = 'tap',
  hoverEffect = true,
  ...props 
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Hover animation
  const hoverStyle = useSpring({
    transform: isHovered && hoverEffect && animationsEnabled ? 'translateX(5px)' : 'translateX(0px)',
    backgroundColor: isHovered && hoverEffect && animationsEnabled 
      ? alpha(theme.palette.primary.main, 0.05)
      : 'transparent',
    config: { tension: 350, friction: 18 }, // Apple-like immediate response
    immediate: !animationsEnabled
  });
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
    if (animationsEnabled && soundType === 'hover') {
      soundEffects.hover();
    }
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (animationsEnabled && soundType === 'tap') {
      soundEffects.tap();
    }
    
    // Call original onClick if provided
    if (props.onClick) {
      props.onClick(e);
    }
  };
  
  return (
    <AnimatedListItem
      {...props}
      style={{ ...hoverStyle, ...props.style }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {children}
    </AnimatedListItem>
  );
};

/**
 * Enhanced ListItemButton with Apple-like animations and interactions
 */
export const AnimatedEnhancedListItemButton: React.FC<ListItemButtonProps & {
  soundType?: 'tap' | 'hover';
  hoverEffect?: boolean;
}> = ({ 
  children, 
  soundType = 'tap',
  hoverEffect = true,
  ...props 
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Hover animation
  const hoverStyle = useSpring({
    transform: isHovered && hoverEffect && animationsEnabled ? 'translateX(5px)' : 'translateX(0px)',
    backgroundColor: isHovered && hoverEffect && animationsEnabled 
      ? alpha(theme.palette.primary.main, 0.05)
      : 'transparent',
    config: { tension: 350, friction: 18 }, // Apple-like immediate response
    immediate: !animationsEnabled
  });
  
  // Selected state animation
  const selectedStyle = useSpring({
    transform: props.selected && animationsEnabled ? 'translateX(3px)' : 'translateX(0px)',
    fontWeight: props.selected ? 600 : 400,
    immediate: !animationsEnabled
  });
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
    if (animationsEnabled && soundType === 'hover') {
      soundEffects.hover();
    }
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (animationsEnabled && soundType === 'tap') {
      soundEffects.tap();
    }
    
    // Call original onClick if provided
    if (props.onClick) {
      props.onClick(e);
    }
  };
  
  return (
    <AnimatedListItemButton
      {...props}
      style={{ 
        ...hoverStyle, 
        ...selectedStyle,
        ...props.style,
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      sx={{
        borderRadius: 1,
        mx: 0.5,
        px: 1.5,
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s ease',
        ...(props.selected && {
          '&::before': {
            content: '""',
            position: 'absolute',
            left: 0,
            top: '25%',
            height: '50%',
            width: 3,
            borderRadius: '0 2px 2px 0',
            backgroundColor: theme.palette.primary.main,
          },
        }),
        ...props.sx
      }}
    >
      {children}
    </AnimatedListItemButton>
  );
};

/**
 * Enhanced nested list with expandable sections
 */
export interface EnhancedNestedListProps {
  /** List sections with title and children */
  sections: {
    id: string;
    title: string;
    icon?: React.ReactNode;
    children: React.ReactNode;
  }[];
  /** Initially expanded section IDs */
  initialExpanded?: string[];
}

export const EnhancedNestedList: React.FC<EnhancedNestedListProps> = ({
  sections,
  initialExpanded = []
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [expanded, setExpanded] = useState<string[]>(initialExpanded);
  
  const toggleSection = (sectionId: string) => {
    if (expanded.includes(sectionId)) {
      setExpanded(expanded.filter(id => id !== sectionId));
    } else {
      setExpanded([...expanded, sectionId]);
      
      // Play sound if animations are enabled
      if (animationsEnabled) {
        soundEffects.switch();
      }
    }
  };
  
  return (
    <AnimatedEnhancedList sx={{ width: '100%' }}>
      {sections.map((section) => {
        const isExpanded = expanded.includes(section.id);
        
        return (
          <React.Fragment key={section.id}>
            <AnimatedEnhancedListItemButton
              onClick={() => toggleSection(section.id)}
              sx={{
                borderRadius: 1,
                mb: 0.5,
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.05),
                },
                ...(isExpanded && {
                  backgroundColor: alpha(theme.palette.primary.main, 0.07),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  },
                })
              }}
            >
              {section.icon && (
                <ListItemIcon 
                  sx={{ 
                    minWidth: 40,
                    color: isExpanded ? theme.palette.primary.main : 'inherit',
                    transition: 'all 0.3s ease',
                    transform: isExpanded ? 'scale(1.1)' : 'scale(1)',
                  }}
                >
                  {section.icon}
                </ListItemIcon>
              )}
              <ListItemText 
                primary={section.title}
                primaryTypographyProps={{
                  sx: {
                    fontWeight: isExpanded ? 600 : 400,
                    color: isExpanded ? theme.palette.primary.main : 'inherit',
                    transition: 'all 0.3s ease',
                  }
                }}
              />
              <animated.div
                style={useSpring({
                  transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                  config: { tension: 300, friction: 20 },
                  immediate: !animationsEnabled
                })}
              >
                <ExpandMoreIcon />
              </animated.div>
            </AnimatedEnhancedListItemButton>
            
            <AnimatedCollapse 
              in={isExpanded} 
              timeout={300}
              style={useSpring({
                from: { opacity: 0, height: 0 },
                to: { 
                  opacity: isExpanded ? 1 : 0, 
                  height: isExpanded ? 'auto' : 0 
                },
                config: { tension: 280, friction: 60 },
                immediate: !animationsEnabled
              })}
            >
              <Box sx={{ pl: 2 }}>
                {section.children}
              </Box>
            </AnimatedCollapse>
          </React.Fragment>
        );
      })}
    </AnimatedEnhancedList>
  );
};

// Import after definition to avoid circular references
import { ExpandMore as ExpandMoreIcon } from '@mui/icons-material';

export default {
  List: AnimatedEnhancedList,
  ListItem: AnimatedEnhancedListItem,
  ListItemButton: AnimatedEnhancedListItemButton,
  NestedList: EnhancedNestedList
};