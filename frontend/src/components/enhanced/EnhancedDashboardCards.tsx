import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardProps,
  Typography,
  Box,
  alpha,
  useTheme
} from '@mui/material';
import { animated, useSpring } from '@react-spring/web';
import { useAnimationContext } from '../../context/AnimationContext';
import { useBatchAnimations } from '../../hooks/useAnimations';
import { soundEffects } from '../../utils/soundEffects';

// Create animated versions of MUI components
const AnimatedCard = animated(Card);
const AnimatedTypography = animated(Typography);

/**
 * Type definitions for batch animation items
 */
interface BatchAnimationItem {
  /** Reference key for the animation */
  key: string;
  /** Initial delay before starting animation (ms) */
  delay?: number;
  /** Custom animation configuration */
  config?: {
    tension?: number;
    friction?: number;
    mass?: number;
  };
  /** Custom from values */
  from?: Record<string, any>;
  /** Custom to values */
  to?: Record<string, any>;
  /** Whether this animation is enabled */
  enabled?: boolean;
}

/**
 * Props for the EnhancedDashboardCard component
 */
export interface EnhancedDashboardCardProps extends CardProps {
  /** Title for the card */
  title: React.ReactNode;
  /** Content for the card */
  children: React.ReactNode;
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Animation delay in milliseconds */
  animationDelay?: number;
  /** Top border gradient colors (start, end) */
  gradientColors?: [string, string];
  /** Card index for staggered animations */
  index?: number;
  /** Callback function when card is clicked */
  onClick?: () => void;
}

/**
 * Enhanced Dashboard Card with Apple-inspired animations and interactions
 * Uses batch animations for improved performance
 */
export const EnhancedDashboardCard: React.FC<EnhancedDashboardCardProps> = ({
  title,
  children,
  animationsVisible = true,
  animationDelay = 0,
  gradientColors = ['primary.main', 'primary.light'],
  index = 0,
  onClick,
  ...props
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Calculate additional delay based on index
  const totalDelay = animationDelay + (index * 100);
  
  // Set up batch animations for all card elements
  const animations = useBatchAnimations(
    animationsVisible,
    [
      {
        key: 'card',
        from: { opacity: 0, transform: 'translateY(40px) scale(0.95)' },
        to: { 
          opacity: animationsEnabled && animationsVisible ? 1 : 0, 
          transform: animationsEnabled && animationsVisible ? 'scale(1) translateY(0)' : 'translateY(40px) scale(0.95)' 
        },
        delay: totalDelay,
        config: { tension: 280, friction: 60, mass: 1 }
      },
      {
        key: 'title',
        from: { opacity: 0, transform: 'translateY(-10px)' },
        to: { 
          opacity: animationsEnabled && animationsVisible ? 1 : 0, 
          transform: animationsEnabled && animationsVisible ? 'translateY(0)' : 'translateY(-10px)' 
        },
        delay: totalDelay + 100,
        config: { tension: 280, friction: 60, mass: 1 }
      },
      {
        key: 'content',
        from: { opacity: 0, transform: 'scale(0.95)' },
        to: { 
          opacity: animationsEnabled && animationsVisible ? 1 : 0, 
          transform: animationsEnabled && animationsVisible ? 'scale(1)' : 'scale(0.95)' 
        },
        delay: totalDelay + 150,
        config: { tension: 280, friction: 60, mass: 1 }
      },
      {
        key: 'hover',
        to: { 
          transform: isHovered && animationsEnabled ? 'translateY(-4px)' : 'translateY(0)',
          boxShadow: isHovered && animationsEnabled 
            ? `0 8px 24px ${alpha(theme.palette.primary.main, 0.15)}`
            : '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)'
        },
        config: { tension: 350, friction: 18 } // Apple-like immediate response
      }
    ]
  );
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
    if (animationsEnabled) {
      soundEffects.hover();
    }
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  const handleClick = () => {
    if (onClick) {
      onClick();
      if (animationsEnabled) {
        soundEffects.tap();
      }
    }
  };
  
  return (
    <AnimatedCard
      {...props}
      style={{
        ...animations.card,
        ...animations.hover,
        ...props.style
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        transition: 'all 0.3s ease',
        overflow: 'hidden',
        cursor: onClick ? 'pointer' : 'default',
        willChange: 'transform, box-shadow, opacity',
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '4px',
          background: theme => `linear-gradient(90deg, ${theme.palette[gradientColors[0].split('.')[0]][gradientColors[0].split('.')[1]]}, ${theme.palette[gradientColors[1].split('.')[0]][gradientColors[1].split('.')[1]]})`,
          opacity: 0,
          transition: 'opacity 0.3s ease',
        },
        '&:hover::after': {
          opacity: 1,
        },
        ...props.sx
      }}
    >
      <CardHeader
        title={
          <AnimatedTypography 
            variant="h6"
            style={animations.title}
          >
            {title}
          </AnimatedTypography>
        }
        titleTypographyProps={{ variant: 'h6' }}
        sx={{ pb: 0 }}
      />
      <CardContent sx={{ pt: 2, flexGrow: 1 }}>
        <animated.div style={animations.content}>
          {children}
        </animated.div>
      </CardContent>
    </AnimatedCard>
  );
};

/**
 * Props for the EnhancedDashboardCardGrid component
 */
export interface EnhancedDashboardCardGridProps {
  /** Children to render as cards */
  children: React.ReactNode[];
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Base animation delay in milliseconds */
  baseDelay?: number;
}

/**
 * Creates a grid of dashboard cards with optimized batch animations
 * This component significantly improves performance over individually animated cards
 */
export const EnhancedDashboardCardGrid: React.FC<EnhancedDashboardCardGridProps> = ({
  children,
  animationsVisible = true,
  baseDelay = 0
}) => {
  const { animationsEnabled } = useAnimationContext();
  
  // Create multiple cards with staggered animations
  return (
    <>
      {React.Children.map(children, (child, index) => {
        // Apply EnhancedDashboardCard to each child
        if (React.isValidElement(child)) {
          return React.cloneElement(child, {
            animationsVisible,
            animationDelay: baseDelay + (index * 100),
            index,
          });
        }
        return child;
      })}
    </>
  );
};

export default {
  Card: EnhancedDashboardCard,
  CardGrid: EnhancedDashboardCardGrid
};