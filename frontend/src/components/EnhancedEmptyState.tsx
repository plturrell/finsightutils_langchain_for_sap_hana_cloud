import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, useTheme, alpha } from '@mui/material';
import { animated } from '@react-spring/web';
import { useEnhancedEmptyStateAnimation } from '../hooks/useAnimations';
import { useAnimationContext } from '../context/AnimationContext';
import { soundEffects } from '../utils/soundEffects';

// Create animated versions of MUI components
const AnimatedBox = animated(Box);
const AnimatedTypography = animated(Typography);
const AnimatedButton = animated(Button);

interface EnhancedEmptyStateProps {
  /** The icon to display */
  icon: React.ReactNode;
  /** Primary title text */
  title: string;
  /** Secondary description text */
  description: string;
  /** Button text (if actionButton is true) */
  buttonText?: string;
  /** Button click handler (if actionButton is true) */
  onButtonClick?: () => void;
  /** Whether to show an action button */
  actionButton?: boolean;
}

/**
 * Enhanced empty state component with Apple-inspired animations
 * Features advanced animations and interactions inspired by iPadOS
 */
const EnhancedEmptyState: React.FC<EnhancedEmptyStateProps> = ({
  icon,
  title,
  description,
  buttonText = 'Get Started',
  onButtonClick,
  actionButton = true,
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [visible, setVisible] = useState(false);
  const [isButtonHovered, setIsButtonHovered] = useState(false);
  
  // Get the enhanced empty state animations
  const { containerAnimation, iconAnimation, textAnimation } = useEnhancedEmptyStateAnimation(visible);
  
  // Button animation
  const buttonAnimation = {
    transform: isButtonHovered ? 'translateY(-3px)' : 'translateY(0)',
    boxShadow: isButtonHovered 
      ? `0 6px 15px ${alpha(theme.palette.primary.main, 0.3)}`
      : `0 2px 5px ${alpha(theme.palette.primary.main, 0.1)}`,
    background: isButtonHovered
      ? 'linear-gradient(90deg, #0066B3, #19B5FE)'
      : theme.palette.primary.main,
  };
  
  // Show the animation after mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Play sound on button hover (if enabled)
  useEffect(() => {
    if (isButtonHovered && animationsEnabled) {
      soundEffects.hover();
    }
  }, [isButtonHovered, animationsEnabled]);
  
  return (
    <AnimatedBox
      style={containerAnimation}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        padding: 4,
        height: '100%',
        minHeight: 300,
      }}
    >
      {/* Animated icon */}
      <AnimatedBox
        style={iconAnimation}
        sx={{
          mb: 3,
          color: alpha(theme.palette.primary.main, 0.7),
          fontSize: 80,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          filter: `drop-shadow(0 4px 8px ${alpha(theme.palette.primary.main, 0.3)})`,
        }}
      >
        {icon}
      </AnimatedBox>
      
      {/* Text content */}
      <AnimatedTypography
        variant="h4"
        style={textAnimation}
        sx={{
          mb: 1,
          fontWeight: 600,
          background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        {title}
      </AnimatedTypography>
      
      <AnimatedTypography
        variant="body1"
        style={{
          ...textAnimation,
          opacity: visible ? 0.7 : 0,
        }}
        sx={{
          mb: 4,
          maxWidth: 500,
        }}
      >
        {description}
      </AnimatedTypography>
      
      {/* Action button (if enabled) */}
      {actionButton && (
        <AnimatedButton
          variant="contained"
          size="large"
          onClick={() => {
            if (animationsEnabled) {
              soundEffects.tap();
            }
            if (onButtonClick) {
              onButtonClick();
            }
          }}
          onMouseEnter={() => setIsButtonHovered(true)}
          onMouseLeave={() => setIsButtonHovered(false)}
          style={{
            ...textAnimation,
            ...buttonAnimation,
          }}
          sx={{
            borderRadius: 3,
            px: 4,
            py: 1,
            fontWeight: 500,
            transition: 'all 0.3s ease',
            position: 'relative',
            overflow: 'hidden',
            '&::after': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
              transform: 'translateX(-100%)',
              transition: 'transform 0.6s ease',
            },
            '&:hover::after': {
              transform: 'translateX(100%)',
            },
          }}
        >
          {buttonText}
        </AnimatedButton>
      )}
    </AnimatedBox>
  );
};

export default EnhancedEmptyState;