import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  Divider,
  Grid,
  IconButton,
  Tooltip,
  useTheme,
  alpha
} from '@mui/material';
import { 
  ContentCopy as CopyIcon,
  Launch as LaunchIcon
} from '@mui/icons-material';
import { useSpring, animated } from '@react-spring/web';
import { useAnimationContext } from '../../context/AnimationContext';
import { useBatchAnimations } from '../../hooks/useAnimations';
import { soundEffects } from '../../utils/soundEffects';

// Animated versions of MUI components
const AnimatedPaper = animated(Paper);
const AnimatedBox = animated(Box);
const AnimatedTypography = animated(Typography);
const AnimatedDivider = animated(Divider);
const AnimatedChip = animated(Chip);

/**
 * Interface for a search result
 */
export interface SearchResultItem {
  document: {
    page_content: string;
    metadata: Record<string, any>;
  };
  score: number;
}

/**
 * Props for the EnhancedSearchResultCard component
 */
export interface EnhancedSearchResultCardProps {
  /** The search result data */
  result: SearchResultItem;
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Animation delay in milliseconds */
  animationDelay?: number;
  /** Result index for staggered animations */
  index?: number;
  /** Whether to enable hover effect */
  enableHover?: boolean;
  /** Function called when the copy button is clicked */
  onCopy?: () => void;
}

/**
 * Enhanced Search Result Card with Apple-inspired animations
 * Uses batch animations for improved performance
 */
export const EnhancedSearchResultCard: React.FC<EnhancedSearchResultCardProps> = ({
  result,
  animationsVisible = true,
  animationDelay = 0,
  index = 0,
  enableHover = true,
  onCopy
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Calculate the staggered delay based on index
  const totalDelay = animationDelay + (index * 50);
  
  // Use batch animations for all card elements
  const animations = useBatchAnimations(
    animationsVisible,
    [
      {
        key: 'card',
        from: { opacity: 0, transform: 'translateY(30px)' },
        to: { 
          opacity: animationsEnabled && animationsVisible ? 1 : 0, 
          transform: animationsEnabled && animationsVisible ? 'translateY(0)' : 'translateY(30px)'
        },
        delay: totalDelay,
        config: { tension: 280, friction: 60, mass: 1 }
      },
      {
        key: 'title',
        from: { opacity: 0, transform: 'translateY(-5px)' },
        to: { 
          opacity: animationsEnabled && animationsVisible ? 1 : 0, 
          transform: animationsEnabled && animationsVisible ? 'translateY(0)' : 'translateY(-5px)'
        },
        delay: totalDelay + 50,
        config: { tension: 280, friction: 60, mass: 1 }
      },
      {
        key: 'content',
        from: { opacity: 0 },
        to: { opacity: animationsEnabled && animationsVisible ? 1 : 0 },
        delay: totalDelay + 100,
        config: { tension: 280, friction: 60, mass: 1 }
      },
      {
        key: 'metadata',
        from: { opacity: 0, transform: 'translateY(10px)' },
        to: { 
          opacity: animationsEnabled && animationsVisible ? 1 : 0, 
          transform: animationsEnabled && animationsVisible ? 'translateY(0)' : 'translateY(10px)'
        },
        delay: totalDelay + 150,
        config: { tension: 280, friction: 60, mass: 1 }
      },
      {
        key: 'hover',
        to: enableHover && { 
          transform: isHovered && animationsEnabled ? 'translateY(-4px)' : 'translateY(0)',
          boxShadow: isHovered && animationsEnabled 
            ? '0 12px 28px rgba(0,0,0,0.1), 0 8px 10px rgba(0,0,0,0.08)'
            : '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
          borderColor: isHovered && animationsEnabled 
            ? 'rgba(0, 102, 179, 0.4)'
            : 'rgba(0, 102, 179, 0.2)'
        },
        config: { tension: 350, friction: 18 } // Apple-like immediate response
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
  
  // Handle copy
  const handleCopy = () => {
    navigator.clipboard.writeText(result.document.page_content);
    if (animationsEnabled) {
      soundEffects.tap();
    }
    if (onCopy) {
      onCopy();
    }
  };
  
  return (
    <AnimatedPaper
      style={{
        ...animations.card,
        ...animations.hover
      }}
      variant="outlined"
      sx={{ 
        p: 3, 
        borderRadius: '12px',
        transition: 'all 0.3s ease',
        borderColor: 'rgba(0, 102, 179, 0.2)',
        willChange: 'transform, box-shadow, opacity',
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <AnimatedTypography 
          variant="h6" 
          gutterBottom
          style={animations.title}
          sx={{
            fontWeight: 600,
            color: '#0066B3',
          }}
        >
          {result.document.metadata.title || `Result ${index + 1}`}
        </AnimatedTypography>
        <AnimatedChip
          style={animations.title}
          label={`Score: ${(result.score * 100).toFixed(2)}%`}
          color="primary"
          size="small"
          sx={{ 
            background: 'linear-gradient(90deg, #0066B3, #2a8fd8)',
            fontWeight: 600,
          }}
        />
      </Box>
      
      <AnimatedDivider style={animations.content} sx={{ mb: 2 }} />
      
      <AnimatedTypography variant="body1" style={animations.content}>
        {result.document.page_content}
      </AnimatedTypography>
      
      {result.document.metadata && Object.keys(result.document.metadata).length > 0 && (
        <AnimatedBox sx={{ mt: 2 }} style={animations.metadata}>
          <Typography variant="subtitle2" gutterBottom>
            Metadata
          </Typography>
          <Grid container spacing={1}>
            {Object.entries(result.document.metadata).map(([key, value]) => (
              key !== 'title' && (
                <Grid item key={key}>
                  <Chip
                    label={`${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`}
                    variant="outlined"
                    size="small"
                    sx={{ 
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: '0 2px 8px rgba(0, 102, 179, 0.1)',
                      }
                    }}
                  />
                </Grid>
              )
            ))}
          </Grid>
        </AnimatedBox>
      )}
      
      <AnimatedBox 
        sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}
        style={animations.metadata}
      >
        <Tooltip title="Copy to clipboard">
          <IconButton 
            onClick={handleCopy}
            sx={{
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'scale(1.1)',
                color: '#0066B3',
                backgroundColor: 'rgba(0, 102, 179, 0.05)',
              }
            }}
          >
            <CopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </AnimatedBox>
    </AnimatedPaper>
  );
};

/**
 * Props for the EnhancedSearchResults component
 */
export interface EnhancedSearchResultsProps {
  /** Array of search results */
  results: SearchResultItem[];
  /** Whether animations are visible */
  animationsVisible?: boolean;
  /** Base animation delay in milliseconds */
  baseDelay?: number;
  /** Label for the results section */
  resultsLabel?: string;
  /** Whether to animate the header */
  animateHeader?: boolean;
  /** Whether to enable hover effect */
  enableHover?: boolean;
  /** Callback when a result is copied */
  onCopy?: (result: SearchResultItem) => void;
}

/**
 * Enhanced Search Results with optimized batch animations
 * Displays search results with Apple-inspired animations and interactions
 */
export const EnhancedSearchResults: React.FC<EnhancedSearchResultsProps> = ({
  results,
  animationsVisible = true,
  baseDelay = 0,
  resultsLabel = 'Search Results',
  animateHeader = true,
  enableHover = true,
  onCopy
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  
  // Header animation
  const headerAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(-10px)' },
    to: { 
      opacity: animationsVisible && animationsEnabled && animateHeader ? 1 : 0, 
      transform: animationsVisible && animationsEnabled && animateHeader ? 'translateY(0)' : 'translateY(-10px)' 
    },
    delay: baseDelay,
    config: { tension: 280, friction: 60 }
  });
  
  return (
    <AnimatedBox>
      {/* Header */}
      <AnimatedTypography 
        variant="h6" 
        gutterBottom
        style={headerAnimation}
        sx={{
          fontWeight: 600,
          background: `linear-gradient(90deg, #0066B3, #2a8fd8)`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          textFillColor: 'transparent',
          display: 'inline-block',
          mb: 2
        }}
      >
        {resultsLabel} ({results.length})
      </AnimatedTypography>
      
      {/* Results Grid */}
      <Grid container spacing={3}>
        {results.map((result, index) => (
          <Grid item xs={12} key={index}>
            <EnhancedSearchResultCard
              result={result}
              animationsVisible={animationsVisible}
              animationDelay={baseDelay + 100}
              index={index}
              enableHover={enableHover}
              onCopy={() => onCopy && onCopy(result)}
            />
          </Grid>
        ))}
      </Grid>
    </AnimatedBox>
  );
};

export default {
  SearchResultCard: EnhancedSearchResultCard,
  SearchResults: EnhancedSearchResults
};