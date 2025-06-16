import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { animated, useTransition } from '@react-spring/web';
import { Box } from '@mui/material';
import { useAnimationContext } from '../context/AnimationContext';

interface PageTransitionProps {
  children: React.ReactNode;
}

/**
 * Component for smooth transitions between pages
 * Inspired by Apple's iPadOS transitions - smooth, subtle cross-fades with slight scaling
 */
const PageTransition: React.FC<PageTransitionProps> = ({ children }) => {
  const location = useLocation();
  const { animationsEnabled } = useAnimationContext();
  
  // Apple-like smooth transitions with subtle fade and scale
  const transitions = useTransition(location, {
    from: { 
      opacity: 0, 
      transform: 'scale(1.01)', 
      filter: 'blur(3px)',
      position: 'absolute',
      width: '100%',
      height: '100%',
    },
    enter: { 
      opacity: 1, 
      transform: 'scale(1)', 
      filter: 'blur(0px)',
      position: 'relative',
      width: '100%',
      height: '100%',
    },
    leave: { 
      opacity: 0, 
      transform: 'scale(0.99)', 
      filter: 'blur(3px)',
      position: 'absolute',
      width: '100%',
      height: '100%',
    },
    config: { 
      tension: 280, 
      friction: 60,
      precision: 0.001, // Higher precision for smoother Apple-like transitions
    },
    immediate: !animationsEnabled,
  });
  
  // Play subtle sound on page transition (if enabled)
  useEffect(() => {
    if (animationsEnabled) {
      try {
        // Apple-like subtle transition sound
        // Use a very subtle sound similar to iPadOS navigation sounds
        const audio = new Audio('/assets/sounds/transition.mp3');
        audio.volume = 0.1; // Very subtle
        audio.play().catch(e => {
          // Silently fail if audio can't be played (common in browsers requiring user interaction first)
          console.debug('Transition sound not played:', e);
        });
      } catch (e) {
        // Ignore audio errors
      }
    }
  }, [location.pathname, animationsEnabled]);
  
  return (
    <Box sx={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
      {transitions((style, item) => (
        <animated.div style={style} className="page-transition">
          {children}
        </animated.div>
      ))}
    </Box>
  );
};

export default PageTransition;