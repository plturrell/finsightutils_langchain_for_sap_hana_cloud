import React, { useState, useEffect } from 'react';
import { animated } from '@react-spring/web';
import { useAnimationContext } from '../context/AnimationContext';
import { 
  useFadeUpAnimation, 
  useFadeDownAnimation,
  useFadeLeftAnimation,
  useFadeRightAnimation,
  useScaleAnimation,
  useEnhancedHoverEffect,
  useGradientTextAnimation
} from '../hooks/useAnimations';
import { soundEffects } from '../utils/soundEffects';

/**
 * Available animation types for the withAnimation HOC
 */
export type AnimationType = 
  | 'fadeUp' 
  | 'fadeDown' 
  | 'fadeLeft' 
  | 'fadeRight' 
  | 'scale' 
  | 'none';

/**
 * Configuration options for the withAnimation HOC
 */
export interface WithAnimationOptions {
  /** The type of entrance animation to apply */
  animationType?: AnimationType;
  /** Delay before starting animation in ms */
  delay?: number;
  /** Whether to enable hover animations */
  enableHover?: boolean;
  /** Whether to enable sound feedback */
  enableSound?: boolean;
  /** The type of sound to play on interaction */
  soundType?: 'tap' | 'hover' | 'switch' | 'success' | 'error' | 'transition';
  /** Whether to add a gradient text effect */
  gradientText?: boolean;
  /** Spring tension (default: 280) */
  tension?: number;
  /** Spring friction (default: 60) */
  friction?: number;
  /** Spring mass (default: 1) */
  mass?: number;
}

/**
 * Higher-Order Component that adds animations to any component
 * @param WrappedComponent The component to enhance with animations
 * @param options Animation configuration options
 * @returns Enhanced component with animations
 */
export function withAnimation<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: WithAnimationOptions = {}
): React.FC<P> {
  // Set default options
  const {
    animationType = 'fadeUp',
    delay = 0,
    enableHover = false,
    enableSound = false,
    soundType = 'tap',
    gradientText = false,
    tension = 280,
    friction = 60,
    mass = 1,
  } = options;
  
  // Create the enhanced component
  const AnimatedComponent: React.FC<P> = (props) => {
    const { animationsEnabled } = useAnimationContext();
    const [visible, setVisible] = useState(false);
    const [isHovered, setIsHovered] = useState(false);
    
    // Create animated version of the wrapped component
    const AnimatedWrapped = animated(WrappedComponent);
    
    // Handle entrance animation
    useEffect(() => {
      const timer = setTimeout(() => {
        setVisible(true);
      }, 100);
      
      return () => clearTimeout(timer);
    }, []);
    
    // Determine which animation to use
    let animationStyle: any = {};
    
    // Animation configuration options
    const animationOptions = {
      delay,
      tension,
      friction,
      mass,
      enabled: animationsEnabled
    };
    
    // Choose the correct animation based on type
    switch (animationType) {
      case 'fadeUp':
        animationStyle = useFadeUpAnimation(visible, animationOptions);
        break;
      case 'fadeDown':
        animationStyle = useFadeDownAnimation(visible, animationOptions);
        break;
      case 'fadeLeft':
        animationStyle = useFadeLeftAnimation(visible, animationOptions);
        break;
      case 'fadeRight':
        animationStyle = useFadeRightAnimation(visible, animationOptions);
        break;
      case 'scale':
        animationStyle = useScaleAnimation(visible, animationOptions);
        break;
      case 'none':
      default:
        // No animation
        animationStyle = {};
        break;
    }
    
    // Add hover effect if enabled
    const hoverStyle = enableHover 
      ? useEnhancedHoverEffect(isHovered, {
          enabled: animationsEnabled,
          tension: 350, // Higher tension for faster response
          friction: 18, // Lower friction for quicker settling
        })
      : {};
    
    // Add gradient text effect if enabled
    const textStyle = gradientText ? useGradientTextAnimation() : {};
    
    // Merge all styles
    const combinedStyle = {
      ...animationStyle,
      ...(enableHover ? hoverStyle : {}),
      ...(gradientText ? textStyle : {})
    };
    
    // Handle sound effects
    const handleMouseEnter = () => {
      setIsHovered(true);
      const { soundEffectsEnabled } = useAnimationContext();
      if (enableSound && animationsEnabled && soundEffectsEnabled && soundType === 'hover') {
        soundEffects.hover();
      }
    };
    
    const handleMouseLeave = () => {
      setIsHovered(false);
    };
    
    const handleClick = (e: any) => {
      const { soundEffectsEnabled } = useAnimationContext();
      if (enableSound && animationsEnabled && soundEffectsEnabled) {
        switch (soundType) {
          case 'tap':
            soundEffects.tap();
            break;
          case 'switch':
            soundEffects.switch();
            break;
          case 'success':
            soundEffects.success();
            break;
          case 'error':
            soundEffects.error();
            break;
          case 'transition':
            soundEffects.transition();
            break;
          default:
            soundEffects.tap();
            break;
        }
      }
      
      // Call the original onClick if it exists
      if (props.onClick) {
        props.onClick(e);
      }
    };
    
    // Enhanced props
    const enhancedProps = {
      ...props,
      style: combinedStyle,
      onMouseEnter: handleMouseEnter,
      onMouseLeave: handleMouseLeave,
      onClick: handleClick,
    };
    
    return <AnimatedWrapped {...enhancedProps as P} />;
  };
  
  // Set display name for debugging
  const displayName = WrappedComponent.displayName || WrappedComponent.name || 'Component';
  AnimatedComponent.displayName = `withAnimation(${displayName})`;
  
  return AnimatedComponent;
}

/**
 * Shorthand HOCs for common animation patterns
 */

/**
 * HOC that adds fade up animation to a component
 */
export function withFadeUp<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: Omit<WithAnimationOptions, 'animationType'> = {}
): React.FC<P> {
  return withAnimation(WrappedComponent, { ...options, animationType: 'fadeUp' });
}

/**
 * HOC that adds fade down animation to a component
 */
export function withFadeDown<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: Omit<WithAnimationOptions, 'animationType'> = {}
): React.FC<P> {
  return withAnimation(WrappedComponent, { ...options, animationType: 'fadeDown' });
}

/**
 * HOC that adds hover effects to a component
 */
export function withHoverEffect<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: Omit<WithAnimationOptions, 'enableHover'> = {}
): React.FC<P> {
  return withAnimation(WrappedComponent, { ...options, enableHover: true });
}

/**
 * HOC that adds sound feedback to a component
 */
export function withSoundFeedback<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  soundType: 'tap' | 'hover' | 'switch' | 'success' | 'error' = 'tap',
  options: Omit<WithAnimationOptions, 'enableSound' | 'soundType'> = {}
): React.FC<P> {
  return withAnimation(WrappedComponent, { 
    ...options, 
    enableSound: true,
    soundType
  });
}

/**
 * HOC that adds gradient text effect to a component
 */
export function withGradientText<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: Omit<WithAnimationOptions, 'gradientText'> = {}
): React.FC<P> {
  return withAnimation(WrappedComponent, { ...options, gradientText: true });
}

/**
 * HOC that adds complete Apple-like interaction to a component
 * Combines all effects: animation, hover, sound feedback
 */
export function withCompleteAnimation<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: WithAnimationOptions = {}
): React.FC<P> {
  return withAnimation(WrappedComponent, { 
    animationType: 'fadeUp',
    enableHover: true,
    enableSound: true,
    soundType: 'tap',
    tension: 350,
    friction: 18,
    ...options
  });
}