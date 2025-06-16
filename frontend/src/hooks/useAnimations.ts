import { useState, useEffect, useCallback } from 'react';
import { useSpring, useTrail, useChain, useSpringRef, SpringConfig, Globals } from '@react-spring/web';

// Performance optimization: Set defaults that match Apple's animation style
// Higher framerates for smoother animations on capable devices
Globals.assign({
  frameLoop: 'always',
  requestAnimationFrame: typeof window !== 'undefined' 
    ? (cb) => window.requestAnimationFrame(cb) 
    : undefined
});

/**
 * Configuration options for the entrance animations
 */
interface EntranceAnimationOptions {
  /** Delay before starting animation in ms */
  delay?: number;
  /** Spring tension (default: 280) */
  tension?: number;
  /** Spring friction (default: 60) */
  friction?: number;
  /** Spring mass (default: 1) */
  mass?: number;
  /** Whether animations are enabled (default: true) */
  enabled?: boolean;
}

/**
 * A hook to control animation visibility state with auto-enable after mount
 * @param autoEnableDelay Delay in ms before auto-enabling animations (default: 200)
 * @returns Animation visibility state and setter
 */
export const useAnimationVisibility = (autoEnableDelay = 200) => {
  const [animationsVisible, setAnimationsVisible] = useState<boolean>(false);
  
  useEffect(() => {
    // Auto-enable animations after a short delay
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, autoEnableDelay);
    
    return () => clearTimeout(timer);
  }, [autoEnableDelay]);
  
  return { animationsVisible, setAnimationsVisible };
};

/**
 * A hook for creating fade up animation style
 * @param visible Whether the animation is visible
 * @param options Animation configuration options
 * @returns Spring style object
 */
export const useFadeUpAnimation = (
  visible: boolean,
  options: EntranceAnimationOptions = {}
) => {
  const {
    delay = 0,
    tension = 280,
    friction = 60,
    mass = 1,
    enabled = true,
  } = options;
  
  const config: SpringConfig = { tension, friction, mass };
  
  return useSpring({
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'translateY(0)' : 'translateY(20px)',
    },
    delay,
    config,
  });
};

/**
 * A hook for creating fade down animation style
 * @param visible Whether the animation is visible
 * @param options Animation configuration options
 * @returns Spring style object
 */
export const useFadeDownAnimation = (
  visible: boolean,
  options: EntranceAnimationOptions = {}
) => {
  const {
    delay = 0,
    tension = 280,
    friction = 60,
    mass = 1,
    enabled = true,
  } = options;
  
  const config: SpringConfig = { tension, friction, mass };
  
  return useSpring({
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'translateY(0)' : 'translateY(-20px)',
    },
    delay,
    config,
  });
};

/**
 * A hook for creating fade left animation style
 * @param visible Whether the animation is visible
 * @param options Animation configuration options
 * @returns Spring style object
 */
export const useFadeLeftAnimation = (
  visible: boolean,
  options: EntranceAnimationOptions = {}
) => {
  const {
    delay = 0,
    tension = 280,
    friction = 60,
    mass = 1,
    enabled = true,
  } = options;
  
  const config: SpringConfig = { tension, friction, mass };
  
  return useSpring({
    from: { opacity: 0, transform: 'translateX(-20px)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'translateX(0)' : 'translateX(-20px)',
    },
    delay,
    config,
  });
};

/**
 * A hook for creating fade right animation style
 * @param visible Whether the animation is visible
 * @param options Animation configuration options
 * @returns Spring style object
 */
export const useFadeRightAnimation = (
  visible: boolean,
  options: EntranceAnimationOptions = {}
) => {
  const {
    delay = 0,
    tension = 280,
    friction = 60,
    mass = 1,
    enabled = true,
  } = options;
  
  const config: SpringConfig = { tension, friction, mass };
  
  return useSpring({
    from: { opacity: 0, transform: 'translateX(20px)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'translateX(0)' : 'translateX(20px)',
    },
    delay,
    config,
  });
};

/**
 * A hook for creating scale animation style
 * @param visible Whether the animation is visible
 * @param options Animation configuration options
 * @returns Spring style object
 */
export const useScaleAnimation = (
  visible: boolean,
  options: EntranceAnimationOptions = {}
) => {
  const {
    delay = 0,
    tension = 280,
    friction = 60,
    mass = 1,
    enabled = true,
  } = options;
  
  const config: SpringConfig = { tension, friction, mass };
  
  return useSpring({
    from: { opacity: 0, transform: 'scale(0.9)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'scale(1)' : 'scale(0.9)',
    },
    delay,
    config,
  });
};

/**
 * Interface for chained animations configuration
 */
interface ChainedAnimationsConfig {
  fadeHeader?: boolean;
  fadeContent?: boolean;
  fadeFooter?: boolean;
  staggerItems?: boolean;
  itemCount?: number;
  enabled?: boolean;
}

/**
 * A hook to create chained animations with refs and sequencing
 * @param visible Whether animations are visible
 * @param config Configuration for which animations to include
 * @returns Object with animation refs and styles
 */
export const useChainedAnimations = (
  visible: boolean,
  config: ChainedAnimationsConfig
) => {
  const {
    fadeHeader = true,
    fadeContent = true,
    fadeFooter = true,
    staggerItems = false,
    itemCount = 0,
    enabled = true,
  } = config;
  
  // Animation spring refs
  const headerSpringRef = useSpringRef();
  const contentSpringRef = useSpringRef();
  const footerSpringRef = useSpringRef();
  const itemsSpringRef = useSpringRef();
  
  // Header animation
  const headerAnimation = fadeHeader ? useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'translateY(0)' : 'translateY(-20px)',
    },
    config: { tension: 280, friction: 60 }
  }) : {};
  
  // Content animation
  const contentAnimation = fadeContent ? useSpring({
    ref: contentSpringRef,
    from: { opacity: 0, transform: 'scale(0.95)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'scale(1)' : 'scale(0.95)',
    },
    config: { tension: 280, friction: 60 }
  }) : {};
  
  // Footer animation
  const footerAnimation = fadeFooter ? useSpring({
    ref: footerSpringRef,
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'translateY(0)' : 'translateY(20px)',
    },
    config: { tension: 280, friction: 60 }
  }) : {};
  
  // Staggered items animation trail
  const itemsTrail = staggerItems && itemCount > 0 ? useTrail(itemCount, {
    ref: itemsSpringRef,
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: {
      opacity: enabled && visible ? 1 : 0,
      transform: enabled && visible ? 'translateY(0)' : 'translateY(20px)',
    },
    config: { mass: 1, tension: 280, friction: 60 }
  }) : [];
  
  // Chain the animations in sequence
  useChain(
    enabled && visible
      ? [headerSpringRef, contentSpringRef, itemsSpringRef, footerSpringRef]
      : [footerSpringRef, itemsSpringRef, contentSpringRef, headerSpringRef],
    enabled && visible
      ? [0, 0.2, 0.3, 0.5]
      : [0, 0.1, 0.2, 0.3]
  );
  
  return {
    headerAnimation,
    contentAnimation,
    footerAnimation,
    itemsTrail,
    headerSpringRef,
    contentSpringRef,
    footerSpringRef,
    itemsSpringRef
  };
};

/**
 * A hook to create an animated gradient text effect
 * @returns Style object for gradient text animation
 */
export const useGradientTextAnimation = () => {
  return {
    background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundSize: '200% 100%',
    backgroundPosition: 'right bottom',
    ...useSpring({
      from: { backgroundPosition: '0% 50%' },
      to: { backgroundPosition: '100% 50%' },
      config: { duration: 3000 },
      loop: { reverse: true }
    })
  };
};

/**
 * A hook to create a shimmer effect animation
 * @param enabled Whether the animation is enabled
 * @returns Style object for shimmer effect
 */
export const useShimmerAnimation = (enabled = true) => {
  return {
    position: 'relative',
    overflow: 'hidden',
    '&::after': enabled ? {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
      transform: 'translateX(-100%)',
    } : {},
    '&:hover::after': enabled ? {
      transform: 'translateX(100%)',
      transition: 'transform 0.6s ease',
    } : {}
  };
};

/**
 * Interface for enhanced hover effect options
 */
interface EnhancedHoverOptions {
  /** Base scale value (default: 1) */
  baseScale?: number;
  /** Scale amount on hover (default: 1.03) */
  hoverScale?: number;
  /** Base shadow blur (default: 0px) */
  baseShadowBlur?: string;
  /** Hover shadow blur (default: 15px) */
  hoverShadowBlur?: string;
  /** Base shadow opacity (default: 0.1) */
  baseShadowOpacity?: number;
  /** Hover shadow opacity (default: 0.15) */
  hoverShadowOpacity?: number;
  /** Base Y offset (default: 0px) */
  baseYOffset?: string;
  /** Hover Y offset - negative = up (default: -3px) */
  hoverYOffset?: string;
  /** Spring tension (default: 350) */
  tension?: number;
  /** Spring friction (default: 18) */
  friction?: number;
  /** Color for glow/shadow effect */
  glowColor?: string;
  /** Whether the animation is enabled */
  enabled?: boolean;
}

/**
 * A hook for enhanced, Apple-like hover effects with instantaneous response
 * Uses higher tension and lower friction for immediate response like iPadOS UI
 * @param isHovered Current hover state
 * @param options Configuration options
 * @returns Spring style object
 */
export const useEnhancedHoverEffect = (
  isHovered: boolean,
  options: EnhancedHoverOptions = {}
) => {
  const {
    baseScale = 1,
    hoverScale = 1.03,
    baseShadowBlur = '0px',
    hoverShadowBlur = '15px',
    baseShadowOpacity = 0.1,
    hoverShadowOpacity = 0.15,
    baseYOffset = '0px',
    hoverYOffset = '-3px',
    tension = 350, // Higher tension for immediate response like Apple UI
    friction = 18, // Lower friction for quick settling
    glowColor = 'rgba(0, 102, 179, 1)',
    enabled = true
  } = options;
  
  // Generate the shadow values
  const baseShadow = `0 4px ${baseShadowBlur} rgba(0, 0, 0, ${baseShadowOpacity})`;
  const hoverShadow = `0 8px ${hoverShadowBlur} rgba(0, 0, 0, ${hoverShadowOpacity}), 0 0 ${hoverShadowBlur} ${glowColor.replace('1)', `${baseShadowOpacity})`)}`;
  
  // Create the spring animation with Apple-like responsiveness
  return useSpring({
    transform: `scale(${isHovered && enabled ? hoverScale : baseScale}) translateY(${isHovered && enabled ? hoverYOffset : baseYOffset})`,
    boxShadow: isHovered && enabled ? hoverShadow : baseShadow,
    config: { 
      tension, 
      friction,
      precision: 0.001 // Higher precision for smoother Apple-like transitions
    },
    immediate: !enabled
  });
};

/**
 * A hook to create a pulse animation effect
 * @returns Style object for pulse animation
 */
export const usePulseAnimation = () => {
  return {
    animation: 'pulse 2s ease-in-out infinite',
    '@keyframes pulse': {
      '0%': {
        opacity: 0.8,
        transform: 'scale(1)',
      },
      '50%': {
        opacity: 1,
        transform: 'scale(1.1)',
      },
      '100%': {
        opacity: 0.8,
        transform: 'scale(1)',
      },
    },
  };
};

/**
 * A hook to create a float animation effect
 * @returns Style object for float animation
 */
export const useFloatAnimation = () => {
  return {
    animation: 'float 3s ease-in-out infinite',
    '@keyframes float': {
      '0%': {
        transform: 'translateY(0px)',
      },
      '50%': {
        transform: 'translateY(-10px)',
      },
      '100%': {
        transform: 'translateY(0px)',
      },
    },
  };
};

/**
 * Interface for batch animation item configuration
 */
interface BatchAnimationItem {
  /** Reference key for the animation */
  key: string;
  /** Initial delay before starting animation (ms) */
  delay?: number;
  /** Custom animation configuration */
  config?: SpringConfig;
  /** Custom from values */
  from?: Record<string, any>;
  /** Custom to values */
  to?: Record<string, any>;
  /** Whether this animation is enabled */
  enabled?: boolean;
}

/**
 * A hook to create batched animations for improved performance
 * Inspired by Apple's efficient animation batching in iPadOS
 * @param visible Whether animations are visible
 * @param items Array of animation configurations
 * @returns Object with animation styles keyed by the provided keys
 */
export const useBatchAnimations = (
  visible: boolean,
  items: BatchAnimationItem[]
) => {
  // Default animation values
  const defaultFrom = { opacity: 0, transform: 'translateY(20px)' };
  const defaultTo = (enabled: boolean) => ({ 
    opacity: enabled && visible ? 1 : 0,
    transform: enabled && visible ? 'translateY(0)' : 'translateY(20px)'
  });
  const defaultConfig = { tension: 280, friction: 60, mass: 1 };
  
  // Create a single controller for all animations to batch them together
  const batchedAnimations = useCallback(() => {
    const result: Record<string, any> = {};
    
    items.forEach(item => {
      const {
        key,
        delay = 0,
        config = defaultConfig,
        from = defaultFrom,
        to,
        enabled = true
      } = item;
      
      result[key] = useSpring({
        from,
        to: to || defaultTo(enabled),
        delay,
        config,
        immediate: !enabled
      });
    });
    
    return result;
  }, [visible, items]);
  
  return batchedAnimations();
};

/**
 * Apple-inspired enhanced empty state animation
 * Creates a more engaging empty state with floating and pulsing animations
 * @param visible Whether animation is visible
 * @returns Style object for empty state animation
 */
export const useEnhancedEmptyStateAnimation = (visible: boolean = true) => {
  // Main container animation
  const containerAnimation = useSpring({
    from: { opacity: 0, transform: 'scale(0.9)' },
    to: { opacity: visible ? 1 : 0, transform: visible ? 'scale(1)' : 'scale(0.9)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Icon animation - combines float and pulse
  const iconAnimation = useSpring({
    from: { opacity: 0.7, transform: 'translateY(0px) scale(1)' },
    to: async (next) => {
      while (visible) {
        // Float up while scaling up slightly
        await next({ opacity: 1, transform: 'translateY(-15px) scale(1.05)', config: { tension: 120, friction: 14 } });
        // Float down while scaling down slightly
        await next({ opacity: 0.8, transform: 'translateY(0px) scale(0.98)', config: { tension: 120, friction: 14 } });
      }
    },
    config: { tension: 120, friction: 14 }
  });
  
  // Text animation with slight delay
  const textAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: visible ? 1 : 0, transform: visible ? 'translateY(0)' : 'translateY(20px)' },
    delay: 300,
    config: { tension: 280, friction: 60 }
  });
  
  return { containerAnimation, iconAnimation, textAnimation };
};