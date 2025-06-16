# Animation Patterns Guide

This document outlines the animation patterns and best practices implemented in the Langchain Integration UI for SAP HANA Cloud.

## Table of Contents
- [Overview](#overview)
- [Animation Hooks](#animation-hooks)
- [Performance Considerations](#performance-considerations)
- [Common Patterns](#common-patterns)
- [Enhanced Form Inputs](#enhanced-form-inputs)
- [Testing Animations](#testing-animations)
- [Accessibility](#accessibility)

## Overview

The animation system in our application is built on [React Spring](https://react-spring.dev/), which provides physics-based animations that feel natural and responsive. We've created a set of reusable hooks and patterns that make it easy to add consistent animations throughout the application while maintaining good performance.

### Key Features

- **Physics-based animations**: Natural feeling with configurable tension, friction, and mass
- **Performance optimized**: Only animates transform and opacity properties when possible
- **Accessibility support**: Respects user preferences for reduced motion
- **Toggle mechanism**: Users can disable animations for performance or preference
- **Consistent patterns**: Standard animations for common UI interactions
- **Apple-inspired design**: Follows iPadOS design patterns with immediate response and subtle feedback

## Animation Hooks

We provide several custom hooks for common animation patterns:

### Basic Animation Hooks

- `useAnimationVisibility`: Controls the animation visibility state with auto-enable after mount
- `useFadeUpAnimation`: Creates a fade-in animation that moves upward
- `useFadeDownAnimation`: Creates a fade-in animation that moves downward
- `useFadeLeftAnimation`: Creates a fade-in animation that moves from left
- `useFadeRightAnimation`: Creates a fade-in animation that moves from right
- `useScaleAnimation`: Creates a scale-in animation
- `useGradientTextAnimation`: Creates an animated gradient text effect
- `useShimmerAnimation`: Creates a shimmer effect animation
- `usePulseAnimation`: Creates a pulse animation effect
- `useFloatAnimation`: Creates a floating up and down animation effect

### Advanced Animation Hooks

- `useChainedAnimations`: Creates a sequence of animations with configurable order
- `useAnimationContext`: Access the application-wide animation settings

## Performance Considerations

To ensure good performance, especially on lower-powered devices:

1. **Limit animated elements**: Don't animate too many elements simultaneously
2. **Use hardware-accelerated properties**: Prefer animating `transform` and `opacity`
3. **Apply animation toggle**: Allow users to disable animations
4. **Test on lower-end devices**: Ensure animations don't cause performance issues

## Common Patterns

### Page Transitions

For page transitions, we typically use a sequence of animations:

```tsx
// Import necessary hooks
import { useAnimationVisibility, useFadeUpAnimation } from '../hooks/useAnimations';

const MyComponent: React.FC = () => {
  // Animation state
  const { animationsVisible } = useAnimationVisibility();
  
  // Header animation
  const headerAnimation = useFadeUpAnimation(animationsVisible, { delay: 100 });
  
  return (
    <AnimatedBox style={headerAnimation}>
      <Typography variant="h4">My Page Title</Typography>
    </AnimatedBox>
  );
};
```

### Staggered List Animations

For lists, we use staggered animations:

```tsx
// Import necessary hooks
import { useAnimationVisibility } from '../hooks/useAnimations';
import { useTrail } from '@react-spring/web';

const MyListComponent: React.FC<{ items: string[] }> = ({ items }) => {
  // Animation state
  const { animationsVisible } = useAnimationVisibility();
  
  // Create a trail for the list items
  const trail = useTrail(items.length, {
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { 
      opacity: animationsVisible ? 1 : 0, 
      transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)'
    },
    config: { mass: 1, tension: 280, friction: 60 }
  });
  
  return (
    <List>
      {trail.map((style, index) => (
        <animated.li key={index} style={style}>
          {items[index]}
        </animated.li>
      ))}
    </List>
  );
};
```

### Button Animations

For interactive elements like buttons:

```tsx
const MyButton: React.FC = () => {
  // Animation on hover
  const [isHovered, setIsHovered] = useState(false);
  
  const buttonSpring = useSpring({
    transform: isHovered ? 'translateY(-2px)' : 'translateY(0)',
    boxShadow: isHovered 
      ? '0 6px 20px rgba(0, 102, 179, 0.3)' 
      : '0 2px 5px rgba(0, 102, 179, 0.1)',
    config: { tension: 300, friction: 20 }
  });
  
  return (
    <AnimatedButton
      style={buttonSpring}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      Click Me
    </AnimatedButton>
  );
};
```

## Testing Animations

We've created utilities to test animations, located in `src/tests/animation-test-utils.tsx`:

- `renderWithAnimationProviders`: Renders a component with all necessary providers
- `createAnimatedTestComponent`: Creates a component for testing animations
- `waitForAnimation`: Waits for an animation to complete
- `testAnimationSequence`: Tests if animations complete in the correct order
- `testStaggeredAnimations`: Tests staggered animations of list items

Example test:

```tsx
import { renderWithAnimationProviders, waitForAnimation } from './animation-test-utils';

test('Component animates correctly', async () => {
  renderWithAnimationProviders(<MyAnimatedComponent />);
  
  // Trigger the animation
  fireEvent.click(screen.getByTestId('trigger-button'));
  
  // Wait for the animation to complete
  await waitForAnimation('animated-element', 'opacity', '1');
  
  // Additional assertions
  expect(screen.getByTestId('animated-element')).toHaveStyle('transform: translateY(0)');
});
```

## Accessibility

Our animation system respects user preferences for reduced motion:

1. **Animation toggle**: Users can disable animations via the AnimationToggle component
2. **Persistence**: Animation preference is saved to localStorage
3. **Prefers-reduced-motion**: The system detects and respects the OS-level reduced motion setting

To ensure your animations are accessible:

- Ensure all animated content is still accessible when animations are disabled
- Don't rely solely on animation to convey important information
- Keep animations subtle and non-distracting
- Test with screen readers and keyboard navigation

## Enhanced Form Inputs

We've created a set of enhanced form input components that follow Apple's iPadOS design language with smooth animations, immediate feedback, and subtle sound effects.

### Form Input Animation Design Principles

1. **Immediate Response**: Higher tension (350) and lower friction (18) for instant animation start
2. **Subtle Feedback**: Visual and audio feedback is minimalist and unobtrusive
3. **Coherent Motion**: All form elements follow consistent animation patterns
4. **Progressive Disclosure**: Information and options are revealed progressively
5. **Clear State Indication**: Current state is always visually communicated

### Form Input Animation Types

#### Entrance Animations

All enhanced input components use entrance animations when they mount:

```tsx
// Example entrance animation in EnhancedTextField
const entranceAnimation = useSpring({
  from: { opacity: 0, transform: 'translateY(10px)' },
  to: { 
    opacity: visible && animationsEnabled ? 1 : 0,
    transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
  },
  config: { tension: 280, friction: 60 },
  immediate: !animationsEnabled
});
```

#### Focus/Hover Animations

Form components respond to focus and hover with subtle but immediate feedback:

```tsx
// Example focus animation in EnhancedTextField
const focusAnimation = useSpring({
  transform: isFocused ? 'scale(1.01)' : 'scale(1)',
  boxShadow: isFocused 
    ? `0 0 0 2px ${alpha(theme.palette.primary.main, 0.25)}`
    : 'none',
  config: { 
    tension: 350, // Higher tension for immediate response
    friction: 18,  // Lower friction for quicker settling
    precision: 0.001 // Higher precision for smoother transitions
  }
});
```

#### State Change Animations

Form inputs animate smoothly between states:

```tsx
// Example checkbox animation
const checkStyle = useSpring({
  transform: props.checked ? 'scale(1)' : 'scale(0)',
  opacity: props.checked ? 1 : 0,
  config: { tension: 350, friction: 18 }
});
```

### Sound Feedback

Form elements have subtle sound feedback following Apple's design patterns:

```tsx
// Example sound trigger
const handleChange = () => {
  if (animationsEnabled) {
    soundEffects.switch();
  }
  // Additional change handling
};
```

### Available Enhanced Form Components

- `EnhancedTextField`: Text input with animations and sound feedback
- `EnhancedSelect`: Dropdown select with animated options
- `EnhancedCheckbox`: Animated checkbox with sound feedback
- `EnhancedRadio`: Animated radio button with sound feedback
- `EnhancedSwitch`: Toggle switch with Apple-like animations
- `EnhancedSlider`: Slider with immediate response animations
- `EnhancedFormControlLabel`: Animated form control labels
- `EnhancedAutocomplete`: Enhanced autocomplete with animations
- `EnhancedInputGroup`: Grouping component for form fields

See the `EnhancedFormExample` component for a demonstration of all enhanced form inputs.

## Using the Animation Context

To properly respect user preferences:

```tsx
import { useAnimationContext } from '../context/AnimationContext';

const MyComponent: React.FC = () => {
  const { animationsEnabled } = useAnimationContext();
  
  const animationStyle = useSpring({
    opacity: animationsEnabled ? 1 : 0,
    transform: animationsEnabled ? 'translateY(0)' : 'translateY(20px)',
    // Only perform the animation if enabled
    immediate: !animationsEnabled
  });
  
  return <animated.div style={animationStyle}>Content</animated.div>;
};
```

## Further Reading

- [React Spring Documentation](https://react-spring.dev/)
- [Animation Performance](https://web.dev/animations-guide/)
- [Designing with Animation](https://material.io/design/motion/understanding-motion.html)
- [Apple Human Interface Guidelines - Animation](https://developer.apple.com/design/human-interface-guidelines/animation)