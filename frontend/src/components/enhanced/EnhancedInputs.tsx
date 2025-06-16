import React, { useState, useEffect, useRef } from 'react';
import {
  TextField,
  TextFieldProps,
  InputAdornment,
  IconButton,
  Select,
  SelectProps,
  MenuItem,
  FormControl,
  FormControlLabel,
  FormControlLabelProps,
  InputLabel,
  Checkbox,
  CheckboxProps,
  Radio,
  RadioProps,
  Switch,
  SwitchProps,
  Slider,
  SliderProps,
  Autocomplete,
  AutocompleteProps,
  FormHelperText,
  useTheme,
  alpha,
  Box,
  Grow,
  Fade,
  Typography
} from '@mui/material';
import {
  Check as CheckIcon,
  Clear as ClearIcon,
  VisibilityOff as VisibilityOffIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';
import { animated } from '@react-spring/web';
import { 
  useAnimationContext,
  useFadeUpAnimation,
  useScaleAnimation,
  withAnimation,
  withSoundFeedback,
  soundEffects
} from '@finsightdev/ui-animations';

// Create animated versions of MUI components using the shared package
const AnimatedTextField = animated(TextField);
const AnimatedSelect = animated(Select);
const AnimatedFormControl = animated(FormControl);
const AnimatedFormControlLabel = animated(FormControlLabel);

// Enhanced components with sound feedback
const SoundCheckbox = withSoundFeedback(Checkbox, 'switch');
const SoundRadio = withSoundFeedback(Radio, 'tap');
const SoundSwitch = withSoundFeedback(Switch, 'switch');
const SoundSlider = withSoundFeedback(Slider, 'tap');

// Animated components with both animations and sound
const AnimatedCheckbox = withAnimation(SoundCheckbox, { animationType: 'scale' });
const AnimatedRadio = withAnimation(SoundRadio, { animationType: 'scale' });
const AnimatedSwitch = withAnimation(SoundSwitch, { animationType: 'scale' });
const AnimatedSlider = withAnimation(SoundSlider, { animationType: 'scale' });
const AnimatedAutocomplete = animated(Autocomplete) as any; // Type casting to handle generics
const AnimatedFormHelperText = animated(FormHelperText);
const AnimatedInputLabel = animated(InputLabel);

/**
 * Enhanced TextField with Apple-like animations and interactions
 */
export const EnhancedTextField: React.FC<TextFieldProps & {
  clearable?: boolean;
  animationDelay?: number;
}> = ({
  clearable = false,
  animationDelay = 0,
  ...props
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isFocused, setIsFocused] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [visible, setVisible] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Set visible after mount with delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100 + animationDelay);
    
    return () => clearTimeout(timer);
  }, [animationDelay]);
  
  // Entrance animation
  const entranceAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
    },
    config: { tension: 280, friction: 60 },
    immediate: !animationsEnabled
  });
  
  // Focus animation (more pronounced on iPadOS)
  const focusAnimation = useSpring({
    transform: isFocused && animationsEnabled ? 'scale(1.01)' : 'scale(1)',
    boxShadow: isFocused && animationsEnabled 
      ? `0 0 0 2px ${alpha(theme.palette.primary.main, 0.25)}`
      : 'none',
    config: { 
      tension: 350, // Higher tension for immediate response
      friction: 18,  // Lower friction for quicker settling
      precision: 0.001 // Higher precision for smoother transitions
    },
    immediate: !animationsEnabled
  });
  
  // Hover animation (subtle effect)
  const hoverAnimation = useSpring({
    backgroundColor: isHovered && !isFocused && animationsEnabled
      ? alpha(theme.palette.primary.main, 0.02)
      : 'transparent',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Combined animation styles
  const combinedStyle = {
    ...entranceAnimation,
    ...focusAnimation,
    ...hoverAnimation,
  };
  
  // Handle focus events
  const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
    setIsFocused(true);
    if (animationsEnabled) {
      soundEffects.tap();
    }
    if (props.onFocus) {
      props.onFocus(e);
    }
  };
  
  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    setIsFocused(false);
    if (props.onBlur) {
      props.onBlur(e);
    }
  };
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  // Handle clear button click
  const handleClear = () => {
    if (props.onChange) {
      // Create a fake event to simulate input clearing
      const event = {
        target: { value: '' }
      } as React.ChangeEvent<HTMLInputElement>;
      props.onChange(event);
    }
    
    // Focus the input after clearing
    if (inputRef.current) {
      inputRef.current.focus();
    }
    
    if (animationsEnabled) {
      soundEffects.tap();
    }
  };
  
  // Toggle password visibility
  const handleTogglePasswordVisibility = () => {
    setShowPassword(!showPassword);
    if (animationsEnabled) {
      soundEffects.switch();
    }
  };
  
  // Determine if input has value
  const hasValue = props.value !== undefined && props.value !== '';
  
  // Determine if we should show the clear button
  const showClearButton = clearable && hasValue && (isFocused || isHovered);
  
  // Determine if this is a password field
  const isPassword = props.type === 'password';
  
  // End adornment with clear button and/or password toggle
  const endAdornment = (
    <InputAdornment position="end">
      {showClearButton && (
        <Fade in={showClearButton}>
          <IconButton 
            size="small" 
            onClick={handleClear}
            edge="end"
            tabIndex={-1}
            aria-label="clear text"
            sx={{ 
              transition: 'all 0.15s ease-in-out',
              transform: 'scale(0.8)',
            }}
          >
            <ClearIcon fontSize="small" />
          </IconButton>
        </Fade>
      )}
      
      {isPassword && (
        <IconButton
          size="small"
          onClick={handleTogglePasswordVisibility}
          edge="end"
          tabIndex={-1}
          aria-label={showPassword ? 'hide password' : 'show password'}
          sx={{ transition: 'all 0.15s ease-in-out' }}
        >
          {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
        </IconButton>
      )}
      
      {props.InputProps?.endAdornment}
    </InputAdornment>
  );
  
  // Apply valid state styling
  const validationAnimation = props.error
    ? { 
        transform: 'translateX(0)',
        animation: 'shake 0.5s cubic-bezier(0.36, 0.07, 0.19, 0.97) both',
        '@keyframes shake': {
          '0%, 100%': { transform: 'translateX(0)' },
          '10%, 30%, 50%, 70%, 90%': { transform: 'translateX(-2px)' },
          '20%, 40%, 60%, 80%': { transform: 'translateX(2px)' }
        }
      }
    : {};
  
  // Apply success state styling when valid and not empty
  const successState = hasValue && !props.error && !isFocused;
  
  const successAnimation = successState
    ? {
        '& .MuiOutlinedInput-root': {
          '& fieldset': {
            borderColor: theme.palette.success.main,
          },
          '&:hover fieldset': {
            borderColor: theme.palette.success.main,
          },
        }
      }
    : {};
  
  return (
    <AnimatedTextField
      {...props}
      style={combinedStyle}
      onFocus={handleFocus}
      onBlur={handleBlur}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      inputRef={inputRef}
      type={isPassword ? (showPassword ? 'text' : 'password') : props.type}
      InputProps={{
        ...props.InputProps,
        endAdornment: endAdornment,
      }}
      sx={{
        // Base styles
        position: 'relative',
        transition: 'all 0.2s ease-in-out',
        transform: 'translateZ(0)',
        willChange: 'transform, opacity, box-shadow',
        
        // Focus state styles
        '& .MuiOutlinedInput-root.Mui-focused': {
          '& fieldset': {
            transition: 'all 0.2s ease-in-out',
            borderWidth: '2px',
          }
        },
        
        // Animation styles
        ...validationAnimation,
        ...successAnimation,
        
        // Add success indicator
        ...(successState && {
          '& .MuiOutlinedInput-root': {
            '&::after': {
              content: '""',
              position: 'absolute',
              right: 12,
              top: 'calc(50% - 8px)',
              width: 16,
              height: 16,
              borderRadius: '50%',
              backgroundColor: theme.palette.success.main,
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/%3E%3C/svg%3E")`,
              backgroundSize: 'cover',
              transform: 'scale(0.8)',
              zIndex: 1,
              transition: 'all 0.2s ease-in-out',
            }
          }
        }),
        
        // User styles
        ...props.sx
      }}
    />
  );
};

/**
 * Enhanced Select with Apple-like animations and interactions
 */
export const EnhancedSelect = React.forwardRef<HTMLInputElement, SelectProps & {
  label?: string;
  animationDelay?: number;
}>((props, ref) => {
  const { 
    label, 
    animationDelay = 0,
    ...selectProps 
  } = props;
  
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isFocused, setIsFocused] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [visible, setVisible] = useState(false);
  
  // Set visible after mount with delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100 + animationDelay);
    
    return () => clearTimeout(timer);
  }, [animationDelay]);
  
  // Entrance animation
  const entranceAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
    },
    config: { tension: 280, friction: 60 },
    immediate: !animationsEnabled
  });
  
  // Focus animation
  const focusAnimation = useSpring({
    transform: isFocused && animationsEnabled ? 'scale(1.01)' : 'scale(1)',
    boxShadow: isFocused && animationsEnabled 
      ? `0 0 0 2px ${alpha(theme.palette.primary.main, 0.25)}`
      : 'none',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Hover animation
  const hoverAnimation = useSpring({
    backgroundColor: isHovered && !isFocused && animationsEnabled
      ? alpha(theme.palette.primary.main, 0.02)
      : 'transparent',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Combined animation styles
  const combinedStyle = {
    ...entranceAnimation,
    ...focusAnimation,
    ...hoverAnimation,
  };
  
  // Handle focus events
  const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
    setIsFocused(true);
    if (animationsEnabled) {
      soundEffects.tap();
    }
    if (selectProps.onFocus) {
      selectProps.onFocus(e);
    }
  };
  
  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    setIsFocused(false);
    if (selectProps.onBlur) {
      selectProps.onBlur(e);
    }
  };
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  // Handle open/close events
  const handleOpen = () => {
    if (animationsEnabled) {
      soundEffects.tap();
    }
    if (selectProps.onOpen) {
      selectProps.onOpen();
    }
  };
  
  // Handle selection change
  const handleChange = (e: any) => {
    if (animationsEnabled) {
      soundEffects.switch();
    }
    if (selectProps.onChange) {
      selectProps.onChange(e, null);
    }
  };
  
  return (
    <AnimatedFormControl 
      variant={selectProps.variant || "outlined"} 
      fullWidth={selectProps.fullWidth}
      style={entranceAnimation}
      error={selectProps.error}
      disabled={selectProps.disabled}
      sx={{ ...props.sx }}
    >
      {label && (
        <AnimatedInputLabel 
          id={`${selectProps.id || 'select'}-label`}
          style={entranceAnimation}
        >
          {label}
        </AnimatedInputLabel>
      )}
      
      <AnimatedSelect
        {...selectProps}
        labelId={`${selectProps.id || 'select'}-label`}
        style={combinedStyle}
        onFocus={handleFocus}
        onBlur={handleBlur}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onOpen={handleOpen}
        onChange={handleChange}
        ref={ref}
        MenuProps={{
          ...selectProps.MenuProps,
          PaperProps: {
            ...selectProps.MenuProps?.PaperProps,
            sx: {
              mt: 1,
              borderRadius: '8px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
              overflow: 'hidden',
              ...selectProps.MenuProps?.PaperProps?.sx
            }
          },
          TransitionComponent: Grow,
          TransitionProps: {
            timeout: 200,
            ...selectProps.MenuProps?.TransitionProps
          }
        }}
        sx={{
          '&.MuiOutlinedInput-root': {
            transition: 'all 0.2s ease-in-out',
            borderRadius: '8px',
            
            '&.Mui-focused': {
              '& .MuiOutlinedInput-notchedOutline': {
                borderWidth: '2px',
              }
            },
          },
          ...selectProps.sx
        }}
      />
      
      {selectProps.helperText && (
        <AnimatedFormHelperText
          style={useSpring({
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(-5px)',
            config: { tension: 280, friction: 60 },
            delay: 150 + animationDelay,
            immediate: !animationsEnabled
          })}
          error={selectProps.error}
        >
          {selectProps.helperText}
        </AnimatedFormHelperText>
      )}
    </AnimatedFormControl>
  );
});

/**
 * Enhanced Checkbox with Apple-like animations and interactions
 */
export const EnhancedCheckbox: React.FC<CheckboxProps> = (props) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Hover animation
  const hoverStyle = useSpring({
    transform: isHovered && animationsEnabled ? 'scale(1.1)' : 'scale(1)',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Check animation
  const checkStyle = useSpring({
    transform: props.checked && animationsEnabled ? 'scale(1)' : 'scale(0)',
    opacity: props.checked && animationsEnabled ? 1 : 0,
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  // Handle change event
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (animationsEnabled) {
      soundEffects.switch();
    }
    if (props.onChange) {
      props.onChange(e, e.target.checked);
    }
  };
  
  return (
    <AnimatedCheckbox
      {...props}
      style={hoverStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onChange={handleChange}
      icon={
        <Box
          sx={{
            width: 18,
            height: 18,
            borderRadius: 1,
            border: `2px solid ${alpha(theme.palette.text.primary, 0.3)}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease-in-out',
          }}
        />
      }
      checkedIcon={
        <Box
          sx={{
            width: 18,
            height: 18,
            borderRadius: 1,
            border: 'none',
            backgroundColor: theme.palette.primary.main,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease-in-out',
          }}
        >
          <animated.div style={checkStyle}>
            <CheckIcon sx={{ fontSize: 14, color: 'white' }} />
          </animated.div>
        </Box>
      }
      sx={{
        padding: 1,
        ...props.sx
      }}
    />
  );
};

/**
 * Enhanced Radio button with Apple-like animations and interactions
 */
export const EnhancedRadio: React.FC<RadioProps> = (props) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Hover animation
  const hoverStyle = useSpring({
    transform: isHovered && animationsEnabled ? 'scale(1.1)' : 'scale(1)',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Selected animation
  const selectedStyle = useSpring({
    transform: props.checked && animationsEnabled ? 'scale(1)' : 'scale(0)',
    opacity: props.checked && animationsEnabled ? 1 : 0,
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  // Handle change event
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (animationsEnabled) {
      soundEffects.switch();
    }
    if (props.onChange) {
      props.onChange(e, e.target.checked);
    }
  };
  
  return (
    <AnimatedRadio
      {...props}
      style={hoverStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onChange={handleChange}
      icon={
        <Box
          sx={{
            width: 18,
            height: 18,
            borderRadius: '50%',
            border: `2px solid ${alpha(theme.palette.text.primary, 0.3)}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease-in-out',
          }}
        />
      }
      checkedIcon={
        <Box
          sx={{
            width: 18,
            height: 18,
            borderRadius: '50%',
            border: `2px solid ${theme.palette.primary.main}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease-in-out',
          }}
        >
          <animated.div 
            style={selectedStyle}
            className="inner-circle"
          >
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                backgroundColor: theme.palette.primary.main,
              }}
            />
          </animated.div>
        </Box>
      }
      sx={{
        padding: 1,
        ...props.sx
      }}
    />
  );
};

/**
 * Enhanced Switch with Apple-like animations and interactions
 */
export const EnhancedSwitch: React.FC<SwitchProps> = (props) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isHovered, setIsHovered] = useState(false);
  
  // Hover animation
  const hoverStyle = useSpring({
    transform: isHovered && animationsEnabled ? 'scale(1.05)' : 'scale(1)',
    boxShadow: isHovered && animationsEnabled 
      ? `0 2px 8px ${alpha(theme.palette.primary.main, 0.3)}`
      : 'none',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  // Handle change event
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (animationsEnabled) {
      soundEffects.switch();
    }
    if (props.onChange) {
      props.onChange(e, e.target.checked);
    }
  };
  
  return (
    <AnimatedSwitch
      {...props}
      style={hoverStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onChange={handleChange}
      sx={{
        '& .MuiSwitch-switchBase': {
          transition: 'transform 0.15s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        '& .MuiSwitch-thumb': {
          transition: 'width 0.15s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.2)',
        },
        '& .MuiSwitch-track': {
          borderRadius: 16,
          opacity: 1,
          backgroundColor: props.checked 
            ? theme.palette.primary.main 
            : alpha(theme.palette.text.primary, 0.3),
          transition: 'background-color 0.15s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        ...props.sx
      }}
    />
  );
};

/**
 * Enhanced Slider with Apple-like animations and interactions
 */
export const EnhancedSlider: React.FC<SliderProps & {
  animationDelay?: number;
}> = ({
  animationDelay = 0,
  ...props
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isActive, setIsActive] = useState(false);
  const [visible, setVisible] = useState(false);
  
  // Set visible after mount with delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100 + animationDelay);
    
    return () => clearTimeout(timer);
  }, [animationDelay]);
  
  // Entrance animation
  const entranceAnimation = useSpring({
    from: { opacity: 0, transform: 'scale(0.95)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'scale(1)' : 'scale(0.95)'
    },
    config: { tension: 280, friction: 60 },
    immediate: !animationsEnabled
  });
  
  // Handle active state
  const handleMouseDown = () => {
    setIsActive(true);
    if (animationsEnabled) {
      soundEffects.tap();
    }
  };
  
  const handleMouseUp = () => {
    setIsActive(false);
  };
  
  // Handle change
  const handleChange = (event: Event, value: number | number[]) => {
    if (animationsEnabled && !isActive) {
      soundEffects.tap();
    }
    if (props.onChange) {
      props.onChange(event, value);
    }
  };
  
  return (
    <Box
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onTouchStart={handleMouseDown}
      onTouchEnd={handleMouseUp}
      sx={{ 
        padding: '10px 0',
        ...entranceAnimation
      }}
    >
      <AnimatedSlider
        {...props}
        onChange={handleChange}
        sx={{
          height: 4,
          padding: '13px 0',
          
          '& .MuiSlider-thumb': {
            width: 16,
            height: 16,
            transition: 'box-shadow 0.2s ease-in-out',
            boxShadow: isActive 
              ? `0 0 0 8px ${alpha(theme.palette.primary.main, 0.16)}`
              : `0 0 0 0px ${alpha(theme.palette.primary.main, 0)}`,
            '&::before': {
              boxShadow: '0 0 1px 0 rgba(0,0,0,0.12)',
            },
            '&.Mui-active': {
              boxShadow: `0 0 0 14px ${alpha(theme.palette.primary.main, 0.16)}`,
            },
          },
          
          '& .MuiSlider-rail': {
            opacity: 0.32,
          },
          
          ...props.sx
        }}
      />
    </Box>
  );
};

/**
 * Enhanced Form Control Label with Apple-like animations
 */
export const EnhancedFormControlLabel: React.FC<FormControlLabelProps & {
  animationDelay?: number;
}> = ({
  animationDelay = 0,
  ...props
}) => {
  const { animationsEnabled } = useAnimationContext();
  const [visible, setVisible] = useState(false);
  
  // Set visible after mount with delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100 + animationDelay);
    
    return () => clearTimeout(timer);
  }, [animationDelay]);
  
  // Entrance animation
  const entranceAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
    },
    config: { tension: 280, friction: 60 },
    immediate: !animationsEnabled
  });
  
  return (
    <AnimatedFormControlLabel
      {...props}
      style={entranceAnimation}
    />
  );
};

/**
 * Enhanced Autocomplete with Apple-like animations and interactions
 */
export function EnhancedAutocomplete<
  T,
  Multiple extends boolean = false,
  DisableClearable extends boolean = false,
  FreeSolo extends boolean = false
>(props: AutocompleteProps<T, Multiple, DisableClearable, FreeSolo> & {
  label?: string;
  helperText?: string;
  error?: boolean;
  animationDelay?: number;
}) {
  const { 
    label,
    helperText,
    error,
    animationDelay = 0,
    ...autocompleteProps 
  } = props;
  
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [isFocused, setIsFocused] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [visible, setVisible] = useState(false);
  
  // Set visible after mount with delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100 + animationDelay);
    
    return () => clearTimeout(timer);
  }, [animationDelay]);
  
  // Entrance animation
  const entranceAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
    },
    config: { tension: 280, friction: 60 },
    immediate: !animationsEnabled
  });
  
  // Focus animation
  const focusAnimation = useSpring({
    transform: isFocused && animationsEnabled ? 'scale(1.01)' : 'scale(1)',
    boxShadow: isFocused && animationsEnabled 
      ? `0 0 0 2px ${alpha(theme.palette.primary.main, 0.25)}`
      : 'none',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Hover animation
  const hoverAnimation = useSpring({
    backgroundColor: isHovered && !isFocused && animationsEnabled
      ? alpha(theme.palette.primary.main, 0.02)
      : 'transparent',
    config: { tension: 350, friction: 18 },
    immediate: !animationsEnabled
  });
  
  // Combined animation styles
  const combinedStyle = {
    ...entranceAnimation,
    ...focusAnimation,
    ...hoverAnimation,
  };
  
  // Handle focus events
  const handleFocus = () => {
    setIsFocused(true);
    if (animationsEnabled) {
      soundEffects.tap();
    }
  };
  
  const handleBlur = () => {
    setIsFocused(false);
  };
  
  // Handle mouse events
  const handleMouseEnter = () => {
    setIsHovered(true);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
  };
  
  // Handle open/close
  const handleOpen = () => {
    if (animationsEnabled) {
      soundEffects.tap();
    }
  };
  
  // Handle selection
  const handleChange = () => {
    if (animationsEnabled) {
      soundEffects.switch();
    }
  };
  
  return (
    <AnimatedFormControl 
      fullWidth={autocompleteProps.fullWidth}
      style={entranceAnimation}
      error={error}
    >
      <AnimatedAutocomplete
        {...autocompleteProps}
        componentsProps={{
          paper: {
            sx: {
              borderRadius: '8px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
              overflow: 'hidden',
            }
          },
          ...autocompleteProps.componentsProps
        }}
        onFocus={handleFocus}
        onBlur={handleBlur}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onOpen={handleOpen}
        onChange={(event, value) => {
          handleChange();
          if (autocompleteProps.onChange) {
            autocompleteProps.onChange(event, value, 'selectOption', null);
          }
        }}
        PaperComponent={({ children, ...paperProps }) => (
          <Grow in timeout={200}>
            <div {...paperProps}>{children}</div>
          </Grow>
        )}
        renderInput={(params) => (
          <AnimatedTextField
            {...params}
            label={label}
            error={error}
            style={combinedStyle}
            sx={{
              '& .MuiOutlinedInput-root': {
                transition: 'all 0.2s ease-in-out',
                borderRadius: '8px',
                
                '&.Mui-focused': {
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderWidth: '2px',
                  }
                },
              },
            }}
          />
        )}
      />
      
      {helperText && (
        <AnimatedFormHelperText
          style={useSpring({
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(-5px)',
            config: { tension: 280, friction: 60 },
            delay: 150 + animationDelay,
            immediate: !animationsEnabled
          })}
          error={error}
        >
          {helperText}
        </AnimatedFormHelperText>
      )}
    </AnimatedFormControl>
  );
}

/**
 * Enhanced InputGroup with label and multiple input elements
 */
export const EnhancedInputGroup: React.FC<{
  label: string;
  description?: string;
  error?: boolean;
  children: React.ReactNode;
  animationDelay?: number;
}> = ({
  label,
  description,
  error,
  children,
  animationDelay = 0
}) => {
  const theme = useTheme();
  const { animationsEnabled } = useAnimationContext();
  const [visible, setVisible] = useState(false);
  
  // Set visible after mount with delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(true);
    }, 100 + animationDelay);
    
    return () => clearTimeout(timer);
  }, [animationDelay]);
  
  // Entrance animations with staggered timing
  const labelAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
    },
    config: { tension: 280, friction: 60 },
    delay: animationDelay,
    immediate: !animationsEnabled
  });
  
  const descriptionAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
    },
    config: { tension: 280, friction: 60 },
    delay: animationDelay + 50,
    immediate: !animationsEnabled
  });
  
  const contentAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(10px)' },
    to: { 
      opacity: visible && animationsEnabled ? 1 : 0,
      transform: visible && animationsEnabled ? 'translateY(0px)' : 'translateY(10px)'
    },
    config: { tension: 280, friction: 60 },
    delay: animationDelay + 100,
    immediate: !animationsEnabled
  });
  
  return (
    <Box sx={{ mb: 3 }}>
      <animated.div style={labelAnimation}>
        <Typography 
          variant="subtitle1" 
          component="label"
          color={error ? 'error' : 'text.primary'}
          sx={{ 
            display: 'block', 
            mb: 0.5, 
            fontWeight: 500,
          }}
        >
          {label}
        </Typography>
      </animated.div>
      
      {description && (
        <animated.div style={descriptionAnimation}>
          <Typography 
            variant="body2" 
            color={error ? 'error' : 'text.secondary'}
            sx={{ mb: 1.5 }}
          >
            {description}
          </Typography>
        </animated.div>
      )}
      
      <animated.div style={contentAnimation}>
        {children}
      </animated.div>
    </Box>
  );
};

export default {
  TextField: EnhancedTextField,
  Select: EnhancedSelect,
  Checkbox: EnhancedCheckbox,
  Radio: EnhancedRadio,
  Switch: EnhancedSwitch,
  Slider: EnhancedSlider,
  FormControlLabel: EnhancedFormControlLabel,
  Autocomplete: EnhancedAutocomplete,
  InputGroup: EnhancedInputGroup
};