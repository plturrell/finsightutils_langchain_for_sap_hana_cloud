import { createTheme, responsiveFontSizes, alpha } from '@mui/material/styles';

// Define spacing constants for consistency
const SPACE_UNIT = 8;
const SPACE = {
  xs: `${SPACE_UNIT * 0.5}px`,       // 4px
  sm: `${SPACE_UNIT}px`,             // 8px
  md: `${SPACE_UNIT * 2}px`,         // 16px
  lg: `${SPACE_UNIT * 3}px`,         // 24px
  xl: `${SPACE_UNIT * 4}px`,         // 32px
  xxl: `${SPACE_UNIT * 5}px`,        // 40px
};

// Define consistent radiuses
const RADIUS = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  pill: 100,
};

// Define elevation constants (shadows)
const createElevation = (level: number, color = '0, 0, 0') => {
  const opacity = 0.05 + (level * 0.01);
  return `0px ${level * 2}px ${level * 4}px rgba(${color}, ${opacity})`;
};

// Simplified, intentional color palette with precise relationships
const PRIMARY = {
  100: '#E2F1FD', // Lightest shade - backgrounds, highlights
  200: '#B6DCFA', // Light shade - hover states, secondary backgrounds
  300: '#64A5E8', // Medium shade - secondary elements
  400: '#1B78D6', // Medium-dark - interactive elements, links
  500: '#0066B3', // Main color - primary actions, key elements
  600: '#004E8A', // Dark shade - focused states, text on light backgrounds
  700: '#003B6A', // Darkest shade - text, icons
};

// Simplified neutral palette for maximum focus
const NEUTRAL = {
  100: '#FFFFFF', // Pure white - main backgrounds
  200: '#F9FAFB', // Off-white - secondary backgrounds
  300: '#F0F2F5', // Light gray - tertiary backgrounds, disabled states
  400: '#DFE3E8', // Medium-light gray - borders, dividers
  500: '#8C9BAA', // Medium gray - disabled text, icons
  600: '#556677', // Medium-dark gray - secondary text
  700: '#2C3E50', // Dark gray - primary text
};

// Create a meticulously crafted theme with perfect proportions
let theme = createTheme({
  spacing: SPACE_UNIT,
  palette: {
    mode: 'light',
    primary: {
      main: PRIMARY[500],
      light: PRIMARY[300],
      dark: PRIMARY[600],
      contrastText: '#FFFFFF',
      ...PRIMARY,
    },
    // Use neutral palette instead of secondary for maximum focus
    secondary: {
      main: PRIMARY[400],
      light: PRIMARY[300],
      dark: PRIMARY[600],
      contrastText: '#FFFFFF',
    },
    background: {
      default: NEUTRAL[200], // Subtle off-white
      paper: NEUTRAL[100],   // Pure white
    },
    // Simplified, intentional feedback colors
    error: {
      main: '#E53935',
      light: '#FFEBEE',
      dark: '#C62828',
    },
    warning: {
      main: '#F5A623',
      light: '#FFF8E1',
      dark: '#F57F17',
    },
    success: {
      main: '#34C759',
      light: '#E8F5E9',
      dark: '#2E7D32',
    },
    info: {
      main: PRIMARY[400],
      light: PRIMARY[100],
      dark: PRIMARY[600],
    },
    text: {
      primary: NEUTRAL[700],   // Main text color
      secondary: NEUTRAL[600], // Secondary text
      disabled: NEUTRAL[500],  // Disabled text
    },
    divider: NEUTRAL[400],
    action: {
      active: NEUTRAL[600],
      hover: 'rgba(0, 102, 179, 0.04)',
      selected: 'rgba(0, 102, 179, 0.08)',
      disabled: NEUTRAL[500],
      disabledBackground: NEUTRAL[300],
    },
  },
  typography: {
    // Base font is SF Pro Display - Apple's system font
    fontFamily: '"SF Pro Display", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
    
    // Perfect typographic scale with precise 1.2 ratio and consistent optical adjustments
    h1: {
      fontSize: '2.488rem', // Base × 1.2⁴
      fontWeight: 600,
      letterSpacing: '-0.022em', // Precise optical adjustment
      lineHeight: 1.2,
      marginBottom: SPACE.md,
    },
    h2: {
      fontSize: '2.074rem', // Base × 1.2³
      fontWeight: 600,
      letterSpacing: '-0.021em',
      lineHeight: 1.25,
      marginBottom: SPACE.md,
    },
    h3: {
      fontSize: '1.728rem', // Base × 1.2²
      fontWeight: 600,
      letterSpacing: '-0.02em',
      lineHeight: 1.3,
      marginBottom: SPACE.sm,
    },
    h4: {
      fontSize: '1.44rem', // Base × 1.2¹
      fontWeight: 500,
      letterSpacing: '-0.017em',
      lineHeight: 1.35,
      marginBottom: SPACE.sm,
    },
    h5: {
      fontSize: '1.2rem', // Base × 1.2⁰
      fontWeight: 600,
      letterSpacing: '-0.015em',
      lineHeight: 1.4,
      marginBottom: SPACE.sm,
    },
    h6: {
      fontSize: '1rem', // Base
      fontWeight: 600,
      letterSpacing: '-0.01em',
      lineHeight: 1.4,
      marginBottom: SPACE.xs,
    },
    subtitle1: {
      fontSize: '1rem',
      fontWeight: 500,
      letterSpacing: '-0.008em',
      lineHeight: 1.5,
      marginBottom: SPACE.xs,
    },
    subtitle2: {
      fontSize: '0.875rem', // Base × 0.875
      fontWeight: 600,
      letterSpacing: '-0.006em',
      lineHeight: 1.5,
      marginBottom: SPACE.xs,
    },
    body1: {
      fontSize: '1rem',
      fontWeight: 400,
      letterSpacing: '-0.005em',
      lineHeight: 1.5, // Perfect line height for reading
      marginBottom: SPACE.md,
    },
    body2: {
      fontSize: '0.875rem',
      fontWeight: 400,
      letterSpacing: '-0.003em',
      lineHeight: 1.5,
      marginBottom: SPACE.sm,
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 600, // Increased weight for better tap affordance
      letterSpacing: '0',
      lineHeight: 1,
      textTransform: 'none', // Apple never uses text-transform
    },
    caption: {
      fontSize: '0.75rem', // Base × 0.75
      fontWeight: 400,
      letterSpacing: '0',
      lineHeight: 1.4,
    },
    overline: {
      fontSize: '0.75rem',
      fontWeight: 500,
      letterSpacing: '0.02em',
      lineHeight: 1.4,
      textTransform: 'uppercase',
    },
  },
  shape: {
    borderRadius: RADIUS.md,
  },
  shadows: [
    'none',
    createElevation(1),
    createElevation(2),
    createElevation(3),
    createElevation(4),
    createElevation(5),
    createElevation(6),
    createElevation(7),
    createElevation(8),
    createElevation(9),
    createElevation(10),
    createElevation(11),
    createElevation(12),
    createElevation(13),
    createElevation(14),
    createElevation(15),
    createElevation(16),
    createElevation(17),
    createElevation(18),
    createElevation(19),
    createElevation(20),
    createElevation(21),
    createElevation(22),
    createElevation(23),
    createElevation(24),
  ],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        // Global styles
        'html, body': {
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale',
          height: '100%',
          width: '100%',
        },
        '#root': {
          height: '100%',
        },
        // Apply consistent focus styles for better accessibility
        '*, *:focus, *:focus-visible': {
          outline: 'none',
        },
        '*:focus-visible': {
          outline: `2px solid ${PRIMARY[500]}`,
          outlineOffset: '2px',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.sm,
          padding: `${SPACE.sm} ${SPACE.md}`,
          fontWeight: 500,
          boxShadow: 'none',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: createElevation(3),
          },
        },
        contained: {
          '&:hover': {
            boxShadow: createElevation(3),
          },
        },
        outlined: {
          borderWidth: 1.5,
          padding: `${parseInt(SPACE.sm) - 1.5}px ${parseInt(SPACE.md) - 1.5}px`,
        },
        text: {
          '&:hover': {
            backgroundColor: alpha(PRIMARY[500], 0.05),
          },
        },
        sizeLarge: {
          padding: `${SPACE.md} ${SPACE.lg}`,
          fontSize: '1rem',
        },
        sizeSmall: {
          padding: `${SPACE.xs} ${SPACE.sm}`,
          fontSize: '0.75rem',
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.sm,
          padding: SPACE.xs,
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(PRIMARY[500], 0.05),
          },
        },
        sizeLarge: {
          padding: SPACE.sm,
          fontSize: '1.5rem',
        },
        sizeSmall: {
          padding: '2px',
          fontSize: '0.875rem',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.lg,
          boxShadow: createElevation(3),
          overflow: 'hidden',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            boxShadow: createElevation(5),
          },
        },
      },
    },
    MuiCardHeader: {
      styleOverrides: {
        root: {
          padding: SPACE.lg,
          paddingBottom: SPACE.md,
        },
        title: {
          fontSize: '1.125rem',
          fontWeight: 600,
        },
        subheader: {
          fontSize: '0.875rem',
          color: 'text.secondary',
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: {
          padding: SPACE.lg,
          '&:last-child': {
            paddingBottom: SPACE.lg,
          },
        },
      },
    },
    MuiCardActions: {
      styleOverrides: {
        root: {
          padding: SPACE.md,
          paddingTop: 0,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          marginBottom: SPACE.md,
          '& .MuiOutlinedInput-root': {
            borderRadius: RADIUS.sm,
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              boxShadow: `0 0 0 1px ${alpha(PRIMARY[500], 0.2)}`,
            },
            '&.Mui-focused': {
              boxShadow: `0 0 0 2px ${alpha(PRIMARY[500], 0.3)}`,
            },
          },
          '& .MuiInputLabel-root': {
            fontSize: '0.875rem',
            fontWeight: 500,
          },
          '& .MuiInputBase-input': {
            padding: `${SPACE.sm} ${SPACE.md}`,
          },
          '& .MuiFormHelperText-root': {
            marginTop: SPACE.xs,
            fontSize: '0.75rem',
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.sm,
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          padding: `${SPACE.sm} ${SPACE.md}`,
          borderRadius: RADIUS.xs,
          margin: '2px',
          '&:hover': {
            backgroundColor: alpha(PRIMARY[500], 0.05),
          },
          '&.Mui-selected': {
            backgroundColor: alpha(PRIMARY[500], 0.08),
            '&:hover': {
              backgroundColor: alpha(PRIMARY[500], 0.12),
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Remove default gradient for more consistency
        },
        rounded: {
          borderRadius: RADIUS.lg,
        },
        elevation1: {
          boxShadow: createElevation(2),
        },
        elevation2: {
          boxShadow: createElevation(4),
        },
        elevation3: {
          boxShadow: createElevation(6),
        },
        elevation4: {
          boxShadow: createElevation(8),
        },
        elevation5: {
          boxShadow: createElevation(10),
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: createElevation(2),
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
        },
      },
    },
    MuiToolbar: {
      styleOverrides: {
        root: {
          padding: `0 ${SPACE.lg}`,
          minHeight: '64px',
          '@media (min-width:600px)': {
            minHeight: '64px',
            padding: `0 ${SPACE.lg}`,
          },
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          borderRight: 'none',
          boxShadow: createElevation(5),
          borderRadius: 0,
          backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.9), rgba(245, 247, 250, 0.9))',
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          margin: `${SPACE.md} 0`,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontSize: '0.875rem',
          fontWeight: 500,
          minWidth: 120,
          padding: `${SPACE.md} ${SPACE.md}`,
          transition: 'all 0.2s ease-in-out',
          '&.Mui-selected': {
            fontWeight: 600,
            color: PRIMARY[600],
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          minHeight: 48,
        },
        indicator: {
          height: 3,
          borderTopLeftRadius: RADIUS.xs,
          borderTopRightRadius: RADIUS.xs,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.pill,
          height: 32,
          fontSize: '0.75rem',
          fontWeight: 500,
          padding: `0 ${SPACE.sm}`,
          '&.MuiChip-outlined': {
            backgroundColor: 'transparent',
          },
        },
        label: {
          padding: `0 ${SPACE.sm}`,
        },
        deleteIcon: {
          margin: `0 ${SPACE.xs} 0 -${SPACE.xs}`,
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: 'rgba(33, 33, 33, 0.9)',
          fontSize: '0.75rem',
          padding: `${SPACE.xs} ${SPACE.sm}`,
          borderRadius: RADIUS.sm,
          boxShadow: createElevation(2),
        },
        arrow: {
          color: 'rgba(33, 33, 33, 0.9)',
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.xs,
          height: 6,
          backgroundColor: alpha(PRIMARY[500], 0.12),
        },
        bar: {
          borderRadius: RADIUS.xs,
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        root: {
          width: 58,
          height: 38,
          padding: SPACE.xs,
        },
        switchBase: {
          padding: 8,
          '&.Mui-checked': {
            transform: 'translateX(20px)',
          },
        },
        thumb: {
          width: 22,
          height: 22,
          boxShadow: createElevation(1),
        },
        track: {
          borderRadius: RADIUS.pill,
          backgroundColor: '#E0E0E0',
          opacity: 1,
          '.Mui-checked.Mui-checked + &': {
            backgroundColor: alpha(PRIMARY[500], 0.8),
            opacity: 1,
          },
        },
      },
    },
    MuiAvatar: {
      styleOverrides: {
        root: {
          backgroundColor: PRIMARY[100],
          color: PRIMARY[700],
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          padding: `${SPACE.xs} ${SPACE.sm}`,
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.sm,
          padding: `${SPACE.sm} ${SPACE.md}`,
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(PRIMARY[500], 0.05),
          },
          '&.Mui-selected': {
            backgroundColor: alpha(PRIMARY[500], 0.08),
            '&:hover': {
              backgroundColor: alpha(PRIMARY[500], 0.12),
            },
          },
        },
      },
    },
    MuiListItemText: {
      styleOverrides: {
        root: {
          margin: `${SPACE.xs} 0`,
        },
        primary: {
          fontSize: '0.875rem',
          fontWeight: 500,
        },
        secondary: {
          fontSize: '0.75rem',
        },
      },
    },
    MuiListItemIcon: {
      styleOverrides: {
        root: {
          minWidth: 40,
          color: 'inherit',
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.md,
          overflow: 'hidden',
          '&:before': {
            display: 'none', // Remove default line
          },
          '&.Mui-expanded': {
            margin: 0,
            '&:first-of-type': {
              marginTop: 0,
            },
            '&:last-of-type': {
              marginBottom: 0,
            },
          },
        },
      },
    },
    MuiAccordionSummary: {
      styleOverrides: {
        root: {
          minHeight: 56,
          padding: `0 ${SPACE.lg}`,
          '&.Mui-expanded': {
            minHeight: 56,
          },
        },
        content: {
          margin: `${SPACE.md} 0`,
          '&.Mui-expanded': {
            margin: `${SPACE.md} 0`,
          },
        },
        expandIconWrapper: {
          color: 'text.secondary',
        },
      },
    },
    MuiAccordionDetails: {
      styleOverrides: {
        root: {
          padding: `${SPACE.md} ${SPACE.lg} ${SPACE.lg}`,
          borderTop: `1px solid ${alpha('#000', 0.08)}`,
        },
      },
    },
    MuiRating: {
      styleOverrides: {
        root: {
          color: PRIMARY[500],
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          padding: SPACE.md,
          borderBottom: `1px solid ${alpha('#000', 0.08)}`,
        },
        head: {
          fontWeight: 600,
          backgroundColor: alpha('#000', 0.02),
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: alpha('#000', 0.02),
          },
        },
      },
    },
    MuiLink: {
      styleOverrides: {
        root: {
          color: PRIMARY[600],
          textDecoration: 'none',
          fontWeight: 500,
          '&:hover': {
            textDecoration: 'underline',
          },
        },
      },
    },
    MuiContainer: {
      styleOverrides: {
        root: {
          padding: `0 ${SPACE.lg}`,
          [theme.breakpoints.up('sm')]: {
            padding: `0 ${SPACE.lg}`,
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.md,
          padding: `${SPACE.sm} ${SPACE.md}`,
        },
        icon: {
          padding: `${SPACE.xs} 0`,
        },
        message: {
          padding: `${SPACE.xs} 0`,
        },
      },
    },
    MuiAlertTitle: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          marginBottom: SPACE.xs,
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: RADIUS.lg,
          boxShadow: createElevation(8),
          padding: SPACE.md,
        },
      },
    },
    MuiDialogTitle: {
      styleOverrides: {
        root: {
          padding: `${SPACE.lg} ${SPACE.lg} ${SPACE.md}`,
          fontSize: '1.25rem',
          fontWeight: 600,
        },
      },
    },
    MuiDialogContent: {
      styleOverrides: {
        root: {
          padding: `0 ${SPACE.lg} ${SPACE.md}`,
        },
      },
    },
    MuiDialogActions: {
      styleOverrides: {
        root: {
          padding: `${SPACE.md} ${SPACE.lg} ${SPACE.lg}`,
          '& > :not(:first-of-type)': {
            marginLeft: SPACE.sm,
          },
        },
      },
    },
    MuiBreadcrumbs: {
      styleOverrides: {
        root: {
          marginBottom: SPACE.md,
        },
        separator: {
          marginLeft: SPACE.xs,
          marginRight: SPACE.xs,
        },
      },
    },
    MuiSkeleton: {
      styleOverrides: {
        root: {
          borderRadius: RADIUS.sm,
        },
      },
    },
    MuiToggleButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          color: 'text.primary',
          borderColor: alpha('#000', 0.12),
          '&.Mui-selected': {
            backgroundColor: alpha(PRIMARY[500], 0.12),
            color: PRIMARY[700],
            '&:hover': {
              backgroundColor: alpha(PRIMARY[500], 0.2),
            },
          },
        },
      },
    },
    MuiBadge: {
      styleOverrides: {
        badge: {
          fontWeight: 600,
          fontSize: '0.625rem',
          padding: '0 4px',
          minWidth: 18,
          height: 18,
        },
      },
    },
    MuiPopover: {
      styleOverrides: {
        paper: {
          boxShadow: createElevation(3),
          borderRadius: RADIUS.md,
        },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          marginTop: SPACE.xs,
        },
        list: {
          padding: `${SPACE.xs} ${SPACE.xs}`,
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: {
          height: 8,
          padding: `${SPACE.sm} 0`,
        },
        track: {
          height: 4,
          borderRadius: RADIUS.xs,
        },
        rail: {
          height: 4,
          borderRadius: RADIUS.xs,
          opacity: 0.3,
        },
        thumb: {
          width: 16,
          height: 16,
          boxShadow: createElevation(2),
          '&:before': {
            boxShadow: '0 0 0 8px rgba(0, 102, 179, 0.1)',
          },
          '&:hover, &.Mui-focusVisible': {
            boxShadow: '0 0 0 0.25rem rgba(0, 102, 179, 0.2)',
          },
          '&.Mui-active': {
            boxShadow: '0 0 0 0.4rem rgba(0, 102, 179, 0.3)',
          },
        },
        valueLabel: {
          padding: `${SPACE.xs} ${SPACE.sm}`,
          borderRadius: RADIUS.sm,
          backgroundColor: PRIMARY[700],
        },
        mark: {
          height: 8,
          width: 1,
          backgroundColor: alpha('#000', 0.3),
        },
        markActive: {
          backgroundColor: PRIMARY[500],
        },
      },
    },
  },
});

// Apply responsive font sizes
theme = responsiveFontSizes(theme);

export default theme;