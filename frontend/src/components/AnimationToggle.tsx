import React from 'react';
import {
  Box,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Menu,
  MenuItem,
  Typography,
  useTheme,
  alpha,
  Divider,
} from '@mui/material';
import {
  Animation as AnimationIcon,
  AnimationDisabled as AnimationDisabledIcon,
  VolumeUp as VolumeUpIcon,
  VolumeOff as VolumeOffIcon,
  Speed as SpeedIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';
import AnimationToggleBase from '@finsightdev/ui-animations/dist/components/AnimationToggle';

interface AnimationToggleProps {
  /** Position the toggle button at the top (true) or bottom (false) */
  positionTop?: boolean;
}

/**
 * A component that allows users to toggle animations and sound effects
 * This is a wrapper around the shared AnimationToggle component
 */
const AnimationToggle: React.FC<AnimationToggleProps> = ({ positionTop = true }) => {
  const theme = useTheme();
  
  // Pass MUI components to the shared AnimationToggle
  return (
    <AnimationToggleBase
      positionTop={positionTop}
      spacing={theme.spacing}
      theme={{
        palette: {
          primary: { main: theme.palette.primary.main },
          background: { paper: theme.palette.background.paper },
          text: { secondary: theme.palette.text.secondary },
          divider: theme.palette.divider
        }
      }}
      alpha={alpha}
      components={{
        Box,
        IconButton,
        Tooltip,
        Switch,
        FormControlLabel,
        Menu,
        MenuItem,
        Typography,
        Divider,
        AnimationIcon,
        AnimationDisabledIcon,
        VolumeUpIcon,
        VolumeOffIcon,
        SpeedIcon,
        TuneIcon
      }}
    />
  );
};

export default AnimationToggle;