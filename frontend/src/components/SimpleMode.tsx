import React, { useState } from 'react';
import {
  Box,
  Typography,
  Switch,
  FormControlLabel,
  Paper,
  Divider,
  useTheme,
  useMediaQuery,
  Fade,
  Container,
  Button,
} from '@mui/material';
import { Settings as SettingsIcon } from '@mui/icons-material';
import HumanText from './HumanText';
import { humanize } from '../utils/humanLanguage';

interface SimpleModeProps {
  isEnabled: boolean;
  onToggle: (enabled: boolean) => void;
  onOpenSettings?: () => void;
}

/**
 * SimpleMode provides a cleaner, more elegant user experience by hiding
 * technical parameters behind intelligent defaults.
 */
const SimpleMode: React.FC<SimpleModeProps> = ({ 
  isEnabled, 
  onToggle,
  onOpenSettings
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [hovered, setHovered] = useState(false);

  return (
    <Container maxWidth="md" sx={{ my: 4 }}>
      <Paper 
        elevation={0}
        sx={{
          borderRadius: 4,
          overflow: 'hidden',
          border: '1px solid',
          borderColor: isEnabled ? 'primary.light' : 'divider',
          transition: 'all 0.3s ease',
          position: 'relative',
          '&:hover': {
            boxShadow: isEnabled ? '0 8px 24px rgba(0, 0, 100, 0.1)' : 'none',
          }
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <Box 
          sx={{ 
            p: 3,
            display: 'flex',
            flexDirection: isMobile ? 'column' : 'row',
            alignItems: isMobile ? 'flex-start' : 'center',
            justifyContent: 'space-between',
            backgroundColor: isEnabled ? 'primary.light' : 'background.paper',
            color: isEnabled ? 'primary.contrastText' : 'text.primary',
          }}
        >
          <Box>
            <HumanText 
              variant="h5" 
              fontWeight="500"
              sx={{ mb: 0.5 }}
            >
              {isEnabled ? 'Simple Experience' : 'Advanced Experience'}
            </HumanText>
            <HumanText 
              variant={isEnabled ? "body2" : "body1"}
              color={isEnabled ? 'primary.contrastText' : 'text.secondary'}
            >
              {isEnabled 
                ? 'Enjoy a streamlined interface with intelligent defaults' 
                : 'All technical parameters and advanced settings are visible'}
            </HumanText>
          </Box>
          
          <FormControlLabel
            control={
              <Switch
                checked={isEnabled}
                onChange={(e) => onToggle(e.target.checked)}
                color="default"
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: '#fff',
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    backgroundColor: '#fff',
                    opacity: 0.5,
                  },
                }}
              />
            }
            label=""
            sx={{ mt: isMobile ? 2 : 0 }}
          />
        </Box>
        
        <Divider />
        
        <Box sx={{ p: 3, position: 'relative' }}>
          {isEnabled ? (
            <Box>
              <HumanText variant="body1" paragraph>
                In Simple Experience mode, the system will:
              </HumanText>
              
              <Box component="ul" sx={{ pl: 2 }}>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Automatically choose optimal GPU acceleration settings based on your hardware
                </HumanText>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Select the best embedding model for your content
                </HumanText>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Dynamically adjust batch sizes for peak performance
                </HumanText>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Hide technical parameters while maintaining full functionality
                </HumanText>
                <HumanText component="li" variant="body2">
                  Provide intuitive search with optimal default settings
                </HumanText>
              </Box>
              
              <Fade in={hovered}>
                <Button
                  variant="text"
                  color="inherit"
                  size="small"
                  startIcon={<SettingsIcon />}
                  onClick={onOpenSettings}
                  sx={{ 
                    position: 'absolute', 
                    bottom: 16, 
                    right: 16,
                    color: 'primary.contrastText',
                    opacity: 0.7,
                    '&:hover': {
                      opacity: 1,
                      backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    }
                  }}
                >
                  <HumanText>Configure</HumanText>
                </Button>
              </Fade>
            </Box>
          ) : (
            <Box>
              <HumanText variant="body1" paragraph>
                In Advanced Experience mode, you can:
              </HumanText>
              
              <Box component="ul" sx={{ pl: 2 }}>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Fine-tune GPU acceleration parameters (TensorRT, precision, batch size)
                </HumanText>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Select specific embedding models and dimensions
                </HumanText>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Configure database connection settings
                </HumanText>
                <HumanText component="li" variant="body2" sx={{ mb: 1 }}>
                  Customize search algorithms (similarity, MMR)
                </HumanText>
                <HumanText component="li" variant="body2">
                  Access all technical settings for complete control
                </HumanText>
              </Box>
            </Box>
          )}
        </Box>
      </Paper>
    </Container>
  );
};

export default SimpleMode;