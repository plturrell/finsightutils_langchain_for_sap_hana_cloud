import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  Paper,
  Collapse,
  Fade,
  IconButton,
  Tooltip,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Info as InfoIcon,
  Code as CodeIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import HumanText from './HumanText';
import { humanize } from '../utils/humanLanguage';

interface ProgressiveDisclosureProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  technicalDetails?: React.ReactNode;
  defaultExpanded?: boolean;
  infoTooltip?: string;
}

/**
 * ProgressiveDisclosure provides a clean UI pattern for hiding technical complexity
 * while allowing advanced users to access it when needed.
 */
const ProgressiveDisclosure: React.FC<ProgressiveDisclosureProps> = ({
  title,
  description,
  children,
  technicalDetails,
  defaultExpanded = false,
  infoTooltip,
}) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [showTechnical, setShowTechnical] = useState(false);
  
  return (
    <Paper
      elevation={expanded ? 2 : 1}
      sx={{
        border: '1px solid',
        borderColor: expanded ? alpha(theme.palette.primary.main, 0.2) : alpha('#000', 0.08),
        borderRadius: 3,
        overflow: 'hidden',
        mb: { xs: 3, sm: 4, md: 5 },
        transition: theme.transitions.create(['box-shadow', 'border-color'], {
          duration: theme.transitions.duration.standard,
        }),
        position: 'relative',
        '&:hover': {
          boxShadow: expanded ? '0 6px 20px rgba(0, 0, 0, 0.08)' : '0 3px 10px rgba(0, 0, 0, 0.05)',
        },
        // Add subtle highlight when expanded
        '&:after': expanded ? {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, primary.main, primary.light)',
          backgroundImage: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
          borderRadius: '3px 3px 0 0',
          opacity: 0.85,
        } : {},
      }}
    >
      {/* Header with enhanced white space and material consistency */}
      <Box
        sx={{
          p: { xs: 2, sm: 2.5, md: 3 },
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: expanded 
            ? alpha(theme.palette.primary.main, 0.06)
            : alpha(theme.palette.background.default, 0.8),
          color: expanded ? 'primary.main' : 'text.primary',
          transition: theme.transitions.create(
            ['background-color', 'color', 'padding'], 
            { duration: theme.transitions.duration.shorter }
          ),
          borderBottom: expanded 
            ? `1px solid ${alpha(theme.palette.primary.main, 0.12)}`
            : 'none',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <HumanText 
            variant="h6" 
            fontWeight={expanded ? 600 : 500}
            sx={{
              transition: theme.transitions.create(['font-weight', 'letter-spacing'], {
                duration: theme.transitions.duration.shorter,
              }),
              fontSize: { xs: '1rem', sm: '1.125rem', md: '1.25rem' },
              letterSpacing: expanded ? '-0.01em' : 'normal',
            }}
          >
            {title}
          </HumanText>
          
          {infoTooltip && (
            <Tooltip 
              title={infoTooltip} 
              arrow 
              placement="top"
              componentsProps={{
                tooltip: {
                  sx: {
                    maxWidth: 320,
                    fontSize: '0.75rem',
                    lineHeight: 1.5,
                    padding: '10px 14px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                    backgroundColor: 'rgba(33, 33, 33, 0.92)',
                    borderRadius: 2,
                  }
                },
                arrow: {
                  sx: {
                    color: 'rgba(33, 33, 33, 0.92)',
                  }
                }
              }}
            >
              <IconButton 
                size="small" 
                sx={{ 
                  ml: 1.5, 
                  color: 'inherit', 
                  opacity: 0.7,
                  width: 32,
                  height: 32,
                  '&:hover': {
                    opacity: 1,
                    backgroundColor: alpha(theme.palette.primary.main, 0.08),
                  }
                }}
              >
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
        
        <IconButton
          onClick={() => setExpanded(!expanded)}
          sx={{ 
            color: 'inherit',
            width: 36,
            height: 36,
            transform: expanded ? 'rotate(0deg)' : 'rotate(0deg)',
            transition: theme.transitions.create(['transform', 'background-color'], {
              duration: theme.transitions.duration.shorter,
            }),
            '&:hover': {
              backgroundColor: alpha(theme.palette.primary.main, 0.08),
            }
          }}
        >
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>
      
      {/* Content with improved white space and transitions */}
      <Collapse 
        in={expanded}
        timeout={{ enter: 450, exit: 350 }}
        easing={{
          enter: theme.transitions.easing.easeOut,
          exit: theme.transitions.easing.easeIn,
        }}
      >
        <Box 
          sx={{ 
            p: { xs: 2.5, sm: 3, md: 4 },
            pt: { xs: 3, sm: 3.5, md: 4.5 },
            // Add subtle background pattern for visual interest
            backgroundImage: expanded 
              ? 'radial-gradient(circle at 25px 25px, rgba(0, 102, 179, 0.02) 2%, transparent 0%), radial-gradient(circle at 75px 75px, rgba(0, 102, 179, 0.02) 2%, transparent 0%)'
              : 'none',
            backgroundSize: '100px 100px',
            // Add a subtle sheen effect with gradient overlay
            '&:before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              pointerEvents: 'none',
              background: 'linear-gradient(180deg, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0) 50%)',
              opacity: 0.5,
            }
          }}
        >
          {description && (
            <HumanText 
              variant="body1" 
              color="text.secondary" 
              sx={{ 
                mb: { xs: 3, md: 4 },
                maxWidth: '42rem',
                lineHeight: 1.7,
                fontSize: { xs: '0.9375rem', md: '1rem' },
                letterSpacing: '0.01em',
                // Animated entry with improved timing
                animation: 'fadeIn 0.6s ease-out',
                '@keyframes fadeIn': {
                  from: { opacity: 0, transform: 'translateY(10px)' },
                  to: { opacity: 1, transform: 'translateY(0)' }
                }
              }}
            >
              {description}
            </HumanText>
          )}
          
          <Box sx={{ 
            // Enhanced fade-in animation for content with staggered delay
            animation: 'fadeIn 0.7s ease-out 0.15s both', 
            '@keyframes fadeIn': {
              from: { opacity: 0, transform: 'translateY(10px)' },
              to: { opacity: 1, transform: 'translateY(0)' }
            },
            // Add extra spacing at the bottom when no technical details
            mb: !technicalDetails ? { xs: 1, md: 2 } : 0,
          }}>
            {children}
          </Box>
          
          {/* Technical Details with improved visual hierarchy and white space */}
          {technicalDetails && (
            <>
              <Divider sx={{ 
                my: { xs: 4, md: 5 },
                opacity: 0.6,
                width: '100%',
                maxWidth: '42rem',
                mx: 'auto',
                background: 'linear-gradient(90deg, transparent, rgba(0,0,0,0.09) 50%, transparent 100%)',
                height: '1px',
                border: 'none',
              }} />
              
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center',
                mb: showTechnical ? { xs: 3, md: 4 } : 0,
                mt: { xs: 1, md: 2 },
              }}>
                <Button
                  variant={showTechnical ? "contained" : "outlined"}
                  color={showTechnical ? "primary" : "inherit"}
                  size="medium"
                  startIcon={showTechnical ? <ExpandLessIcon /> : <CodeIcon />}
                  onClick={() => setShowTechnical(!showTechnical)}
                  sx={{ 
                    borderRadius: '24px',
                    px: { xs: 2.5, md: 3 },
                    py: 1,
                    height: 40,
                    fontWeight: 500,
                    transition: theme.transitions.create(
                      ['background-color', 'box-shadow', 'border-color'],
                      { duration: theme.transitions.duration.shorter }
                    ),
                    ...(showTechnical ? {
                      boxShadow: '0 2px 8px rgba(0, 102, 179, 0.25)',
                      '&:hover': {
                        boxShadow: '0 4px 12px rgba(0, 102, 179, 0.35)',
                      }
                    } : {
                      borderColor: alpha('#000', 0.15),
                      color: 'text.secondary',
                      '&:hover': {
                        borderColor: alpha(theme.palette.primary.main, 0.5),
                        backgroundColor: alpha(theme.palette.primary.main, 0.04),
                      }
                    })
                  }}
                >
                  <HumanText sx={{ letterSpacing: '0.01em' }}>
                    {showTechnical ? 'Hide Technical Details' : 'Show Technical Details'}
                  </HumanText>
                </Button>
              </Box>
              
              <Collapse 
                in={showTechnical}
                timeout={{ enter: 550, exit: 350 }}
                easing={{
                  enter: theme.transitions.easing.easeOut,
                  exit: theme.transitions.easing.easeIn,
                }}
              >
                <Paper
                  elevation={0}
                  sx={{ 
                    mt: { xs: 2, md: 3 }, 
                    p: { xs: 2.5, sm: 3, md: 3.5 },
                    borderRadius: 2.5, 
                    bgcolor: alpha(theme.palette.background.paper, 0.65),
                    border: `1px solid ${alpha(theme.palette.primary.main, 0.08)}`,
                    backdropFilter: 'blur(10px)',
                    animation: 'fadeInUp 0.6s ease-out',
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.06)',
                    '@keyframes fadeInUp': {
                      from: { opacity: 0, transform: 'translateY(16px)' },
                      to: { opacity: 1, transform: 'translateY(0)' }
                    }
                  }}
                >
                  <Box sx={{ 
                    mb: { xs: 2, md: 2.5 }, 
                    display: 'flex', 
                    alignItems: 'center',
                    pb: 1.5,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.6)}`
                  }}>
                    <SettingsIcon 
                      fontSize="small" 
                      sx={{ 
                        mr: 1.5, 
                        color: 'primary.main',
                        animation: 'pulse 2.5s infinite ease-in-out',
                        '@keyframes pulse': {
                          '0%': { opacity: 0.6 },
                          '50%': { opacity: 1 },
                          '100%': { opacity: 0.6 }
                        }
                      }} 
                    />
                    <HumanText 
                      variant="subtitle1" 
                      color="primary.main"
                      fontWeight={600}
                      sx={{ letterSpacing: '-0.01em' }}
                    >
                      Advanced Configuration
                    </HumanText>
                  </Box>
                  
                  <Box sx={{ 
                    animation: 'fadeIn 0.6s ease-out 0.25s both',
                    px: { xs: 0, md: 0.5 },
                  }}>
                    {technicalDetails}
                  </Box>
                </Paper>
              </Collapse>
            </>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default ProgressiveDisclosure;