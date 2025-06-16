import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Breadcrumbs,
  Link,
  Tab,
  Tabs,
  Button,
  Alert,
  Tooltip,
  IconButton,
  Switch,
  FormControlLabel,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Home as HomeIcon,
  Search as SearchIcon,
  NavigateNext as NavigateNextIcon,
  Storage as StorageIcon,
  BlurOn as BlurOnIcon,
  OpenInNew as OpenInNewIcon,
  Info as InfoIcon,
  ArrowBack as ArrowBackIcon,
  ArrowForward as ArrowForwardIcon,
  Analytics as AnalyticsIcon,
  Business as BusinessIcon,
} from '@mui/icons-material';
import { useParams, useNavigate, Link as RouterLink } from 'react-router-dom';
import { useSpring, animated, useTrail, useChain, useSpringRef } from '@react-spring/web';
import VectorExplorer from '../components/VectorExplorer';
import HumanText from '../components/HumanText';
import FinancialVisualizationExample from '../../components/visualization/FinancialVisualizationExample';

// Create animated versions of MUI components
const AnimatedBox = animated(Box);
const AnimatedTypography = animated(Typography);
const AnimatedPaper = animated(Paper);
const AnimatedCard = animated(Card);
const AnimatedAlert = animated(Alert);
const AnimatedButton = animated(Button);
const AnimatedGrid = animated(Grid);

interface VectorExplorationParams {
  vectorTable?: string;
}

const VectorExploration: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { vectorTable } = useParams<VectorExplorationParams>();
  
  // State
  const [tabValue, setTabValue] = useState<number>(0);
  const [selectedVectorId, setSelectedVectorId] = useState<string | null>(null);
  const [selectedMetadata, setSelectedMetadata] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [enhancedViewEnabled, setEnhancedViewEnabled] = useState<boolean>(false);
  const [recentTables, setRecentTables] = useState<string[]>([
    'VECTOR_CUSTOMER_DATA',
    'VECTOR_PRODUCT_DESCRIPTIONS',
    'VECTOR_FINANCIAL_REPORTS',
  ]);
  const [animationsVisible, setAnimationsVisible] = useState(false);

  // Animation spring refs for chaining
  const headerSpringRef = useSpringRef();
  const contentSpringRef = useSpringRef();
  const cardsSpringRef = useSpringRef();
  const lifecycleSpringRef = useSpringRef();

  // Header animation
  const headerAnimation = useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });

  // Content animation
  const contentAnimation = useSpring({
    ref: contentSpringRef,
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Cards trail animation
  const cardTrail = useTrail(recentTables.length, {
    ref: cardsSpringRef,
    from: { opacity: 0, transform: 'translateY(40px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(40px)' },
    config: { mass: 1, tension: 280, friction: 60 }
  });

  // Lifecycle animation
  const lifecycleAnimation = useSpring({
    ref: lifecycleSpringRef,
    from: { opacity: 0, transform: 'translateX(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateX(0)' : 'translateX(30px)' },
    config: { tension: 280, friction: 60 }
  });

  // Chain animations in sequence
  useChain(
    animationsVisible 
      ? [headerSpringRef, contentSpringRef, cardsSpringRef, lifecycleSpringRef] 
      : [lifecycleSpringRef, cardsSpringRef, contentSpringRef, headerSpringRef],
    animationsVisible 
      ? [0, 0.2, 0.4, 0.6] 
      : [0, 0.1, 0.2, 0.3]
  );

  // Make animations visible after initial render
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, 150);
    return () => clearTimeout(timer);
  }, []);
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Handle vector selection
  const handleVectorSelected = (vectorId: string, metadata: Record<string, any>) => {
    setSelectedVectorId(vectorId);
    setSelectedMetadata(metadata);
  };
  
  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Breadcrumbs */}
      <AnimatedBox
        style={useSpring({
          opacity: animationsVisible ? 1 : 0,
          transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
          config: { tension: 280, friction: 60 }
        })}
      >
        <Breadcrumbs 
          separator={<NavigateNextIcon fontSize="small" />} 
          aria-label="breadcrumb"
          sx={{ mb: 3 }}
        >
          <Link
            underline="hover"
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              transition: 'all 0.2s ease',
              '&:hover': {
                color: theme.palette.primary.main,
              }
            }}
            color="inherit"
            component={RouterLink}
            to="/"
          >
            <HomeIcon sx={{ mr: 0.5 }} fontSize="inherit" />
            Home
          </Link>
          {vectorTable ? (
            <>
              <Link
                underline="hover"
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    color: theme.palette.primary.main,
                  }
                }}
                color="inherit"
                component={RouterLink}
                to="/vector-exploration"
              >
                <BlurOnIcon sx={{ mr: 0.5 }} fontSize="inherit" />
                Vector Exploration
              </Link>
              <AnimatedTypography 
                color="text.primary" 
                sx={{ display: 'flex', alignItems: 'center' }}
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                  delay: 200,
                  config: { tension: 280, friction: 60 }
                })}
              >
                <StorageIcon sx={{ mr: 0.5 }} fontSize="inherit" />
                {vectorTable}
              </AnimatedTypography>
            </>
          ) : (
            <AnimatedTypography 
              color="text.primary" 
              sx={{ display: 'flex', alignItems: 'center' }}
              style={useSpring({
                opacity: animationsVisible ? 1 : 0,
                transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                delay: 200,
                config: { tension: 280, friction: 60 }
              })}
            >
              <BlurOnIcon sx={{ mr: 0.5 }} fontSize="inherit" />
              Vector Exploration
            </AnimatedTypography>
          )}
        </Breadcrumbs>
      </AnimatedBox>
      
      {/* Page Header */}
      <AnimatedBox 
        style={headerAnimation}
        sx={{ mb: 4 }}
      >
        <AnimatedTypography 
          variant="h4" 
          component={HumanText}
          gutterBottom
          style={useSpring({
            from: { 
              opacity: 0, 
              transform: 'translateY(-10px)',
              background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
              backgroundSize: '200% 100%',
              backgroundPosition: '0% 50%'
            },
            to: { 
              opacity: animationsVisible ? 1 : 0, 
              transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
              backgroundPosition: '100% 50%'
            },
            delay: 100,
            config: { tension: 280, friction: 60 },
          })}
          sx={{ 
            background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            transition: 'all 0.3s ease',
          }}
        >
          {vectorTable ? `Explore ${vectorTable}` : 'Vector Exploration'}
        </AnimatedTypography>
        <AnimatedTypography 
          variant="body1" 
          component={HumanText}
          color="text.secondary"
          style={useSpring({
            opacity: animationsVisible ? 1 : 0,
            transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
            delay: 150,
            config: { tension: 280, friction: 60 }
          })}
        >
          {vectorTable 
            ? 'Search, visualize, and explore vectors to gain insights and analyze patterns.'
            : 'Select a vector table to explore or search across all vectors in your database.'}
        </AnimatedTypography>
      </AnimatedBox>
      
      {/* Error Alert */}
      {error && (
        <AnimatedAlert 
          severity="error" 
          style={useSpring({
            opacity: animationsVisible ? 1 : 0,
            transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
            config: { tension: 280, friction: 60 }
          })}
          sx={{ 
            mb: 3,
            animation: 'slideIn 0.3s ease-out',
            '@keyframes slideIn': {
              from: { opacity: 0, transform: 'translateY(-10px)' },
              to: { opacity: 1, transform: 'translateY(0)' }
            }
          }}
        >
          {error}
        </AnimatedAlert>
      )}
      
      {/* View Mode Toggle */}
      {vectorTable && (
        <AnimatedBox 
          style={useSpring({
            opacity: animationsVisible ? 1 : 0,
            transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
            delay: 200,
            config: { tension: 280, friction: 60 }
          })}
          sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end' }}
        >
          <FormControlLabel
            control={
              <Switch
                checked={enhancedViewEnabled}
                onChange={(e) => setEnhancedViewEnabled(e.target.checked)}
                color="primary"
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: theme.palette.primary.main,
                    '&:hover': {
                      backgroundColor: alpha(theme.palette.primary.main, 0.08),
                    },
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    backgroundColor: theme.palette.primary.main,
                  },
                }}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <BusinessIcon sx={{ 
                  mr: 0.5, 
                  fontSize: 18,
                  color: enhancedViewEnabled ? theme.palette.primary.main : 'inherit',
                  transition: 'all 0.3s ease',
                }} />
                <Typography variant="body2" sx={{
                  color: enhancedViewEnabled ? theme.palette.primary.main : 'inherit',
                  fontWeight: enhancedViewEnabled ? 600 : 400,
                  transition: 'all 0.3s ease',
                }}>
                  Enhanced Financial View
                </Typography>
              </Box>
            }
            sx={{
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
              }
            }}
          />
        </AnimatedBox>
      )}
      
      {/* Main Content */}
      {vectorTable ? (
        <AnimatedBox 
          style={contentAnimation}
          sx={{ height: 'calc(100vh - 240px)', minHeight: '500px' }}
        >
          {enhancedViewEnabled ? (
            <AnimatedBox
              style={useSpring({
                opacity: 1,
                from: { opacity: 0 },
                config: { tension: 280, friction: 60 }
              })}
            >
              <FinancialVisualizationExample />
            </AnimatedBox>
          ) : (
            <AnimatedBox
              style={useSpring({
                opacity: animationsVisible ? 1 : 0,
                transform: animationsVisible ? 'scale(1)' : 'scale(0.98)',
                config: { tension: 280, friction: 60 }
              })}
            >
              <VectorExplorer 
                vectorTable={vectorTable}
                onVectorSelected={handleVectorSelected}
              />
            </AnimatedBox>
          )}
        </AnimatedBox>
      ) : (
        <AnimatedGrid 
          container 
          spacing={3}
          style={contentAnimation}
        >
          {/* Recent Vector Tables */}
          <AnimatedGrid 
            item 
            xs={12} 
            md={8}
            style={useSpring({
              opacity: animationsVisible ? 1 : 0,
              transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
              delay: 250,
              config: { tension: 280, friction: 60 }
            })}
          >
            <AnimatedCard 
              style={useSpring({
                transform: animationsVisible ? 'scale(1)' : 'scale(0.98)',
                boxShadow: animationsVisible ? '0 4px 20px rgba(0, 0, 0, 0.08)' : '0 0px 0px rgba(0, 0, 0, 0)',
                config: { tension: 280, friction: 60 }
              })}
              sx={{ 
                height: '100%',
                borderRadius: 2,
                transition: 'all 0.3s ease',
                overflow: 'hidden',
              }}
            >
              <CardContent sx={{ p: 0 }}>
                <AnimatedBox 
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    delay: 300,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ borderBottom: 1, borderColor: 'divider' }}
                >
                  <Tabs 
                    value={tabValue} 
                    onChange={handleTabChange}
                    sx={{ 
                      px: 2,
                      '& .MuiTabs-indicator': {
                        height: 3,
                        borderRadius: '3px 3px 0 0',
                        transition: 'all 0.3s ease',
                      },
                      '& .MuiTab-root': {
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          color: theme.palette.primary.main,
                        },
                        '&.Mui-selected': {
                          fontWeight: 600,
                          color: theme.palette.primary.main,
                        }
                      }
                    }}
                  >
                    <Tab label="Recent Vector Tables" />
                    <Tab label="Search All Vectors" />
                  </Tabs>
                </AnimatedBox>
                
                <TabPanel value={tabValue} index={0}>
                  <AnimatedBox 
                    style={useSpring({
                      opacity: animationsVisible ? 1 : 0,
                      transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                      delay: 350,
                      config: { tension: 280, friction: 60 }
                    })}
                    sx={{ p: 3 }}
                  >
                    <AnimatedTypography 
                      variant="subtitle1" 
                      component={HumanText}
                      gutterBottom
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                        delay: 400,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{
                        fontWeight: 600,
                        color: theme.palette.text.primary,
                      }}
                    >
                      Recently Used Vector Tables
                    </AnimatedTypography>
                    
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      {cardTrail.map((style, index) => {
                        const table = recentTables[index];
                        return (
                          <Grid item xs={12} sm={6} md={4} key={table}>
                            <AnimatedPaper
                              style={style}
                              elevation={0}
                              sx={{
                                p: 2,
                                border: '1px solid',
                                borderColor: alpha(theme.palette.divider, 0.5),
                                borderRadius: 2,
                                height: '100%',
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                overflow: 'hidden',
                                '&:hover': {
                                  borderColor: theme.palette.primary.main,
                                  transform: 'translateY(-4px)',
                                  boxShadow: '0 8px 16px rgba(0, 0, 0, 0.1)',
                                  '&::after': {
                                    opacity: 1,
                                  }
                                },
                                '&::after': {
                                  content: '""',
                                  position: 'absolute',
                                  top: 0,
                                  left: 0,
                                  width: '100%',
                                  height: '4px',
                                  background: theme => `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
                                  opacity: 0,
                                  transition: 'opacity 0.3s ease',
                                },
                              }}
                              onClick={() => navigate(`/vector-exploration/${table}`)}
                            >
                              <AnimatedBox 
                                style={useSpring({
                                  opacity: animationsVisible ? 1 : 0,
                                  transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                                  delay: 450 + (index * 50),
                                  config: { tension: 280, friction: 60 }
                                })}
                                sx={{ display: 'flex', alignItems: 'center', mb: 1 }}
                              >
                                <StorageIcon 
                                  color="primary" 
                                  sx={{ 
                                    mr: 1, 
                                    opacity: 0.8,
                                    transition: 'all 0.3s ease',
                                    transform: 'rotate(0deg)',
                                    '&:hover': {
                                      transform: 'rotate(5deg)',
                                    }
                                  }} 
                                />
                                <HumanText 
                                  variant="subtitle2" 
                                  noWrap
                                  sx={{
                                    fontWeight: 600,
                                    transition: 'all 0.3s ease',
                                  }}
                                >
                                  {table}
                                </HumanText>
                              </AnimatedBox>
                              
                              <AnimatedTypography
                                variant="caption"
                                component={HumanText}
                                color="text.secondary"
                                style={useSpring({
                                  opacity: animationsVisible ? 1 : 0,
                                  transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                                  delay: 500 + (index * 50),
                                  config: { tension: 280, friction: 60 }
                                })}
                              >
                                Last accessed 2 days ago • ~10,000 vectors
                              </AnimatedTypography>
                              
                              <AnimatedBox 
                                style={useSpring({
                                  opacity: animationsVisible ? 1 : 0,
                                  transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                                  delay: 550 + (index * 50),
                                  config: { tension: 280, friction: 60 }
                                })}
                                sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}
                              >
                                <AnimatedButton
                                  size="small"
                                  endIcon={<OpenInNewIcon fontSize="small" />}
                                  style={useSpring({
                                    transform: animationsVisible ? 'scale(1)' : 'scale(0.9)',
                                    delay: 600 + (index * 50),
                                    config: { tension: 280, friction: 60 }
                                  })}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    navigate(`/vector-exploration/${table}`);
                                  }}
                                  sx={{
                                    borderRadius: 4,
                                    transition: 'all 0.3s ease',
                                    '&:hover': {
                                      transform: 'translateY(-2px)',
                                      boxShadow: '0 4px 8px rgba(0, 102, 179, 0.15)',
                                    }
                                  }}
                                >
                                  Explore
                                </AnimatedButton>
                              </AnimatedBox>
                            </AnimatedPaper>
                          </Grid>
                        );
                      })}
                    </Grid>
                    
                    <AnimatedBox 
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                        delay: 650,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ mt: 3, textAlign: 'center' }}
                    >
                      <AnimatedButton
                        variant="outlined"
                        style={useSpring({
                          transform: animationsVisible ? 'scale(1)' : 'scale(0.9)',
                          delay: 700,
                          config: { tension: 280, friction: 60 }
                        })}
                        onClick={() => setTabValue(1)}
                        sx={{
                          borderRadius: 8,
                          px: 3,
                          fontWeight: 500,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: '0 4px 8px rgba(0, 102, 179, 0.15)',
                          }
                        }}
                      >
                        Search Across All Vectors
                      </AnimatedButton>
                    </AnimatedBox>
                  </AnimatedBox>
                </TabPanel>
                
                <TabPanel value={tabValue} index={1}>
                  <AnimatedBox 
                    style={useSpring({
                      opacity: tabValue === 1 ? 1 : 0,
                      transform: tabValue === 1 ? 'scale(1)' : 'scale(0.98)',
                      config: { tension: 280, friction: 60 }
                    })}
                    sx={{ height: 'calc(100vh - 310px)', minHeight: '400px' }}
                  >
                    <VectorExplorer 
                      onVectorSelected={handleVectorSelected}
                    />
                  </AnimatedBox>
                </TabPanel>
              </CardContent>
            </AnimatedCard>
          </AnimatedGrid>
          
          {/* Vector Lifecycle */}
          <AnimatedGrid 
            item 
            xs={12} 
            md={4}
            style={lifecycleAnimation}
          >
            <AnimatedCard 
              style={useSpring({
                transform: animationsVisible ? 'scale(1)' : 'scale(0.98)',
                boxShadow: animationsVisible ? '0 4px 20px rgba(0, 0, 0, 0.08)' : '0 0px 0px rgba(0, 0, 0, 0)',
                config: { tension: 280, friction: 60 }
              })}
              sx={{ 
                height: '100%',
                borderRadius: 2,
                transition: 'all 0.3s ease',
                overflow: 'hidden',
                '&:hover': {
                  boxShadow: '0 8px 30px rgba(0, 0, 0, 0.12)',
                },
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '4px',
                  background: theme => `linear-gradient(90deg, ${theme.palette.info.main}, ${theme.palette.info.light})`,
                  opacity: 0,
                  transition: 'opacity 0.3s ease',
                },
                '&:hover::after': {
                  opacity: 1,
                },
              }}
            >
              <CardContent>
                <AnimatedTypography 
                  variant="h6" 
                  component={HumanText}
                  gutterBottom
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
                    delay: 250,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{
                    fontWeight: 600,
                    color: theme.palette.primary.main,
                  }}
                >
                  Vector Data Lifecycle
                </AnimatedTypography>
                
                <AnimatedTypography 
                  variant="body2" 
                  component={HumanText}
                  color="text.secondary" 
                  paragraph
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                    delay: 300,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  Follow the complete lifecycle of your data from tables to vectors and back again.
                </AnimatedTypography>
                
                <AnimatedBox 
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    delay: 350,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ py: 2 }}
                >
                  <Box sx={{ position: 'relative', mb: 4 }}>
                    <AnimatedBox 
                      style={useSpring({
                        height: animationsVisible ? '100%' : '0%',
                        opacity: animationsVisible ? 1 : 0,
                        delay: 400,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ 
                        position: 'absolute', 
                        top: 0, 
                        bottom: 0, 
                        left: '7px', 
                        width: '2px', 
                        bgcolor: alpha(theme.palette.primary.main, 0.2),
                        zIndex: 0,
                      }}
                    />
                    
                    {/* Step 1 */}
                    <AnimatedBox 
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                        delay: 450,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ position: 'relative', display: 'flex', alignItems: 'flex-start', mb: 3, zIndex: 1 }}
                    >
                      <AnimatedBox 
                        style={useSpring({
                          transform: animationsVisible ? 'scale(1)' : 'scale(0)',
                          delay: 500,
                          config: { tension: 280, friction: 20 }
                        })}
                        sx={{ 
                          width: 16, 
                          height: 16, 
                          borderRadius: '50%', 
                          bgcolor: alpha(theme.palette.primary.main, 0.2),
                          border: '2px solid',
                          borderColor: theme.palette.primary.main,
                          mr: 2,
                          mt: 0.5,
                          transition: 'all 0.3s ease',
                        }}
                      />
                      <Box>
                        <AnimatedTypography 
                          variant="subtitle2" 
                          component={HumanText}
                          gutterBottom
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            delay: 550,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          1. Schema and Table Exploration
                        </AnimatedTypography>
                        <AnimatedTypography 
                          variant="body2"
                          component={HumanText} 
                          color="text.secondary"
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            delay: 600,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          Discover and explore your HANA schemas and tables
                        </AnimatedTypography>
                        <AnimatedButton
                          size="small"
                          component={RouterLink}
                          to="/vector-creation"
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                            delay: 650,
                            config: { tension: 280, friction: 60 }
                          })}
                          sx={{ 
                            mt: 1,
                            borderRadius: 4,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                              transform: 'translateY(-2px)',
                              boxShadow: '0 2px 8px rgba(0, 102, 179, 0.15)',
                            }
                          }}
                        >
                          Go to Explorer
                        </AnimatedButton>
                      </Box>
                    </AnimatedBox>
                    
                    {/* Step 2 */}
                    <AnimatedBox 
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                        delay: 700,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ position: 'relative', display: 'flex', alignItems: 'flex-start', mb: 3, zIndex: 1 }}
                    >
                      <AnimatedBox 
                        style={useSpring({
                          transform: animationsVisible ? 'scale(1)' : 'scale(0)',
                          delay: 750,
                          config: { tension: 280, friction: 20 }
                        })}
                        sx={{ 
                          width: 16, 
                          height: 16, 
                          borderRadius: '50%', 
                          bgcolor: alpha(theme.palette.primary.main, 0.2),
                          border: '2px solid',
                          borderColor: theme.palette.primary.main,
                          mr: 2,
                          mt: 0.5,
                          transition: 'all 0.3s ease',
                        }}
                      />
                      <Box>
                        <AnimatedTypography 
                          variant="subtitle2" 
                          component={HumanText}
                          gutterBottom
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            delay: 800,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          2. Configure Transformations
                        </AnimatedTypography>
                        <AnimatedTypography 
                          variant="body2" 
                          component={HumanText}
                          color="text.secondary"
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            delay: 850,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          Define how to transform your relational data for vectorization
                        </AnimatedTypography>
                        <AnimatedButton
                          size="small"
                          component={RouterLink}
                          to="/vector-creation"
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                            delay: 900,
                            config: { tension: 280, friction: 60 }
                          })}
                          sx={{ 
                            mt: 1,
                            borderRadius: 4,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                              transform: 'translateY(-2px)',
                              boxShadow: '0 2px 8px rgba(0, 102, 179, 0.15)',
                            }
                          }}
                        >
                          Configure
                        </AnimatedButton>
                      </Box>
                    </AnimatedBox>
                    
                    {/* Step 3 */}
                    <AnimatedBox 
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                        delay: 950,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ position: 'relative', display: 'flex', alignItems: 'flex-start', mb: 3, zIndex: 1 }}
                    >
                      <AnimatedBox 
                        style={useSpring({
                          transform: animationsVisible ? 'scale(1)' : 'scale(0)',
                          delay: 1000,
                          config: { tension: 280, friction: 20 }
                        })}
                        sx={{ 
                          width: 16, 
                          height: 16, 
                          borderRadius: '50%', 
                          bgcolor: alpha(theme.palette.primary.main, 0.2),
                          border: '2px solid',
                          borderColor: theme.palette.primary.main,
                          mr: 2,
                          mt: 0.5,
                          transition: 'all 0.3s ease',
                        }}
                      />
                      <Box>
                        <AnimatedTypography 
                          variant="subtitle2" 
                          component={HumanText}
                          gutterBottom
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            delay: 1050,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          3. Create Vector Embeddings
                        </AnimatedTypography>
                        <AnimatedTypography 
                          variant="body2" 
                          component={HumanText}
                          color="text.secondary"
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            delay: 1100,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          Generate vector embeddings using HANA's native capabilities
                        </AnimatedTypography>
                        <AnimatedButton
                          size="small"
                          component={RouterLink}
                          to="/vector-creation"
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                            delay: 1150,
                            config: { tension: 280, friction: 60 }
                          })}
                          sx={{ 
                            mt: 1,
                            borderRadius: 4,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                              transform: 'translateY(-2px)',
                              boxShadow: '0 2px 8px rgba(0, 102, 179, 0.15)',
                            }
                          }}
                        >
                          Create Vectors
                        </AnimatedButton>
                      </Box>
                    </AnimatedBox>
                    
                    {/* Step 4 (Current) */}
                    <AnimatedBox 
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                        delay: 1200,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ position: 'relative', display: 'flex', alignItems: 'flex-start', zIndex: 1 }}
                    >
                      <AnimatedBox 
                        style={useSpring({
                          transform: animationsVisible ? 'scale(1) rotate(0deg)' : 'scale(0) rotate(180deg)',
                          boxShadow: animationsVisible 
                            ? `0 0 0 4px ${alpha(theme.palette.primary.main, 0.2)}`
                            : `0 0 0 0px ${alpha(theme.palette.primary.main, 0)}`,
                          delay: 1250,
                          config: { tension: 280, friction: 20 }
                        })}
                        sx={{ 
                          width: 16, 
                          height: 16, 
                          borderRadius: '50%', 
                          bgcolor: theme.palette.primary.main,
                          mr: 2,
                          mt: 0.5,
                          animation: 'pulse 2s ease-in-out infinite',
                          '@keyframes pulse': {
                            '0%': { boxShadow: `0 0 0 0 ${alpha(theme.palette.primary.main, 0.4)}` },
                            '70%': { boxShadow: `0 0 0 6px ${alpha(theme.palette.primary.main, 0)}` },
                            '100%': { boxShadow: `0 0 0 0 ${alpha(theme.palette.primary.main, 0)}` }
                          }
                        }}
                      />
                      <Box>
                        <AnimatedTypography 
                          variant="subtitle2" 
                          component={HumanText}
                          gutterBottom
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            color: theme.palette.primary.main,
                            fontWeight: 600,
                            delay: 1300,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          4. Explore Vectors
                        </AnimatedTypography>
                        <AnimatedTypography 
                          variant="body2" 
                          component={HumanText}
                          color="text.secondary"
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                            delay: 1350,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          Search, visualize, and analyze your vector embeddings
                        </AnimatedTypography>
                        <AnimatedBox
                          style={useSpring({
                            opacity: animationsVisible ? 1 : 0,
                            transform: animationsVisible ? 'translateY(0) scale(1)' : 'translateY(5px) scale(0.9)',
                            delay: 1400,
                            config: { tension: 280, friction: 20 }
                          })}
                        >
                          <HumanText 
                            variant="caption" 
                            sx={{ 
                              display: 'inline-block', 
                              bgcolor: alpha(theme.palette.success.main, 0.1),
                              color: theme.palette.success.main,
                              px: 1,
                              py: 0.5,
                              borderRadius: 1,
                              mt: 1,
                              fontWeight: 600,
                              boxShadow: `0 2px 8px ${alpha(theme.palette.success.main, 0.2)}`,
                              animation: 'pulse 2s ease-in-out infinite',
                              '@keyframes pulse': {
                                '0%': { boxShadow: `0 2px 8px ${alpha(theme.palette.success.main, 0.2)}` },
                                '50%': { boxShadow: `0 2px 12px ${alpha(theme.palette.success.main, 0.4)}` },
                                '100%': { boxShadow: `0 2px 8px ${alpha(theme.palette.success.main, 0.2)}` }
                              }
                            }}
                          >
                            You are here
                          </HumanText>
                        </AnimatedBox>
                      </Box>
                    </AnimatedBox>
                  </Box>
                </AnimatedBox>
                
                <AnimatedAlert 
                  severity="info" 
                  variant="outlined"
                  icon={<InfoIcon />}
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                    delay: 1450,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ 
                    mt: 2,
                    borderRadius: 2,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                    }
                  }}
                >
                  <AnimatedTypography 
                    variant="body2" 
                    component={HumanText}
                    style={useSpring({
                      opacity: animationsVisible ? 1 : 0,
                      delay: 1500,
                      config: { tension: 280, friction: 60 }
                    })}
                  >
                    The Vector Explorer allows you to search and visualize vectors created from your relational data.
                  </AnimatedTypography>
                </AnimatedAlert>
              </CardContent>
            </AnimatedCard>
          </AnimatedGrid>
        </AnimatedGrid>
      )}
    </Container>
  );
};

// Helper component for tabs
function TabPanel(props: { children?: React.ReactNode; index: number; value: number }) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`vector-explorer-tabpanel-${index}`}
      aria-labelledby={`vector-explorer-tab-${index}`}
      {...other}
    >
      {value === index && children}
    </div>
  );
}

export default VectorExploration;