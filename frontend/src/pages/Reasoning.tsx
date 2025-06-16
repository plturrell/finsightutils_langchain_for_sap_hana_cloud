import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Typography,
  Paper,
  Container,
  Breadcrumbs,
  Link,
  Alert,
  Divider,
  Card,
  CardContent,
  Chip,
  alpha,
} from '@mui/material';
import { 
  Home as HomeIcon,
  InsertChart as InsertChartIcon,
  Psychology as PsychologyIcon,
  Visibility as VisibilityIcon,
  Bolt as BoltIcon,
  Check as CheckIcon,
} from '@mui/icons-material';
import { useSpring, animated, useTrail, useChain, useSpringRef } from '@react-spring/web';
import Layout from '../components/Layout';
import ReasoningVisualizer from '../components/ReasoningVisualizer';
import HumanText from '../components/HumanText';
import { reasoningService } from '../api/services';

// Create animated versions of MUI components
const AnimatedBox = animated(Box);
const AnimatedTypography = animated(Typography);
const AnimatedGrid = animated(Grid);
const AnimatedCard = animated(Card);
const AnimatedChip = animated(Chip);
const AnimatedAlert = animated(Alert);

const Reasoning: React.FC = () => {
  const [reasoningAvailable, setReasoningAvailable] = useState<boolean>(false);
  const [reasoningFeatures, setReasoningFeatures] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [animationsVisible, setAnimationsVisible] = useState(false);

  // Animation spring refs for chaining
  const headerSpringRef = useSpringRef();
  const cardsSpringRef = useSpringRef();
  const statusSpringRef = useSpringRef();
  const contentSpringRef = useSpringRef();

  // Header animation
  const headerAnimation = useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });

  // Status animation
  const statusAnimation = useSpring({
    ref: statusSpringRef,
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
    config: { tension: 280, friction: 60 }
  });

  // Content animation
  const contentAnimation = useSpring({
    ref: contentSpringRef,
    from: { opacity: 0, transform: 'translateY(40px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(40px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Feature cards trail animation
  const cardTrail = useTrail(4, {
    ref: cardsSpringRef,
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
    config: { mass: 1, tension: 280, friction: 60 }
  });

  // Chain animations in sequence
  useChain(
    animationsVisible 
      ? [headerSpringRef, cardsSpringRef, statusSpringRef, contentSpringRef] 
      : [contentSpringRef, statusSpringRef, cardsSpringRef, headerSpringRef],
    animationsVisible 
      ? [0, 0.2, 0.4, 0.6] 
      : [0, 0.1, 0.2, 0.3]
  );

  // Check if reasoning framework is available
  useEffect(() => {
    const checkReasoningStatus = async () => {
      try {
        const response = await reasoningService.getStatus();
        const data = response.data?.data;
        if (data) {
          setReasoningAvailable(data.available);
          setReasoningFeatures(data.features || {});
        } else {
          setReasoningAvailable(false);
        }
      } catch (err) {
        console.error('Error checking reasoning status:', err);
        setError('Failed to check reasoning framework availability');
        setReasoningAvailable(false);
      } finally {
        setLoading(false);
      }
    };

    checkReasoningStatus();
  }, []);

  // Trigger animations after initial load
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, 150);
    return () => clearTimeout(timer);
  }, []);

  return (
    <Layout>
      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* Breadcrumbs */}
        <AnimatedBox
          style={useSpring({
            opacity: animationsVisible ? 1 : 0,
            transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
            config: { tension: 280, friction: 60 }
          })}
        >
          <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 3 }}>
            <Link 
              underline="hover" 
              color="inherit" 
              href="/" 
              sx={{ 
                display: 'flex', 
                alignItems: 'center',
                transition: 'all 0.2s ease',
                '&:hover': {
                  color: theme => theme.palette.primary.main,
                }
              }}
            >
              <HomeIcon sx={{ mr: 0.5 }} fontSize="small" />
              Home
            </Link>
            <AnimatedTypography 
              color="text.primary" 
              style={useSpring({
                opacity: animationsVisible ? 1 : 0,
                transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)',
                delay: 200,
                config: { tension: 280, friction: 60 }
              })}
              sx={{ display: 'flex', alignItems: 'center' }}
            >
              <PsychologyIcon sx={{ mr: 0.5 }} fontSize="small" />
              Reasoning
            </AnimatedTypography>
          </Breadcrumbs>
        </AnimatedBox>

        {/* Header */}
        <AnimatedBox 
          style={headerAnimation}
          sx={{ mb: 4 }}
        >
          <AnimatedTypography 
            variant="h4" 
            component="h1" 
            gutterBottom 
            fontWeight={600}
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
            Reasoning Transparency
          </AnimatedTypography>
          <AnimatedTypography 
            variant="body1" 
            color="text.secondary" 
            gutterBottom
            style={useSpring({
              opacity: animationsVisible ? 1 : 0,
              transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
              delay: 150,
              config: { tension: 280, friction: 60 }
            })}
          >
            Visualize and interact with the transformation process from data to vector embeddings.
            Understand how the system reasons with your data and ensure information preservation.
          </AnimatedTypography>
        </AnimatedBox>

        {/* Error message */}
        {error && (
          <AnimatedAlert 
            severity="error" 
            style={useSpring({
              opacity: animationsVisible ? 1 : 0,
              transform: animationsVisible ? 'translateY(0)' : 'translateY(-10px)',
              config: { tension: 280, friction: 60 }
            })}
            sx={{ 
              mb: 4,
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

        {/* Feature highlight cards */}
        <AnimatedGrid 
          container 
          spacing={3} 
          sx={{ mb: 4 }}
        >
          {cardTrail.map((style, index) => {
            const cardConfig = [
              {
                title: 'Transparency',
                icon: <VisibilityIcon color="primary" />,
                color: '#1976d2',
                description: 'Visualize how the system transforms data into vectors and reasons with them. No more black box - see the entire process.'
              },
              {
                title: 'Interactivity',
                icon: <BoltIcon color="success" />,
                color: '#2e7d32',
                description: 'Guide and influence the transformation process to improve outcomes. Provide feedback and see how it affects reasoning quality.'
              },
              {
                title: 'Metrics',
                icon: <InsertChartIcon color="warning" />,
                color: '#ed6c02',
                description: 'Measure information preservation to ensure meaning is retained through the transformation process.'
              },
              {
                title: 'Validation',
                icon: <PsychologyIcon color="secondary" />,
                color: '#9c27b0',
                description: 'Validate reasoning quality using techniques from cognitive science. Identify and fix issues in the reasoning process.'
              }
            ][index];

            return (
              <AnimatedGrid 
                item 
                xs={12} 
                md={6} 
                lg={3}
                key={cardConfig.title}
                style={style}
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
                      transform: 'translateY(-4px)',
                      boxShadow: theme => `0 8px 24px ${alpha(cardConfig.color, 0.15)}`,
                    },
                    '&::after': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '4px',
                      background: theme => `linear-gradient(90deg, ${cardConfig.color}, ${alpha(cardConfig.color, 0.6)})`,
                      opacity: 0,
                      transition: 'opacity 0.3s ease',
                    },
                    '&:hover::after': {
                      opacity: 1,
                    },
                  }}
                >
                  <CardContent>
                    <AnimatedBox 
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                        delay: 200 + (index * 100),
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        mb: 2,
                      }}
                    >
                      <AnimatedBox 
                        style={useSpring({
                          transform: animationsVisible ? 'scale(1) rotate(0deg)' : 'scale(0.5) rotate(-45deg)',
                          config: { tension: 200, friction: 12 },
                          delay: 250 + (index * 100)
                        })}
                        sx={{ 
                          backgroundColor: alpha(cardConfig.color, 0.1),
                          borderRadius: '50%',
                          width: 40,
                          height: 40,
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center',
                          mr: 2,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'scale(1.1)',
                            backgroundColor: alpha(cardConfig.color, 0.15),
                          }
                        }}
                      >
                        {cardConfig.icon}
                      </AnimatedBox>
                      <AnimatedTypography 
                        variant="h6" 
                        component={HumanText}
                        style={useSpring({
                          opacity: animationsVisible ? 1 : 0,
                          transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                          delay: 300 + (index * 100),
                          config: { tension: 280, friction: 60 }
                        })}
                        sx={{ 
                          fontWeight: 600,
                          background: `linear-gradient(90deg, ${cardConfig.color} 0%, ${alpha(cardConfig.color, 0.7)} 100%)`,
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          transition: 'all 0.3s ease',
                        }}
                      >
                        {cardConfig.title}
                      </AnimatedTypography>
                    </AnimatedBox>
                    <AnimatedTypography 
                      variant="body2" 
                      component={HumanText}
                      color="text.secondary"
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                        delay: 350 + (index * 100),
                        config: { tension: 280, friction: 60 }
                      })}
                    >
                      {cardConfig.description}
                    </AnimatedTypography>
                  </CardContent>
                </AnimatedCard>
              </AnimatedGrid>
            );
          })}
        </AnimatedGrid>

        {/* Status message */}
        <AnimatedBox style={statusAnimation}>
          {loading ? (
            <AnimatedBox 
              style={useSpring({
                opacity: animationsVisible ? 1 : 0,
                transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                config: { tension: 280, friction: 60 }
              })}
              sx={{ py: 2 }}
            >
              <AnimatedTypography 
                variant="body2" 
                color="text.secondary"
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  config: { tension: 280, friction: 60 }
                })}
                sx={{
                  animation: 'pulse 1.5s ease-in-out infinite',
                  '@keyframes pulse': {
                    '0%': { opacity: 0.6 },
                    '50%': { opacity: 1 },
                    '100%': { opacity: 0.6 }
                  }
                }}
              >
                Checking reasoning framework availability...
              </AnimatedTypography>
            </AnimatedBox>
          ) : !reasoningAvailable ? (
            <AnimatedAlert 
              severity="warning" 
              style={useSpring({
                opacity: animationsVisible ? 1 : 0,
                transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                delay: 200,
                config: { tension: 280, friction: 60 }
              })}
              sx={{ 
                mb: 4,
                animation: 'slideIn 0.3s ease-out',
                '@keyframes slideIn': {
                  from: { opacity: 0, transform: 'translateY(10px)' },
                  to: { opacity: 1, transform: 'translateY(0)' }
                },
                borderRadius: 2,
                transition: 'all 0.3s ease',
                '&:hover': {
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                }
              }}
            >
              <AnimatedTypography 
                variant="subtitle1" 
                fontWeight={600} 
                gutterBottom
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(5px)',
                  delay: 250,
                  config: { tension: 280, friction: 60 }
                })}
              >
                Reasoning Framework Not Available
              </AnimatedTypography>
              <AnimatedTypography 
                variant="body2"
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(5px)',
                  delay: 300,
                  config: { tension: 280, friction: 60 }
                })}
              >
                The reasoning transparency framework is not available in this deployment.
                You'll see a mock interface to demonstrate the concept.
              </AnimatedTypography>
            </AnimatedAlert>
          ) : (
            <AnimatedBox 
              style={useSpring({
                opacity: animationsVisible ? 1 : 0,
                transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)',
                delay: 200,
                config: { tension: 280, friction: 60 }
              })}
              sx={{ mb: 4 }}
            >
              <AnimatedChip 
                label="Reasoning Framework Active" 
                color="success" 
                icon={<CheckIcon />}
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'scale(1)' : 'scale(0.9)',
                  delay: 250,
                  config: { tension: 280, friction: 20 }
                })}
                sx={{ 
                  mb: 2,
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: '0 4px 8px rgba(0, 102, 179, 0.15)',
                  }
                }} 
              />
              <AnimatedGrid 
                container 
                spacing={2}
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                  delay: 300,
                  config: { tension: 280, friction: 60 }
                })}
              >
                {Object.entries(reasoningFeatures).map(([feature, available], index) => (
                  <AnimatedGrid 
                    item 
                    key={feature}
                    style={useSpring({
                      opacity: animationsVisible ? 1 : 0,
                      transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                      delay: 350 + (index * 50),
                      config: { tension: 280, friction: 60 }
                    })}
                  >
                    <AnimatedChip 
                      label={feature.charAt(0).toUpperCase() + feature.slice(1)} 
                      color={available ? 'primary' : 'default'} 
                      variant={available ? 'filled' : 'outlined'}
                      size="small"
                      style={useSpring({
                        opacity: animationsVisible ? 1 : 0,
                        transform: animationsVisible ? 'scale(1)' : 'scale(0.9)',
                        delay: 400 + (index * 50),
                        config: { tension: 280, friction: 20 }
                      })}
                      sx={{
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          transform: 'translateY(-2px)',
                          boxShadow: available 
                            ? '0 2px 8px rgba(25, 118, 210, 0.15)'
                            : '0 2px 8px rgba(0, 0, 0, 0.08)',
                        }
                      }}
                    />
                  </AnimatedGrid>
                ))}
              </AnimatedGrid>
            </AnimatedBox>
          )}
        </AnimatedBox>

        {/* Main content */}
        <AnimatedGrid 
          container 
          spacing={3}
          style={contentAnimation}
        >
          <AnimatedGrid 
            item 
            xs={12}
            style={useSpring({
              opacity: animationsVisible ? 1 : 0,
              transform: animationsVisible ? 'scale(1)' : 'scale(0.98)',
              delay: 400,
              config: { tension: 280, friction: 60 }
            })}
          >
            <ReasoningVisualizer 
              tableName="EMBEDDINGS"
              initialQuery="How does SAP HANA Cloud handle vector embeddings for language models?"
            />
          </AnimatedGrid>
        </AnimatedGrid>
      </Container>
    </Layout>
  );
};

export default Reasoning;