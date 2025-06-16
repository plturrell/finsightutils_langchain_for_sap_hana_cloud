import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Typography,
  Container,
  Breadcrumbs,
  Link,
  Alert,
  Paper,
  Divider,
  Card,
  CardContent,
  alpha,
  Button,
} from '@mui/material';
import {
  Home as HomeIcon,
  Storage as StorageIcon,
  Transform as TransformIcon,
  ModelTraining as ModelTrainingIcon,
  Tune as TuneIcon,
  LineAxis as LineAxisIcon,
  Loop as LoopIcon,
} from '@mui/icons-material';
import { useSpring, animated, useTrail, useChain, useSpringRef } from '@react-spring/web';
import Layout from '../components/Layout';
import DataPipelineVisualizer from '../components/DataPipelineVisualizer';
import HumanText from '../components/HumanText';
import { dataPipelineService } from '../api/services';

// Create animated versions of MUI components
const AnimatedBox = animated(Box);
const AnimatedTypography = animated(Typography);
const AnimatedGrid = animated(Grid);
const AnimatedCard = animated(Card);
const AnimatedAlert = animated(Alert);

const DataPipeline: React.FC = () => {
  const [dataPipelineAvailable, setDataPipelineAvailable] = useState<boolean>(false);
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
  const cardTrail = useTrail(6, {
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

  // Check if data pipeline module is available
  useEffect(() => {
    const checkDataPipelineStatus = async () => {
      try {
        const response = await dataPipelineService.getStatus();
        const data = response.data?.data;
        if (data) {
          setDataPipelineAvailable(data.available);
        } else {
          setDataPipelineAvailable(false);
        }
      } catch (err) {
        console.error('Error checking data pipeline status:', err);
        setError('Failed to check data pipeline module availability');
        setDataPipelineAvailable(false);
      } finally {
        setLoading(false);
      }
    };

    checkDataPipelineStatus();
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
              <StorageIcon sx={{ mr: 0.5 }} fontSize="small" />
              Data Pipeline
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
            HANA to Vectors Pipeline
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
            Visualize and interact with the complete data transformation process from HANA tables
            to vector embeddings and back. Explore the entire lifecycle of your data.
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
                title: 'HANA Integration',
                icon: <StorageIcon color="primary" />,
                color: '#1976d2',
                bgColor: '#e3f2fd',
                description: 'Connect directly to your SAP HANA Cloud tables and view the raw relational data that serves as the foundation for your vector embeddings.'
              },
              {
                title: 'Transformation Tracking',
                icon: <TransformIcon color="secondary" />,
                color: '#9c27b0',
                bgColor: '#f3e5f5',
                description: 'Track how data is transformed through each stage of the pipeline. Visualize the process of turning raw data into meaningful embeddings.'
              },
              {
                title: 'Vector Embeddings',
                icon: <ModelTrainingIcon color="success" />,
                color: '#2e7d32',
                bgColor: '#e8f5e9',
                description: 'See how your data is represented as vector embeddings. Understand the model used, dimensions, and the original text that was vectorized.'
              },
              {
                title: 'Transformation Rules',
                icon: <TuneIcon color="warning" />,
                color: '#ed6c02',
                bgColor: '#fff8e1',
                description: 'Define and visualize the rules that govern how your data is transformed. See how input columns map to output columns and understand the transformation logic.'
              },
              {
                title: 'Data Lineage',
                icon: <LineAxisIcon color="info" />,
                color: '#0288d1',
                bgColor: '#e0f7fa',
                description: 'Trace the lineage of your vector embeddings back to the source data. Understand how information flows and transforms through the pipeline.'
              },
              {
                title: 'Reverse Mapping',
                icon: <LoopIcon color="error" />,
                color: '#d32f2f',
                bgColor: '#fce4ec',
                description: 'Map vector embeddings back to source data. Find similar vectors and understand the relationships between them. Close the loop from vectors to tables.'
              }
            ][index];

            return (
              <AnimatedGrid 
                item 
                xs={12} 
                md={6} 
                lg={4}
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
                          backgroundColor: alpha(cardConfig.bgColor, 0.7),
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
                            backgroundColor: alpha(cardConfig.bgColor, 0.9),
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
                Checking data pipeline module availability...
              </AnimatedTypography>
            </AnimatedBox>
          ) : !dataPipelineAvailable ? (
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
                Data Pipeline Module Not Available
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
                The data pipeline module is not available in this deployment.
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
              <AnimatedAlert 
                severity="success" 
                style={useSpring({
                  opacity: animationsVisible ? 1 : 0,
                  transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)',
                  delay: 250,
                  config: { tension: 280, friction: 60 }
                })}
                sx={{ 
                  mb: 2,
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
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(5px)',
                    delay: 300,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  Data Pipeline Module Active
                </AnimatedTypography>
                <AnimatedTypography 
                  variant="body2"
                  style={useSpring({
                    opacity: animationsVisible ? 1 : 0,
                    transform: animationsVisible ? 'translateY(0)' : 'translateY(5px)',
                    delay: 350,
                    config: { tension: 280, friction: 60 }
                  })}
                >
                  The data pipeline module is active and ready to use. You can create new pipelines
                  and visualize your data transformation process.
                </AnimatedTypography>
              </AnimatedAlert>
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
            <DataPipelineVisualizer 
              defaultSchemaName="SYSTEM" 
              defaultTableName="TABLES" 
            />
          </AnimatedGrid>
        </AnimatedGrid>
      </Container>
    </Layout>
  );
};

export default DataPipeline;