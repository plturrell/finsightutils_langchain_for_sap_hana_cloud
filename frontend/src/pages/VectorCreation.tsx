import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Breadcrumbs, 
  Link, 
  Paper, 
  Stepper, 
  Step, 
  StepLabel,
  Button,
  Alert,
  Grid,
  Card,
  CardContent,
  CardMedia,
  useTheme,
  alpha,
} from '@mui/material';
import { 
  Home as HomeIcon, 
  NavigateNext as NavigateNextIcon,
  Storage as StorageIcon,
  Transform as TransformIcon,
  BlurOn as BlurOnIcon,
  Search as SearchIcon,
} from '@mui/icons-material';
import { useLocation, useNavigate } from 'react-router-dom';
import { useSpring, useTrail, useChain, animated, useSpringRef } from 'react-spring';
import VectorCreator from '../components/VectorCreator';
import SchemaExplorer from '../components/SchemaExplorer';
import TransformationConfigurator from '../components/TransformationConfigurator';
import VectorVisualization from '../components/VectorVisualization';
import HumanText from '../components/HumanText';
import { dataPipelineService } from '../api/services';

// Animated components
const AnimatedBox = animated(Box);
const AnimatedContainer = animated(Container);
const AnimatedTypography = animated(Typography);
const AnimatedHumanText = animated(HumanText);
const AnimatedPaper = animated(Paper);
const AnimatedBreadcrumbs = animated(Breadcrumbs);
const AnimatedCard = animated(Card);
const AnimatedGrid = animated(Grid);

const VectorCreationPage: React.FC = () => {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  
  // State
  const [activeStep, setActiveStep] = useState<number>(0);
  const [pipelineId, setPipelineId] = useState<string>('');
  const [sourceId, setSourceId] = useState<string>('');
  const [tableName, setTableName] = useState<string>('');
  const [schemaName, setSchemaName] = useState<string>('');
  const [transformationConfig, setTransformationConfig] = useState<any>(null);
  const [vectorId, setVectorId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [animationsVisible, setAnimationsVisible] = useState<boolean>(false);
  
  // Animation spring refs for sequence chaining
  const breadcrumbsSpringRef = useSpringRef();
  const headerSpringRef = useSpringRef();
  const cardsSpringRef = useSpringRef();
  const stepperSpringRef = useSpringRef();
  const contentSpringRef = useSpringRef();
  const navigationSpringRef = useSpringRef();
  
  // Animation springs
  const breadcrumbsAnimation = useSpring({
    ref: breadcrumbsSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const headerAnimation = useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const stepperAnimation = useSpring({
    ref: stepperSpringRef,
    from: { opacity: 0, transform: 'scale(0.95)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'scale(1)' : 'scale(0.95)' },
    config: { tension: 280, friction: 60 }
  });
  
  const contentAnimation = useSpring({
    ref: contentSpringRef,
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
    config: { tension: 280, friction: 60 }
  });
  
  const navigationAnimation = useSpring({
    ref: navigationSpringRef,
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Process cards animation trail
  const cardTrail = useTrail(4, {
    ref: cardsSpringRef,
    from: { opacity: 0, transform: 'translateY(40px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(40px)' },
    config: { mass: 1, tension: 280, friction: 60 }
  });
  
  // Chain animations sequence
  useChain(
    animationsVisible 
      ? [breadcrumbsSpringRef, headerSpringRef, cardsSpringRef, stepperSpringRef, contentSpringRef, navigationSpringRef] 
      : [navigationSpringRef, contentSpringRef, stepperSpringRef, cardsSpringRef, headerSpringRef, breadcrumbsSpringRef],
    animationsVisible 
      ? [0, 0.1, 0.2, 0.4, 0.5, 0.6] 
      : [0, 0.1, 0.2, 0.3, 0.4, 0.5]
  );
  
  // Define steps
  const steps = [
    'Select Data Source',
    'Configure Transformation',
    'Create Vectors',
    'Explore Vectors'
  ];
  
  // Initialize pipeline on component mount
  useEffect(() => {
    const initializePipeline = async () => {
      try {
        const response = await dataPipelineService.createPipeline({});
        if (response.data && response.data.pipeline_id) {
          setPipelineId(response.data.pipeline_id);
        }
      } catch (error) {
        console.error('Failed to initialize pipeline:', error);
        setError('Failed to initialize data pipeline. Please try again.');
      }
    };
    
    initializePipeline();
  }, []);
  
  // Trigger animations on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);
  
  // Handle step completion
  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };
  
  // Handle data source selection
  const handleDataSourceSelected = (sourceId: string, tableName: string, schemaName: string) => {
    setSourceId(sourceId);
    setTableName(tableName);
    setSchemaName(schemaName);
    handleNext();
  };
  
  // Handle transformation configuration
  const handleTransformationConfigured = (config: any) => {
    setTransformationConfig(config);
    handleNext();
  };
  
  // Handle vector creation
  const handleVectorCreated = (vectorId: string) => {
    setVectorId(vectorId);
    handleNext();
  };
  
  // Render step content
  const getStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <SchemaExplorer
            pipelineId={pipelineId}
            onSourceSelected={handleDataSourceSelected}
          />
        );
      case 1:
        return (
          <TransformationConfigurator
            pipelineId={pipelineId}
            sourceId={sourceId}
            tableName={tableName}
            schemaName={schemaName}
            onConfigurationComplete={handleTransformationConfigured}
          />
        );
      case 2:
        return (
          <VectorCreator
            pipelineId={pipelineId}
            sourceId={sourceId}
            tableName={tableName}
            schemaName={schemaName}
            onVectorCreated={handleVectorCreated}
          />
        );
      case 3:
        return (
          <Box>
            <HumanText variant="h6" gutterBottom>
              Vector Exploration
            </HumanText>
            <HumanText variant="body2" color="text.secondary" paragraph>
              Explore your vector embeddings and perform similarity searches.
            </HumanText>
            <VectorVisualization
              tableName={`VECTOR_${vectorId.replace(/-/g, '_')}`}
              maxPoints={500}
            />
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };
  
  // Render page
  return (
    <AnimatedContainer maxWidth="xl" sx={{ py: 4 }}>
      {/* Breadcrumbs */}
      <AnimatedBreadcrumbs 
        separator={<NavigateNextIcon fontSize="small" />} 
        aria-label="breadcrumb"
        sx={{ mb: 3 }}
        style={breadcrumbsAnimation}
      >
        <Link
          underline="hover"
          sx={{ 
            display: 'flex', 
            alignItems: 'center',
            transition: 'color 0.3s ease',
            '&:hover': {
              color: theme.palette.primary.main,
            }
          }}
          color="inherit"
          href="/"
        >
          <HomeIcon sx={{ mr: 0.5 }} fontSize="inherit" />
          Home
        </Link>
        <Typography 
          color="text.primary" 
          sx={{ 
            display: 'flex', 
            alignItems: 'center',
            background: `linear-gradient(90deg, ${theme.palette.primary.main} 0%, ${theme.palette.text.primary} 100%)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textFillColor: 'transparent',
          }}
        >
          <BlurOnIcon sx={{ mr: 0.5 }} fontSize="inherit" />
          Vector Creation
        </Typography>
      </AnimatedBreadcrumbs>
      
      {/* Page Header */}
      <AnimatedBox sx={{ mb: 4 }} style={headerAnimation}>
        <AnimatedHumanText 
          variant="h4" 
          gutterBottom
          sx={{
            fontWeight: 700,
            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            letterSpacing: '0.02em',
          }}
        >
          Data to Vector Lifecycle
        </AnimatedHumanText>
        <AnimatedHumanText 
          variant="body1" 
          color="text.secondary"
          sx={{
            maxWidth: '800px',
            lineHeight: 1.6,
          }}
        >
          Transform your relational data from SAP HANA into vector embeddings for semantic search and AI applications.
        </AnimatedHumanText>
      </AnimatedBox>
      
      {/* Process Overview */}
      {activeStep === 0 && (
        <AnimatedGrid 
          container 
          spacing={3} 
          sx={{ mb: 4 }}
          style={cardsSpringRef.current}
        >
          {cardTrail.map((style, index) => {
            const cardConfigs = [
              {
                title: '1. Data Discovery',
                icon: <StorageIcon sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontSize: 64,
                  color: theme.palette.primary.main,
                  opacity: 0.8
                }} />,
                description: 'Explore your SAP HANA schemas and tables to find the data you want to vectorize.',
                color: alpha(theme.palette.primary.main, 0.05),
                iconColor: theme.palette.primary.main,
              },
              {
                title: '2. Transformation',
                icon: <TransformIcon sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontSize: 64,
                  color: theme.palette.warning.main,
                  opacity: 0.8
                }} />,
                description: 'Configure how your relational data will be transformed and prepared for vectorization.',
                color: alpha(theme.palette.warning.main, 0.05),
                iconColor: theme.palette.warning.main,
              },
              {
                title: '3. Vectorization',
                icon: <BlurOnIcon sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontSize: 64,
                  color: theme.palette.success.main,
                  opacity: 0.8
                }} />,
                description: 'Create high-dimensional vector embeddings from your transformed data using ML models.',
                color: alpha(theme.palette.success.main, 0.05),
                iconColor: theme.palette.success.main,
              },
              {
                title: '4. Exploration',
                icon: <SearchIcon sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontSize: 64,
                  color: theme.palette.info.main,
                  opacity: 0.8
                }} />,
                description: 'Visualize and explore your vector embeddings, and perform semantic searches.',
                color: alpha(theme.palette.info.main, 0.05),
                iconColor: theme.palette.info.main,
              }
            ];
            
            const card = cardConfigs[index];
            
            return (
              <Grid item xs={12} md={3} key={index}>
                <AnimatedCard 
                  elevation={1}
                  style={style}
                  sx={{ 
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    bgcolor: card.color,
                    transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
                    '&:hover': {
                      transform: 'translateY(-6px)',
                      boxShadow: `0 14px 28px rgba(0,0,0,0.1), 0 10px 10px rgba(0,0,0,0.08)`,
                      '& .card-icon': {
                        transform: 'translate(-50%, -50%) scale(1.05)',
                        opacity: 1,
                      }
                    }
                  }}
                >
                  <CardMedia
                    component="div"
                    sx={{
                      pt: '40%',
                      position: 'relative',
                      overflow: 'hidden',
                      '&:after': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(to bottom, rgba(0,0,0,0), rgba(0,0,0,0.1))',
                      }
                    }}
                  >
                    <Box className="card-icon" sx={{ transition: 'all 0.3s ease-out' }}>
                      {card.icon}
                    </Box>
                  </CardMedia>
                  <CardContent sx={{ flexGrow: 1 }}>
                    <AnimatedHumanText 
                      gutterBottom 
                      variant="h6" 
                      component="h2"
                      sx={{
                        background: `linear-gradient(90deg, ${card.iconColor} 0%, ${theme.palette.text.primary} 80%)`,
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text',
                        textFillColor: 'transparent',
                        fontWeight: 600,
                      }}
                    >
                      {card.title}
                    </AnimatedHumanText>
                    <AnimatedHumanText variant="body2" color="text.secondary">
                      {card.description}
                    </AnimatedHumanText>
                  </CardContent>
                </AnimatedCard>
              </Grid>
            );
          })}
        </AnimatedGrid>
      )}
      
      {/* Error Alert */}
      {error && (
        <animated.div style={contentAnimation}>
          <Alert 
            severity="error" 
            sx={{ 
              mb: 3,
              borderRadius: '8px',
              position: 'relative',
              overflow: 'hidden',
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '4px',
                background: 'linear-gradient(90deg, #f44336, #ff9800)',
                animation: 'shimmer 2s infinite linear',
              },
              '@keyframes shimmer': {
                '0%': {
                  backgroundPosition: '-100% 0',
                },
                '100%': {
                  backgroundPosition: '100% 0',
                },
              },
            }}
          >
            {error}
          </Alert>
        </animated.div>
      )}
      
      {/* Stepper */}
      <AnimatedPaper 
        sx={{ 
          p: 3, 
          mb: 3,
          borderRadius: '12px',
          background: `linear-gradient(145deg, ${alpha(theme.palette.background.paper, 0.8)}, ${theme.palette.background.paper})`,
          boxShadow: '0px 3px 15px rgba(0,0,0,0.05)',
        }}
        style={stepperAnimation}
      >
        <Stepper 
          activeStep={activeStep} 
          alternativeLabel
          sx={{
            '& .MuiStepLabel-root': {
              transition: 'transform 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)'
              }
            },
            '& .MuiStepIcon-root': {
              transition: 'all 0.3s ease',
              '&.Mui-active': {
                filter: 'drop-shadow(0 0 3px rgba(25, 118, 210, 0.5))',
                transform: 'scale(1.1)',
              },
              '&.Mui-completed': {
                color: theme.palette.success.main,
              }
            }
          }}
        >
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>
                <Typography 
                  sx={{ 
                    fontWeight: activeStep === steps.indexOf(label) ? 600 : 400,
                    transition: 'all 0.3s ease',
                    color: activeStep === steps.indexOf(label) ? theme.palette.primary.main : 'inherit',
                  }}
                >
                  {label}
                </Typography>
              </StepLabel>
            </Step>
          ))}
        </Stepper>
      </AnimatedPaper>
      
      {/* Step Content */}
      <AnimatedBox sx={{ mt: 4, mb: 4 }} style={contentAnimation}>
        {getStepContent(activeStep)}
      </AnimatedBox>
      
      {/* Navigation Buttons */}
      {activeStep !== 0 && activeStep !== 2 && activeStep !== 3 && (
        <AnimatedBox 
          sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}
          style={navigationAnimation}
        >
          <Button
            variant="outlined"
            onClick={handleBack}
            disabled={activeStep === 0}
            sx={{
              position: 'relative',
              overflow: 'hidden',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
              },
              '&:after': {
                content: '""',
                position: 'absolute',
                width: '100%',
                height: '100%',
                top: 0,
                left: 0,
                pointerEvents: 'none',
                background: 'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 60%)',
                transform: 'scale(0, 0)',
                opacity: 0,
                transition: 'transform 0.4s, opacity 0.3s',
              },
              '&:active:after': {
                transform: 'scale(2, 2)',
                opacity: 0,
                transition: '0s',
              },
            }}
          >
            Back
          </Button>
          <Button
            variant="contained"
            color="primary"
            onClick={handleNext}
            disabled={
              (activeStep === 0 && !sourceId) ||
              (activeStep === 1 && !transformationConfig) ||
              (activeStep === 2 && !vectorId)
            }
            sx={{
              position: 'relative',
              overflow: 'hidden',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
              },
              '&:after': {
                content: '""',
                position: 'absolute',
                width: '100%',
                height: '100%',
                top: 0,
                left: 0,
                pointerEvents: 'none',
                background: 'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 60%)',
                transform: 'scale(0, 0)',
                opacity: 0,
                transition: 'transform 0.4s, opacity 0.3s',
              },
              '&:active:after': {
                transform: 'scale(2, 2)',
                opacity: 0,
                transition: '0s',
              },
            }}
          >
            {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
          </Button>
        </AnimatedBox>
      )}
      
      {/* Final Step Buttons */}
      {activeStep === steps.length - 1 && (
        <AnimatedBox 
          sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}
          style={navigationAnimation}
        >
          <Button
            variant="contained"
            color="primary"
            onClick={() => navigate('/search')}
            sx={{ 
              mr: 2,
              position: 'relative',
              overflow: 'hidden',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
                background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
              },
              '&:after': {
                content: '""',
                position: 'absolute',
                width: '100%',
                height: '100%',
                top: 0,
                left: 0,
                pointerEvents: 'none',
                background: 'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 60%)',
                transform: 'scale(0, 0)',
                opacity: 0,
                transition: 'transform 0.4s, opacity 0.3s',
              },
              '&:active:after': {
                transform: 'scale(2, 2)',
                opacity: 0,
                transition: '0s',
              },
            }}
          >
            Go to Search
          </Button>
          <Button
            variant="outlined"
            onClick={() => setActiveStep(0)}
            sx={{
              position: 'relative',
              overflow: 'hidden',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: '0 5px 15px rgba(0,0,0,0.1)',
              },
              '&:after': {
                content: '""',
                position: 'absolute',
                width: '100%',
                height: '100%',
                top: 0,
                left: 0,
                pointerEvents: 'none',
                background: 'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 60%)',
                transform: 'scale(0, 0)',
                opacity: 0,
                transition: 'transform 0.4s, opacity 0.3s',
              },
              '&:active:after': {
                transform: 'scale(2, 2)',
                opacity: 0,
                transition: '0s',
              },
            }}
          >
            Start New Process
          </Button>
        </AnimatedBox>
      )}
    </AnimatedContainer>
  );
};

export default VectorCreationPage;