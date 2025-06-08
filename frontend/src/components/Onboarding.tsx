import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Avatar,
  useTheme,
  Fade,
  Grow,
} from '@mui/material';
import {
  Close as CloseIcon,
  Search as SearchIcon,
  Psychology as PsychologyIcon,
  Lightbulb as LightbulbIcon,
  DataObject as DataObjectIcon,
  SvgIconComponent,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import HumanText from './HumanText';
import { humanize } from '../utils/humanLanguage';

interface OnboardingStep {
  title: string;
  description: string;
  icon: React.ReactElement;
  image?: string;
  actions?: Array<{
    label: string;
    onClick?: () => void;
    primary?: boolean;
  }>;
}

interface OnboardingProps {
  open: boolean;
  onClose: () => void;
  onComplete: () => void;
}

/**
 * Onboarding provides an engaging introduction to the application
 * with emotional design principles and clear benefits-focused messaging.
 */
const Onboarding: React.FC<OnboardingProps> = ({
  open,
  onClose,
  onComplete,
}) => {
  const theme = useTheme();
  const [activeStep, setActiveStep] = useState(0);
  
  // Define onboarding steps
  const steps: OnboardingStep[] = [
    {
      title: "Welcome to Knowledge Explorer",
      description: "Discover insights in your data through the power of meaning, not just keywords. We've designed this experience to be intuitive and simple while providing powerful capabilities.",
      icon: <PsychologyIcon fontSize="large" />,
      actions: [
        { label: "Get Started", onClick: () => setActiveStep(1), primary: true },
      ],
    },
    {
      title: "Find What Matters",
      description: "Ask questions in natural language and get relevant answers from your data. The system understands the meaning behind your questions, even if the exact words don't match.",
      icon: <SearchIcon fontSize="large" />,
      actions: [
        { label: "Back", onClick: () => setActiveStep(0) },
        { label: "Continue", onClick: () => setActiveStep(2), primary: true },
      ],
    },
    {
      title: "Understand Your Results",
      description: "See why certain information was returned and explore connections between concepts. Gain insights that would be impossible with traditional search.",
      icon: <LightbulbIcon fontSize="large" />,
      actions: [
        { label: "Back", onClick: () => setActiveStep(1) },
        { label: "Continue", onClick: () => setActiveStep(3), primary: true },
      ],
    },
    {
      title: "Simple by Default, Powerful When Needed",
      description: "We've designed the experience to be simple and focused, but all the advanced capabilities are there when you need them. Just click 'Show Technical Details' whenever you want to go deeper.",
      icon: <DataObjectIcon fontSize="large" />,
      actions: [
        { label: "Back", onClick: () => setActiveStep(2) },
        { label: "Get Started", onClick: onComplete, primary: true },
      ],
    },
  ];
  
  // Current step
  const currentStep = steps[activeStep];
  
  // Handle next step
  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      onComplete();
    } else {
      setActiveStep((prev) => prev + 1);
    }
  };
  
  // Handle previous step
  const handleBack = () => {
    setActiveStep((prev) => Math.max(0, prev - 1));
  };
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 4,
          overflow: 'hidden',
        }
      }}
    >
      {/* Header with progress indication */}
      <Box
        sx={{
          position: 'relative',
          bgcolor: 'primary.dark',
          color: 'primary.contrastText',
          py: 2,
          px: 3,
        }}
      >
        <IconButton
          onClick={onClose}
          sx={{
            position: 'absolute',
            right: 8,
            top: 8,
            color: 'white',
          }}
        >
          <CloseIcon />
        </IconButton>
        
        <Box sx={{ width: '90%', mx: 'auto', mt: 1 }}>
          <Stepper activeStep={activeStep} alternativeLabel>
            {steps.map((step, index) => (
              <Step key={index}>
                <StepLabel
                  StepIconProps={{
                    sx: {
                      color: index <= activeStep ? 'white' : 'rgba(255, 255, 255, 0.5)',
                    }
                  }}
                ></StepLabel>
              </Step>
            ))}
          </Stepper>
        </Box>
      </Box>
      
      <DialogContent
        sx={{
          p: 0,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 400,
          overflow: 'hidden',
        }}
      >
        <Box
          component={motion.div}
          key={activeStep}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
          sx={{ 
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            height: '100%',
            alignItems: 'center',
            px: { xs: 2, sm: 4 },
            py: { xs: 3, sm: 5 },
            flexGrow: 1,
          }}
        >
          {/* Left side - Icon */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              width: { xs: '100%', md: '30%' },
              mb: { xs: 3, md: 0 },
            }}
          >
            <Avatar
              sx={{
                bgcolor: 'primary.main',
                width: 80,
                height: 80,
                boxShadow: theme.shadows[4],
              }}
            >
              {currentStep.icon}
            </Avatar>
          </Box>
          
          {/* Right side - Content */}
          <Box
            sx={{
              width: { xs: '100%', md: '70%' },
              pl: { xs: 0, md: 4 },
            }}
          >
            <HumanText
              variant="h4"
              component="h2"
              gutterBottom
              sx={{ fontWeight: 600 }}
            >
              {currentStep.title}
            </HumanText>
            
            <HumanText
              variant="body1"
              paragraph
              sx={{ mb: 4, fontSize: '1.1rem' }}
            >
              {currentStep.description}
            </HumanText>
            
            {/* Action buttons */}
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'flex-end',
                gap: 2,
                mt: 2,
              }}
            >
              {currentStep.actions?.map((action, index) => (
                <Button
                  key={index}
                  variant={action.primary ? 'contained' : 'outlined'}
                  onClick={action.onClick}
                  size="large"
                  sx={{
                    px: 3,
                    py: 1,
                    borderRadius: 2,
                  }}
                >
                  <HumanText>{action.label}</HumanText>
                </Button>
              ))}
            </Box>
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default Onboarding;