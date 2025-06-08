import React, { useState, useEffect } from 'react';
import { Box, Container, Paper, Typography, Snackbar, Alert, Button } from '@mui/material';
import SimpleMode from './SimpleMode';
import SimpleSearch from './SimpleSearch';
import SimpleSettings from './SimpleSettings';
import Onboarding from './Onboarding';
import HumanText from './HumanText';
import { humanize } from '../utils/humanLanguage';

interface ExperienceManagerProps {
  onOpenAdvanced?: () => void;
  currentComponent?: string;
}

/**
 * ExperienceManager handles the switching between simple and advanced experiences
 * and provides a consistent interface for the simplified components.
 */
const ExperienceManager: React.FC<ExperienceManagerProps> = ({ 
  onOpenAdvanced,
  currentComponent = 'search'
}) => {
  // State for simple mode and onboarding
  const [simpleMode, setSimpleMode] = useState<boolean>(true);
  const [showSnackbar, setShowSnackbar] = useState<boolean>(false);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  const [showOnboarding, setShowOnboarding] = useState<boolean>(false);
  
  // Load user preference and check if first visit
  useEffect(() => {
    const savedPreference = localStorage.getItem('simpleMode');
    if (savedPreference !== null) {
      setSimpleMode(savedPreference === 'true');
    }
    
    // Check if this is the first visit
    const hasSeenOnboarding = localStorage.getItem('hasSeenOnboarding');
    if (hasSeenOnboarding === null && currentComponent === 'search') {
      // Show onboarding on first visit to search page
      setShowOnboarding(true);
    }
  }, [currentComponent]);
  
  // Save preference when it changes
  useEffect(() => {
    localStorage.setItem('simpleMode', simpleMode.toString());
  }, [simpleMode]);
  
  // Handle simple mode toggle
  const handleToggleSimpleMode = (enabled: boolean) => {
    setSimpleMode(enabled);
    setSnackbarMessage(
      enabled 
        ? 'Simple Experience enabled. Technical settings are now managed automatically.' 
        : 'Advanced Experience enabled. All technical settings are now visible.'
    );
    setShowSnackbar(true);
  };
  
  // Handle advanced option click
  const handleOpenAdvanced = () => {
    if (onOpenAdvanced) {
      onOpenAdvanced();
    }
  };
  
  // Render the current component based on mode
  const renderCurrentComponent = () => {
    switch (currentComponent) {
      case 'search':
        return (
          <SimpleSearch 
            simpleMode={simpleMode}
            onOpenAdvanced={handleOpenAdvanced}
          />
        );
      case 'settings':
        return (
          <SimpleSettings 
            onSave={() => {
              setSnackbarMessage('Settings saved successfully');
              setShowSnackbar(true);
            }}
          />
        );
      default:
        return (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <HumanText variant="h5">Component not found</HumanText>
          </Box>
        );
    }
  };
  
  // Handle onboarding completion
  const handleOnboardingComplete = () => {
    setShowOnboarding(false);
    localStorage.setItem('hasSeenOnboarding', 'true');
    setSnackbarMessage("Welcome to Knowledge Explorer! We've set up a simple experience to get you started.");
    setShowSnackbar(true);
  };
  
  // Handle manual onboarding open
  const handleOpenOnboarding = () => {
    setShowOnboarding(true);
  };

  return (
    <Box sx={{ pb: 6 }}>
      {/* Simple Mode Toggle Component */}
      <SimpleMode 
        isEnabled={simpleMode}
        onToggle={handleToggleSimpleMode}
        onOpenSettings={() => {
          if (onOpenAdvanced) {
            onOpenAdvanced();
          }
        }}
      />
      
      {/* Current Component Container */}
      <Container maxWidth="lg">
        <Paper 
          elevation={0}
          sx={{ 
            p: { xs: 2, sm: 3, md: 4 },
            borderRadius: 4,
            border: '1px solid',
            borderColor: 'divider',
            bgcolor: 'background.paper',
          }}
        >
          {renderCurrentComponent()}
          
          {/* Show tour button if on search page */}
          {currentComponent === 'search' && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <Button
                variant="outlined"
                size="small"
                onClick={handleOpenOnboarding}
                sx={{ borderRadius: 2 }}
              >
                <HumanText>Show Introduction Tour</HumanText>
              </Button>
            </Box>
          )}
        </Paper>
      </Container>
      
      {/* Onboarding Dialog */}
      <Onboarding
        open={showOnboarding}
        onClose={() => setShowOnboarding(false)}
        onComplete={handleOnboardingComplete}
      />
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={5000}
        onClose={() => setShowSnackbar(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setShowSnackbar(false)} 
          severity="success"
          variant="filled"
          sx={{ width: '100%', borderRadius: 2 }}
        >
          <HumanText>{snackbarMessage}</HumanText>
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ExperienceManager;