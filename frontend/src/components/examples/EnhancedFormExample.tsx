import React, { useState } from 'react';
import { Box, Grid, Typography, useTheme } from '@mui/material';
import {
  EnhancedPaper,
  EnhancedTextField,
  EnhancedSelect,
  EnhancedCheckbox,
  EnhancedRadio,
  EnhancedSwitch,
  EnhancedSlider,
  EnhancedFormControlLabel,
  EnhancedAutocomplete,
  EnhancedInputGroup,
  EnhancedButton,
  EnhancedTypography,
  EnhancedDivider,
  EnhancedGradientTypography
} from '../enhanced';
import { MenuItem } from '@mui/material';

/**
 * Example component demonstrating all enhanced form inputs with Apple-like animations
 */
const EnhancedFormExample: React.FC = () => {
  const theme = useTheme();
  
  // Form state
  const [formState, setFormState] = useState({
    name: '',
    email: '',
    password: '',
    description: '',
    category: '',
    priority: '',
    notifications: true,
    contactMethod: 'email',
    confidenceLevel: 70,
    tags: [] as string[],
    agreedToTerms: false
  });
  
  // Handle input changes
  const handleChange = (field: keyof typeof formState) => (
    event: React.ChangeEvent<HTMLInputElement | { value: unknown }>
  ) => {
    const value = event.target.value;
    setFormState({ ...formState, [field]: value });
  };
  
  // Handle checkbox and switch changes
  const handleBooleanChange = (field: keyof typeof formState) => (
    event: React.ChangeEvent<HTMLInputElement>, checked: boolean
  ) => {
    setFormState({ ...formState, [field]: checked });
  };
  
  // Handle slider change
  const handleSliderChange = (_event: Event, value: number | number[]) => {
    setFormState({ ...formState, confidenceLevel: value as number });
  };
  
  // Handle autocomplete change
  const handleTagsChange = (_event: React.SyntheticEvent, value: string[]) => {
    setFormState({ ...formState, tags: value });
  };
  
  // Form submission
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    console.log('Form submitted:', formState);
    // You would normally send this data to an API
    alert('Form submitted successfully! Check console for details.');
  };
  
  // Available tags for autocomplete
  const availableTags = [
    'React', 'TypeScript', 'Material UI', 'Animation', 'Form', 
    'Apple', 'Design', 'User Experience', 'Frontend', 'CSS',
    'Responsive', 'Mobile', 'Desktop', 'Web', 'Application'
  ];
  
  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <EnhancedGradientTypography 
        variant="h4" 
        sx={{ mb: 3, fontWeight: 600, textAlign: 'center' }}
      >
        Enhanced Form Inputs
      </EnhancedGradientTypography>
      
      <EnhancedTypography 
        variant="body1" 
        color="text.secondary" 
        sx={{ mb: 4, textAlign: 'center' }}
      >
        Experience Apple-inspired animations, micro-interactions, and subtle audio feedback
      </EnhancedTypography>
      
      <EnhancedPaper sx={{ p: 4, borderRadius: 3 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Text inputs section */}
            <Grid item xs={12}>
              <EnhancedTypography variant="h6" sx={{ mb: 2 }}>
                Text Inputs
              </EnhancedTypography>
              <EnhancedDivider sx={{ mb: 3 }} />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedTextField
                label="Name"
                value={formState.name}
                onChange={handleChange('name')}
                fullWidth
                clearable
                animationDelay={100}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedTextField
                label="Email"
                type="email"
                value={formState.email}
                onChange={handleChange('email')}
                fullWidth
                clearable
                animationDelay={150}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedTextField
                label="Password"
                type="password"
                value={formState.password}
                onChange={handleChange('password')}
                fullWidth
                animationDelay={200}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedTextField
                label="Description"
                value={formState.description}
                onChange={handleChange('description')}
                fullWidth
                multiline
                rows={4}
                clearable
                animationDelay={250}
              />
            </Grid>
            
            {/* Selection inputs section */}
            <Grid item xs={12}>
              <EnhancedTypography variant="h6" sx={{ mb: 2, mt: 2 }}>
                Selection Inputs
              </EnhancedTypography>
              <EnhancedDivider sx={{ mb: 3 }} />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedSelect
                label="Category"
                value={formState.category}
                onChange={handleChange('category')}
                fullWidth
                animationDelay={300}
              >
                <MenuItem value="">
                  <em>Select a category</em>
                </MenuItem>
                <MenuItem value="design">Design</MenuItem>
                <MenuItem value="development">Development</MenuItem>
                <MenuItem value="marketing">Marketing</MenuItem>
                <MenuItem value="business">Business</MenuItem>
              </EnhancedSelect>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedInputGroup
                label="Priority Level"
                description="Select the priority level for this item"
                animationDelay={350}
              >
                <Grid container spacing={1}>
                  <Grid item xs={4}>
                    <EnhancedFormControlLabel
                      control={
                        <EnhancedRadio
                          checked={formState.priority === 'low'}
                          onChange={() => setFormState({ ...formState, priority: 'low' })}
                        />
                      }
                      label="Low"
                      animationDelay={375}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <EnhancedFormControlLabel
                      control={
                        <EnhancedRadio
                          checked={formState.priority === 'medium'}
                          onChange={() => setFormState({ ...formState, priority: 'medium' })}
                        />
                      }
                      label="Medium"
                      animationDelay={400}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <EnhancedFormControlLabel
                      control={
                        <EnhancedRadio
                          checked={formState.priority === 'high'}
                          onChange={() => setFormState({ ...formState, priority: 'high' })}
                        />
                      }
                      label="High"
                      animationDelay={425}
                    />
                  </Grid>
                </Grid>
              </EnhancedInputGroup>
            </Grid>
            
            {/* Boolean inputs section */}
            <Grid item xs={12}>
              <EnhancedTypography variant="h6" sx={{ mb: 2, mt: 2 }}>
                Toggle Controls
              </EnhancedTypography>
              <EnhancedDivider sx={{ mb: 3 }} />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedFormControlLabel
                control={
                  <EnhancedSwitch
                    checked={formState.notifications}
                    onChange={handleBooleanChange('notifications')}
                  />
                }
                label="Enable Notifications"
                animationDelay={450}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedFormControlLabel
                control={
                  <EnhancedCheckbox
                    checked={formState.agreedToTerms}
                    onChange={handleBooleanChange('agreedToTerms')}
                  />
                }
                label="I agree to the terms and conditions"
                animationDelay={500}
              />
            </Grid>
            
            {/* Range inputs section */}
            <Grid item xs={12}>
              <EnhancedTypography variant="h6" sx={{ mb: 2, mt: 2 }}>
                Range & Complex Inputs
              </EnhancedTypography>
              <EnhancedDivider sx={{ mb: 3 }} />
            </Grid>
            
            <Grid item xs={12}>
              <EnhancedInputGroup
                label="Confidence Level"
                description="How confident are you in this submission?"
                animationDelay={550}
              >
                <Box sx={{ px: 2 }}>
                  <EnhancedSlider
                    value={formState.confidenceLevel}
                    onChange={handleSliderChange}
                    valueLabelDisplay="auto"
                    step={10}
                    marks
                    min={0}
                    max={100}
                    animationDelay={575}
                  />
                  <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    mt: 1
                  }}>
                    <Typography variant="caption" color="text.secondary">
                      Not confident
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Very confident
                    </Typography>
                  </Box>
                </Box>
              </EnhancedInputGroup>
            </Grid>
            
            <Grid item xs={12}>
              <EnhancedAutocomplete
                multiple
                label="Tags"
                options={availableTags}
                value={formState.tags}
                onChange={handleTagsChange}
                fullWidth
                animationDelay={625}
                renderInput={(params) => (
                  <EnhancedTextField
                    {...params}
                    label="Tags"
                    placeholder="Select or type tags"
                  />
                )}
              />
            </Grid>
            
            {/* Submit section */}
            <Grid item xs={12} sx={{ mt: 4, textAlign: 'center' }}>
              <EnhancedButton
                variant="contained"
                type="submit"
                size="large"
                sx={{
                  px: 5,
                  py: 1.5,
                  borderRadius: 2,
                  background: 'linear-gradient(90deg, #0066B3, #19B5FE)',
                  boxShadow: '0 4px 20px rgba(0, 102, 179, 0.25)',
                  '&:hover': {
                    boxShadow: '0 6px 25px rgba(0, 102, 179, 0.35)',
                  },
                }}
              >
                Submit Form
              </EnhancedButton>
            </Grid>
          </Grid>
        </form>
      </EnhancedPaper>
    </Box>
  );
};

export default EnhancedFormExample;