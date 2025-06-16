import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Container,
  Grid,
  Divider,
  Tabs,
  Tab,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Slider,
  Switch,
  FormControlLabel,
  Chip,
  IconButton,
  Card,
  CardContent,
  Alert,
  Table,
  TableContainer,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Rating,
  useTheme
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Assessment as AssessmentIcon,
  Save as SaveIcon,
  Clear as ClearIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { animated, useSpring } from '@react-spring/web';
import { useAnimationContext } from '@finsightdev/ui-animations';
import { runPerformanceTests, PerformanceMetrics } from '../../utils/performanceTest';
import {
  EnhancedBox,
  EnhancedButton,
  EnhancedCard,
  EnhancedCardContent,
  EnhancedTextField,
  EnhancedSelect,
  EnhancedTable,
  EnhancedGradientTypography,
  EnhancedTypography,
  EnhancedDashboardCardGrid,
  EnhancedDashboardCard,
  EnhancedSearchResults,
  EnhancedAnimatedTableRow,
  EnhancedAnimatedTableCell,
  EnhancedTableSortHeader,
  EnhancedAnimatedTable,
  EnhancedExpandableTableRow,
  EnhancedSwitch,
  EnhancedSlider,
  EnhancedChip,
  EnhancedPaper,
  EnhancedFormControlLabel,
  EnhancedAlert
} from '../enhanced';
import { 
  measureAnimationPerformance, 
  PerformanceMetrics,
  createPerformanceReport,
  getPerformanceRating
} from '../../utils/animationPerformanceTest';
import { soundEffects } from '../../utils/soundEffects';

// Mock data for tests
const MOCK_SEARCH_RESULTS = Array(10).fill(0).map((_, i) => ({
  document: {
    page_content: `This is search result ${i + 1} with some content to display. It contains information that would be relevant to a search query.`,
    metadata: {
      title: `Result ${i + 1}`,
      score: 0.95 - (i * 0.03),
      date: new Date().toISOString(),
      source: 'database',
      type: i % 2 === 0 ? 'document' : 'article'
    }
  },
  score: 0.95 - (i * 0.03)
}));

const MOCK_DASHBOARD_CARDS = Array(12).fill(0).map((_, i) => ({
  id: `card-${i}`,
  title: `Card ${i + 1}`,
  content: `This is dashboard card ${i + 1} with some content.`,
  metric: Math.floor(Math.random() * 1000),
  change: (Math.random() * 20 - 10).toFixed(1) + '%',
  status: Math.random() > 0.5 ? 'positive' : 'negative'
}));

const MOCK_TABLE_DATA = Array(20).fill(0).map((_, i) => ({
  id: `row-${i}`,
  name: `Item ${i + 1}`,
  value: Math.floor(Math.random() * 1000),
  status: i % 3 === 0 ? 'active' : i % 3 === 1 ? 'pending' : 'inactive',
  date: new Date(Date.now() - Math.floor(Math.random() * 1000000000)).toISOString().split('T')[0],
  progress: Math.floor(Math.random() * 100)
}));

/**
 * Performance Test Case
 */
interface TestCase {
  id: string;
  name: string;
  description: string;
  component: React.ReactNode;
  triggerAnimation: () => void;
  duration?: number;
}

/**
 * Component for testing animation performance
 */
export const AnimationPerformanceTest: React.FC = () => {
  const theme = useTheme();
  const { animationsEnabled, soundEffectsEnabled } = useAnimationContext();
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const [testResults, setTestResults] = useState<Array<{name: string; metrics: PerformanceMetrics}>>([]);
  const [currentTest, setCurrentTest] = useState<string | null>(null);
  const [reportHtml, setReportHtml] = useState<string>('');
  const [showReport, setShowReport] = useState(false);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const reportRef = useRef<HTMLIFrameElement>(null);
  
  // Prepare test cases
  const testCases: TestCase[] = [
    // Dashboard Cards test
    {
      id: 'dashboard-cards',
      name: 'Dashboard Cards',
      description: 'Tests performance of animating multiple dashboard cards simultaneously',
      component: (
        <EnhancedDashboardCardGrid>
          {MOCK_DASHBOARD_CARDS.map((card) => (
            <EnhancedDashboardCard
              key={card.id}
              title={card.title}
              value={card.metric.toString()}
              change={card.change}
              status={card.status as 'positive' | 'negative'}
              animationsVisible={false}
              onClick={() => {}}
            >
              <Typography variant="body2">{card.content}</Typography>
            </EnhancedDashboardCard>
          ))}
        </EnhancedDashboardCardGrid>
      ),
      triggerAnimation: () => {
        // Force re-render with animations visible
        const container = containerRef.current;
        if (container) {
          const cards = container.querySelectorAll('[data-testid="dashboard-card"]');
          cards.forEach((card, i) => {
            setTimeout(() => {
              (card as HTMLElement).setAttribute('data-visible', 'true');
            }, i * 50);
          });
        }
      },
      duration: 3000
    },
    
    // Search Results test
    {
      id: 'search-results',
      name: 'Search Results',
      description: 'Tests performance of animating search result cards with staggered animations',
      component: (
        <EnhancedSearchResults
          results={MOCK_SEARCH_RESULTS}
          animationsVisible={false}
          resultsLabel="Search Results"
        />
      ),
      triggerAnimation: () => {
        // Force re-render with animations visible
        const container = containerRef.current;
        if (container) {
          const searchResults = container.querySelector('[data-testid="search-results"]');
          if (searchResults) {
            (searchResults as HTMLElement).setAttribute('data-visible', 'true');
          }
        }
      },
      duration: 3000
    },
    
    // Table Components test
    {
      id: 'table-components',
      name: 'Table Components',
      description: 'Tests performance of animating table rows and cells',
      component: (
        <EnhancedAnimatedTable
          headers={['Name', 'Value', 'Status', 'Date', 'Progress']}
          data={MOCK_TABLE_DATA}
          getRowKey={(row) => row.id}
          renderCell={(row, header) => {
            switch (header) {
              case 'Name':
                return row.name;
              case 'Value':
                return row.value;
              case 'Status':
                return (
                  <EnhancedChip
                    label={row.status}
                    color={row.status === 'active' ? 'success' : row.status === 'pending' ? 'warning' : 'default'}
                    size="small"
                  />
                );
              case 'Date':
                return row.date;
              case 'Progress':
                return (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                    <EnhancedSlider
                      value={row.progress}
                      disabled
                      sx={{ width: '100px' }}
                    />
                    <Typography variant="body2">{row.progress}%</Typography>
                  </Box>
                );
              default:
                return '';
            }
          }}
          animationsVisible={false}
          title="Data Table"
        />
      ),
      triggerAnimation: () => {
        // Force re-render with animations visible
        const container = containerRef.current;
        if (container) {
          const table = container.querySelector('[data-testid="animated-table"]');
          if (table) {
            (table as HTMLElement).setAttribute('data-visible', 'true');
          }
        }
      },
      duration: 3000
    },
    
    // Form Components test
    {
      id: 'form-components',
      name: 'Form Components',
      description: 'Tests performance of animating multiple form components',
      component: (
        <EnhancedPaper sx={{ p: 3 }}>
          <EnhancedGradientTypography variant="h5" sx={{ mb: 3 }}>
            Form Components
          </EnhancedGradientTypography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <EnhancedTextField
                label="Text Field"
                fullWidth
                animationsVisible={false}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Select</InputLabel>
                <EnhancedSelect
                  label="Select"
                  value=""
                  animationsVisible={false}
                >
                  <MenuItem value="">None</MenuItem>
                  <MenuItem value="option1">Option 1</MenuItem>
                  <MenuItem value="option2">Option 2</MenuItem>
                </EnhancedSelect>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <EnhancedSlider
                defaultValue={50}
                animationsVisible={false}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <EnhancedFormControlLabel
                control={<EnhancedSwitch animationsVisible={false} />}
                label="Toggle Switch"
                animationsVisible={false}
              />
            </Grid>
            <Grid item xs={12}>
              <EnhancedAlert severity="info" animationsVisible={false}>
                This is an alert message with animations
              </EnhancedAlert>
            </Grid>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <EnhancedButton variant="contained" animationsVisible={false}>
                  Primary Button
                </EnhancedButton>
                <EnhancedButton variant="outlined" animationsVisible={false}>
                  Secondary Button
                </EnhancedButton>
                <EnhancedButton variant="text" animationsVisible={false}>
                  Text Button
                </EnhancedButton>
              </Box>
            </Grid>
          </Grid>
        </EnhancedPaper>
      ),
      triggerAnimation: () => {
        // Force re-render with animations visible
        const container = containerRef.current;
        if (container) {
          const formComponents = container.querySelectorAll('[data-animation="form"]');
          formComponents.forEach((component, i) => {
            setTimeout(() => {
              (component as HTMLElement).setAttribute('data-visible', 'true');
            }, i * 100);
          });
        }
      },
      duration: 3000
    },
    
    // Combined Components test (stress test)
    {
      id: 'combined-components',
      name: 'Combined Components (Stress Test)',
      description: 'Tests performance when many different animated components are on screen',
      component: (
        <Box>
          <EnhancedGradientTypography variant="h4" sx={{ mb: 3 }} animationsVisible={false}>
            Dashboard Overview
          </EnhancedGradientTypography>
          
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <EnhancedAlert severity="success" sx={{ mb: 3 }} animationsVisible={false}>
                All systems are operating normally with optimal performance.
              </EnhancedAlert>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <EnhancedDashboardCard
                title="Total Users"
                value="1,234"
                change="+5.2%"
                status="positive"
                animationsVisible={false}
              >
                <Typography variant="body2">User growth is on track with projections.</Typography>
              </EnhancedDashboardCard>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <EnhancedDashboardCard
                title="Revenue"
                value="$45,678"
                change="+8.7%"
                status="positive"
                animationsVisible={false}
              >
                <Typography variant="body2">Revenue has exceeded quarterly targets.</Typography>
              </EnhancedDashboardCard>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <EnhancedDashboardCard
                title="Active Sessions"
                value="892"
                change="-3.1%"
                status="negative"
                animationsVisible={false}
              >
                <Typography variant="body2">Session count decreased slightly from last week.</Typography>
              </EnhancedDashboardCard>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedPaper sx={{ p: 3 }} animationsVisible={false}>
                <EnhancedTypography variant="h6" gutterBottom animationsVisible={false}>
                  Quick Settings
                </EnhancedTypography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <EnhancedFormControlLabel
                      control={<EnhancedSwitch animationsVisible={false} />}
                      label="Automatic Updates"
                      animationsVisible={false}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Theme</InputLabel>
                      <EnhancedSelect
                        label="Theme"
                        value="light"
                        animationsVisible={false}
                      >
                        <MenuItem value="light">Light</MenuItem>
                        <MenuItem value="dark">Dark</MenuItem>
                        <MenuItem value="system">System</MenuItem>
                      </EnhancedSelect>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <EnhancedSlider
                      defaultValue={75}
                      valueLabelDisplay="auto"
                      marks
                      step={25}
                      min={0}
                      max={100}
                      animationsVisible={false}
                    />
                  </Grid>
                </Grid>
              </EnhancedPaper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <EnhancedAnimatedTable
                headers={['Name', 'Status', 'Progress']}
                data={MOCK_TABLE_DATA.slice(0, 5)}
                getRowKey={(row) => row.id}
                renderCell={(row, header) => {
                  switch (header) {
                    case 'Name':
                      return row.name;
                    case 'Status':
                      return (
                        <EnhancedChip
                          label={row.status}
                          color={row.status === 'active' ? 'success' : row.status === 'pending' ? 'warning' : 'default'}
                          size="small"
                          animationsVisible={false}
                        />
                      );
                    case 'Progress':
                      return `${row.progress}%`;
                    default:
                      return '';
                  }
                }}
                animationsVisible={false}
                title="Recent Activities"
              />
            </Grid>
            
            <Grid item xs={12}>
              <EnhancedSearchResults
                results={MOCK_SEARCH_RESULTS.slice(0, 3)}
                animationsVisible={false}
                resultsLabel="Recent Documents"
              />
            </Grid>
          </Grid>
        </Box>
      ),
      triggerAnimation: () => {
        // Trigger all animations in sequence with slight delays
        const container = containerRef.current;
        if (container) {
          const allComponents = container.querySelectorAll('[data-animation]');
          allComponents.forEach((component, i) => {
            setTimeout(() => {
              (component as HTMLElement).setAttribute('data-visible', 'true');
            }, i * 50);
          });
        }
      },
      duration: 5000
    }
  ];
  
  // Handle tab change
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTabIndex(newValue);
    if (soundEffectsEnabled) {
      soundEffects.tap();
    }
  };
  
  // Handle running a single test
  const runTest = async (testId: string) => {
    const testCase = testCases.find(test => test.id === testId);
    if (!testCase) return;
    
    setCurrentTest(testId);
    setIsRunningTests(true);
    
    if (soundEffectsEnabled) {
      soundEffects.tap();
    }
    
    try {
      const metrics = await measureAnimationPerformance({
        testName: testCase.name,
        triggerAnimation: testCase.triggerAnimation,
        duration: testCase.duration,
        targetFps: 55,
      });
      
      // Add to results
      setTestResults(prev => {
        // Replace existing test result if present
        const existingIndex = prev.findIndex(r => r.name === testCase.name);
        if (existingIndex >= 0) {
          const newResults = [...prev];
          newResults[existingIndex] = { name: testCase.name, metrics };
          return newResults;
        }
        
        // Otherwise add new result
        return [...prev, { name: testCase.name, metrics }];
      });
      
      if (soundEffectsEnabled) {
        soundEffects.success();
      }
    } catch (error) {
      console.error('Test failed:', error);
      if (soundEffectsEnabled) {
        soundEffects.error();
      }
    } finally {
      setCurrentTest(null);
      setIsRunningTests(false);
    }
  };
  
  // Handle running all tests
  const runAllTests = async () => {
    setIsRunningTests(true);
    
    if (soundEffectsEnabled) {
      soundEffects.tap();
    }
    
    // Clear previous results
    setTestResults([]);
    
    // Run each test in sequence
    for (const testCase of testCases) {
      setCurrentTest(testCase.id);
      
      try {
        const metrics = await measureAnimationPerformance({
          testName: testCase.name,
          triggerAnimation: testCase.triggerAnimation,
          duration: testCase.duration,
          targetFps: 55,
        });
        
        // Add to results
        setTestResults(prev => [...prev, { name: testCase.name, metrics }]);
      } catch (error) {
        console.error(`Test failed for ${testCase.name}:`, error);
      }
      
      // Pause between tests
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    setCurrentTest(null);
    setIsRunningTests(false);
    
    if (soundEffectsEnabled) {
      soundEffects.success();
    }
  };
  
  // Generate HTML report
  const generateReport = () => {
    if (testResults.length === 0) return;
    
    const reportHtml = createPerformanceReport(testResults);
    setReportHtml(reportHtml);
    setShowReport(true);
    
    if (soundEffectsEnabled) {
      soundEffects.success();
    }
    
    // Set the report HTML
    if (reportRef.current) {
      const doc = reportRef.current.contentDocument;
      if (doc) {
        doc.open();
        doc.write(reportHtml);
        doc.close();
      }
    }
  };
  
  // Download report as HTML
  const downloadReport = () => {
    if (!reportHtml) return;
    
    const blob = new Blob([reportHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'animation-performance-report.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    if (soundEffectsEnabled) {
      soundEffects.tap();
    }
  };
  
  // Render the current test case
  const renderTestCase = () => {
    const testCase = testCases[activeTabIndex];
    if (!testCase) return null;
    
    return (
      <Box ref={containerRef} sx={{ mt: 3 }}>
        {testCase.component}
      </Box>
    );
  };
  
  // Render test controls
  const renderTestControls = () => {
    const testCase = testCases[activeTabIndex];
    if (!testCase) return null;
    
    return (
      <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
        <EnhancedButton
          variant="contained"
          onClick={() => runTest(testCase.id)}
          disabled={isRunningTests}
          startIcon={<SpeedIcon />}
        >
          Run This Test
        </EnhancedButton>
        <EnhancedButton
          variant="outlined"
          onClick={runAllTests}
          disabled={isRunningTests}
          startIcon={<AssessmentIcon />}
        >
          Run All Tests
        </EnhancedButton>
        <EnhancedButton
          variant="outlined"
          onClick={generateReport}
          disabled={testResults.length === 0 || isRunningTests}
          startIcon={<SaveIcon />}
          color="success"
        >
          Generate Report
        </EnhancedButton>
      </Box>
    );
  };
  
  // Render test results
  const renderTestResults = () => {
    if (testResults.length === 0) return null;
    
    return (
      <Box sx={{ mt: 4 }}>
        <EnhancedTypography variant="h6" gutterBottom>
          Test Results
        </EnhancedTypography>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Test</TableCell>
                <TableCell>FPS</TableCell>
                <TableCell>Duration</TableCell>
                <TableCell>Smoothness</TableCell>
                <TableCell>Long Frames</TableCell>
                <TableCell>Rating</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {testResults.map((result) => (
                <TableRow key={result.name}>
                  <TableCell>{result.name}</TableCell>
                  <TableCell>{result.metrics.fps.toFixed(1)}</TableCell>
                  <TableCell>{result.metrics.duration.toFixed(0)} ms</TableCell>
                  <TableCell>{result.metrics.smoothnessPercentage.toFixed(1)}%</TableCell>
                  <TableCell>
                    {result.metrics.longFrames} ({result.metrics.longFramesPercentage.toFixed(1)}%)
                  </TableCell>
                  <TableCell>
                    <Rating 
                      value={getPerformanceRating(result.metrics)} 
                      readOnly 
                      max={5}
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={result.metrics.meetsStandards ? 'PASS' : 'FAIL'}
                      color={result.metrics.meetsStandards ? 'success' : 'error'}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };
  
  return (
    <Container maxWidth="lg">
      <EnhancedGradientTypography variant="h4" gutterBottom>
        Animation Performance Testing
      </EnhancedGradientTypography>
      
      <EnhancedTypography variant="body1" paragraph>
        This tool measures the performance of various animations across components. 
        Select a test case, run the test, and view the results.
      </EnhancedTypography>
      
      {!animationsEnabled && (
        <EnhancedAlert severity="warning" sx={{ mb: 3 }}>
          Animations are currently disabled. Enable animations from the settings panel to run tests.
        </EnhancedAlert>
      )}
      
      {/* Test selection tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTabIndex} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
        >
          {testCases.map((test) => (
            <Tab key={test.id} label={test.name} />
          ))}
        </Tabs>
      </Paper>
      
      {/* Test description */}
      <EnhancedPaper sx={{ p: 3, mb: 3 }}>
        <EnhancedTypography variant="h6" gutterBottom>
          {testCases[activeTabIndex]?.name}
        </EnhancedTypography>
        <EnhancedTypography variant="body2" color="text.secondary">
          {testCases[activeTabIndex]?.description}
        </EnhancedTypography>
        
        {/* Test status */}
        {currentTest && (
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
            <CircularProgress size={20} sx={{ mr: 1 }} />
            <Typography variant="body2">
              Running test: {testCases.find(t => t.id === currentTest)?.name}
            </Typography>
          </Box>
        )}
      </EnhancedPaper>
      
      {/* Test controls */}
      {renderTestControls()}
      
      {/* Test case component */}
      {renderTestCase()}
      
      {/* Test results */}
      {renderTestResults()}
      
      {/* Performance Report */}
      {showReport && (
        <Box sx={{ mt: 4 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <EnhancedTypography variant="h6">
              Performance Report
            </EnhancedTypography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                size="small"
                onClick={downloadReport}
                startIcon={<SaveIcon />}
              >
                Download Report
              </Button>
              <IconButton onClick={() => setShowReport(false)}>
                <ClearIcon />
              </IconButton>
            </Box>
          </Box>
          
          <Box sx={{ height: '600px', border: '1px solid #ddd', borderRadius: 1, overflow: 'hidden' }}>
            <iframe
              ref={reportRef}
              title="Performance Report"
              style={{ width: '100%', height: '100%', border: 'none' }}
            />
          </Box>
        </Box>
      )}
    </Container>
  );
};

export default AnimationPerformanceTest;