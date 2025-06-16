import React from 'react';
import { screen, fireEvent, render } from '@testing-library/react';
import { 
  renderWithAnimationProviders, 
  createAnimatedTestComponent,
  waitForAnimation,
  testAnimationSequence,
  testStaggeredAnimations
} from './animation-test-utils';
import { useFadeUpAnimation, useScaleAnimation, useChainedAnimations, useBatchAnimations } from '../hooks/useAnimations';
import { useAnimationContext } from '../context/AnimationContext';
import AnimationToggle from '../components/AnimationToggle';
import { measureAnimationPerformance, PerformanceMetrics } from '../utils/animationPerformanceTest';
import { 
  EnhancedButton, 
  EnhancedCard, 
  EnhancedDashboardCard,
  EnhancedSearchResults,
  EnhancedAnimatedTable
} from '../components/enhanced';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// Create test components
const FadeUpComponent = createAnimatedTestComponent('fade-up-component');
const ScaleComponent = createAnimatedTestComponent('scale-component');

// A test component that uses the animation context
const AnimationContextTestComponent = () => {
  const { animationsEnabled, toggleAnimations } = useAnimationContext();
  const fadeAnimation = useFadeUpAnimation(true, { enabled: animationsEnabled });
  
  return (
    <div>
      <button 
        data-testid="toggle-button" 
        onClick={toggleAnimations}
      >
        Toggle Animations
      </button>
      <div data-testid="animations-status">
        {animationsEnabled ? 'Enabled' : 'Disabled'}
      </div>
      <FadeUpComponent 
        animationStyle={fadeAnimation}
      />
    </div>
  );
};

// A test component with chained animations
const ChainedAnimationsComponent = () => {
  const [visible, setVisible] = React.useState(false);
  const { animationsEnabled } = useAnimationContext();
  
  const {
    headerAnimation,
    contentAnimation,
    footerAnimation,
    itemsTrail
  } = useChainedAnimations(visible, {
    fadeHeader: true,
    fadeContent: true,
    fadeFooter: true,
    staggerItems: true,
    itemCount: 3,
    enabled: animationsEnabled
  });
  
  return (
    <div>
      <button 
        data-testid="show-animations-button" 
        onClick={() => setVisible(true)}
      >
        Show Animations
      </button>
      
      <div data-testid="header" style={headerAnimation}>
        Header
      </div>
      
      <div data-testid="content" style={contentAnimation}>
        Content
      </div>
      
      <div data-testid="items-container">
        {itemsTrail.map((style, index) => (
          <div 
            key={index}
            data-testid={`item-${index}`}
            style={style}
          >
            Item {index + 1}
          </div>
        ))}
      </div>
      
      <div data-testid="footer" style={footerAnimation}>
        Footer
      </div>
    </div>
  );
};

describe('Animation Hooks', () => {
  test('useFadeUpAnimation creates proper animation styles', () => {
    const TestComponent = () => {
      const [visible, setVisible] = React.useState(false);
      const fadeAnimation = useFadeUpAnimation(visible);
      
      return (
        <div>
          <button 
            data-testid="show-button" 
            onClick={() => setVisible(true)}
          >
            Show
          </button>
          <FadeUpComponent animationStyle={fadeAnimation} />
        </div>
      );
    };
    
    renderWithAnimationProviders(<TestComponent />);
    
    // Initially the component should be invisible (opacity: 0)
    const component = screen.getByTestId('fade-up-component');
    expect(component.style.opacity).toBe('0');
    
    // Click the button to trigger the animation
    fireEvent.click(screen.getByTestId('show-button'));
    
    // The component should now be animating to visible
    expect(component.style.opacity).not.toBe('0');
  });
  
  test('useScaleAnimation creates proper animation styles', () => {
    const TestComponent = () => {
      const [visible, setVisible] = React.useState(false);
      const scaleAnimation = useScaleAnimation(visible);
      
      return (
        <div>
          <button 
            data-testid="show-button" 
            onClick={() => setVisible(true)}
          >
            Show
          </button>
          <ScaleComponent animationStyle={scaleAnimation} />
        </div>
      );
    };
    
    renderWithAnimationProviders(<TestComponent />);
    
    // Initially the component should be scaled down (transform: scale(0.9))
    const component = screen.getByTestId('scale-component');
    expect(component.style.transform).toContain('scale(0.9)');
    
    // Click the button to trigger the animation
    fireEvent.click(screen.getByTestId('show-button'));
    
    // The component should now be animating to full scale
    // This is a bit hard to test precisely due to how React Spring works,
    // but we can check that it's not the initial value anymore
    expect(component.style.transform).not.toBe('scale(0.9)');
  });
});

describe('Animation Context', () => {
  test('AnimationProvider controls animation state properly', () => {
    renderWithAnimationProviders(<AnimationContextTestComponent />);
    
    // Initially animations should be enabled
    expect(screen.getByTestId('animations-status').textContent).toBe('Enabled');
    
    // Click the toggle button to disable animations
    fireEvent.click(screen.getByTestId('toggle-button'));
    
    // Now animations should be disabled
    expect(screen.getByTestId('animations-status').textContent).toBe('Disabled');
    
    // And the component's animation style should reflect that
    const component = screen.getByTestId('fade-up-component');
    expect(component.style.opacity).toBe('0');
  });
});

describe('Animation Components', () => {
  test('AnimationToggle shows correct icon based on animation state', () => {
    renderWithAnimationProviders(<AnimationToggle />);
    
    // Find the button and click it to open the menu
    const toggleButton = screen.getByRole('button');
    fireEvent.click(toggleButton);
    
    // Initially, the switch should be checked (animations enabled)
    const switchElement = screen.getByRole('checkbox');
    expect(switchElement).toBeChecked();
    
    // Click the switch to disable animations
    fireEvent.click(switchElement);
    
    // The switch should now be unchecked
    expect(switchElement).not.toBeChecked();
  });
});

describe('Animation Sequences', () => {
  test('Chained animations execute in the correct order', async () => {
    renderWithAnimationProviders(<ChainedAnimationsComponent />);
    
    // Initially animations are not shown
    const header = screen.getByTestId('header');
    expect(header.style.opacity).toBe('0');
    
    // Trigger the animations
    fireEvent.click(screen.getByTestId('show-animations-button'));
    
    // Test that animations occur in the expected sequence
    // This is the expected order: header -> content -> items -> footer
    await testAnimationSequence(
      ['header', 'content', 'item-0', 'item-1', 'item-2', 'footer'],
      'opacity',
      '1',
      3000 // Longer timeout for the whole sequence
    );
  });
});

/**
 * Test batch animations with useBatchAnimations hook
 */
describe('Batch Animations', () => {
  test('useBatchAnimations properly batches multiple animations', () => {
    // Create a test component that uses batch animations
    const BatchAnimationsTestComponent = () => {
      const [visible, setVisible] = React.useState(false);
      const { animationsEnabled } = useAnimationContext();
      
      const animations = useBatchAnimations(
        visible,
        [
          {
            key: 'first',
            from: { opacity: 0, transform: 'translateY(20px)' },
            to: { 
              opacity: animationsEnabled && visible ? 1 : 0, 
              transform: animationsEnabled && visible ? 'translateY(0)' : 'translateY(20px)'
            },
            delay: 0,
            config: { tension: 280, friction: 60 }
          },
          {
            key: 'second',
            from: { opacity: 0, transform: 'scale(0.9)' },
            to: { 
              opacity: animationsEnabled && visible ? 1 : 0, 
              transform: animationsEnabled && visible ? 'scale(1)' : 'scale(0.9)'
            },
            delay: 100,
            config: { tension: 280, friction: 60 }
          }
        ]
      );
      
      return (
        <div>
          <button 
            data-testid="show-button" 
            onClick={() => setVisible(true)}
          >
            Show
          </button>
          <div data-testid="first-element" style={animations.first}>First Element</div>
          <div data-testid="second-element" style={animations.second}>Second Element</div>
        </div>
      );
    };
    
    renderWithAnimationProviders(<BatchAnimationsTestComponent />);
    
    // Initially the elements should be invisible
    const firstElement = screen.getByTestId('first-element');
    const secondElement = screen.getByTestId('second-element');
    
    expect(firstElement.style.opacity).toBe('0');
    expect(secondElement.style.opacity).toBe('0');
    
    // Click the button to trigger the animations
    fireEvent.click(screen.getByTestId('show-button'));
    
    // Now the elements should be animating to visible
    expect(firstElement.style.opacity).not.toBe('0');
    expect(secondElement.style.opacity).not.toBe('0');
  });
});

/**
 * Performance testing for animations
 * These tests are skipped by default since they're resource-intensive
 * and require browser environment to be meaningful
 */
describe.skip('Animation Performance', () => {
  // Mock search result for testing
  const mockSearchResult = {
    document: {
      page_content: 'This is a test search result with content that would be displayed.',
      metadata: {
        title: 'Test Result',
        date: new Date().toISOString(),
        source: 'test',
        type: 'document'
      }
    },
    score: 0.95
  };
  
  // Performance standards
  const PERFORMANCE_STANDARDS = {
    MIN_FPS: 30,
    TARGET_FPS: 55,
    MAX_LONG_FRAMES_PERCENTAGE: 20,
    MIN_SMOOTHNESS_PERCENTAGE: 75
  };
  
  // Helper to check if performance meets minimum standards
  function meetsMinimumStandards(metrics: PerformanceMetrics): boolean {
    return (
      metrics.fps >= PERFORMANCE_STANDARDS.MIN_FPS &&
      metrics.longFramesPercentage <= PERFORMANCE_STANDARDS.MAX_LONG_FRAMES_PERCENTAGE &&
      metrics.smoothnessPercentage >= PERFORMANCE_STANDARDS.MIN_SMOOTHNESS_PERCENTAGE
    );
  }
  
  // Test wrapper for components
  const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <ThemeProvider theme={createTheme()}>
      <AnimationProvider>
        {children}
      </AnimationProvider>
    </ThemeProvider>
  );
  
  // Increase timeout for performance tests
  jest.setTimeout(30000);
  
  test('Button performance meets standards', async () => {
    // Render button
    const { container } = render(
      <TestWrapper>
        <EnhancedButton 
          variant="contained"
          data-testid="test-button"
        >
          Test Button
        </EnhancedButton>
      </TestWrapper>
    );
    
    // Find button element
    const button = container.querySelector('[data-testid="test-button"]');
    expect(button).not.toBeNull();
    
    // Measure performance
    const metrics = await measureAnimationPerformance({
      testName: 'Button Animation',
      triggerAnimation: () => {
        // Trigger hover animation
        const event = new MouseEvent('mouseenter', {
          bubbles: true,
          cancelable: true,
        });
        button?.dispatchEvent(event);
      },
      duration: 1000
    });
    
    // Log results
    console.log('Button Animation Performance:', metrics);
    
    // Assert performance meets standards
    expect(meetsMinimumStandards(metrics)).toBe(true);
  });
  
  test('Dashboard Card performance meets standards', async () => {
    // Render dashboard card
    const { container } = render(
      <TestWrapper>
        <EnhancedDashboardCard
          title="Test Dashboard Card"
          value="1,234"
          change="+5.2%"
          status="positive"
          animationsVisible={false}
          data-testid="test-dashboard-card"
        >
          <div>Test content</div>
        </EnhancedDashboardCard>
      </TestWrapper>
    );
    
    // Find card element
    const card = container.querySelector('[data-testid="test-dashboard-card"]');
    expect(card).not.toBeNull();
    
    // Measure performance
    const metrics = await measureAnimationPerformance({
      testName: 'Dashboard Card Animation',
      triggerAnimation: () => {
        // Set animations visible
        if (card) {
          (card as HTMLElement).setAttribute('data-visible', 'true');
        }
      },
      duration: 2000
    });
    
    // Log results
    console.log('Dashboard Card Animation Performance:', metrics);
    
    // Assert performance meets standards
    expect(meetsMinimumStandards(metrics)).toBe(true);
  });
  
  test('Search Results performance meets standards', async () => {
    // Create multiple search results
    const results = Array(10).fill(0).map((_, i) => ({
      ...mockSearchResult,
      document: {
        ...mockSearchResult.document,
        metadata: {
          ...mockSearchResult.document.metadata,
          title: `Test Result ${i + 1}`
        }
      },
      score: 0.95 - (i * 0.03)
    }));
    
    // Render search results
    const { container } = render(
      <TestWrapper>
        <EnhancedSearchResults
          results={results}
          animationsVisible={false}
          data-testid="test-search-results"
        />
      </TestWrapper>
    );
    
    // Find search results element
    const searchResults = container.querySelector('[data-testid="test-search-results"]');
    expect(searchResults).not.toBeNull();
    
    // Measure performance
    const metrics = await measureAnimationPerformance({
      testName: 'Search Results Animation',
      triggerAnimation: () => {
        // Set animations visible
        if (searchResults) {
          (searchResults as HTMLElement).setAttribute('data-visible', 'true');
        }
      },
      duration: 2000
    });
    
    // Log results
    console.log('Search Results Animation Performance:', metrics);
    
    // Assert performance meets standards
    expect(meetsMinimumStandards(metrics)).toBe(true);
  });
  
  test('Table performance meets standards', async () => {
    // Create test data
    const tableData = Array(20).fill(0).map((_, i) => ({
      id: `row-${i}`,
      name: `Item ${i + 1}`,
      value: Math.floor(Math.random() * 1000),
      status: i % 3 === 0 ? 'active' : i % 3 === 1 ? 'pending' : 'inactive',
    }));
    
    // Render table
    const { container } = render(
      <TestWrapper>
        <EnhancedAnimatedTable
          headers={['Name', 'Value', 'Status']}
          data={tableData}
          getRowKey={(row) => row.id}
          renderCell={(row, header) => {
            switch (header) {
              case 'Name': return row.name;
              case 'Value': return row.value;
              case 'Status': return row.status;
              default: return '';
            }
          }}
          animationsVisible={false}
          data-testid="test-table"
        />
      </TestWrapper>
    );
    
    // Find table element
    const table = container.querySelector('[data-testid="test-table"]');
    expect(table).not.toBeNull();
    
    // Measure performance
    const metrics = await measureAnimationPerformance({
      testName: 'Table Animation',
      triggerAnimation: () => {
        // Set animations visible
        if (table) {
          (table as HTMLElement).setAttribute('data-visible', 'true');
        }
      },
      duration: 2000
    });
    
    // Log results
    console.log('Table Animation Performance:', metrics);
    
    // Assert performance meets standards
    expect(meetsMinimumStandards(metrics)).toBe(true);
  });
});