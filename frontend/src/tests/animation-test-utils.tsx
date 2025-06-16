import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import AnimationProvider from '../context/AnimationContext';
import { animated } from '@react-spring/web';

/**
 * Creates a test wrapper with animation providers
 * @param ui The component to render
 * @returns Rendered component with all providers
 */
export const renderWithAnimationProviders = (ui: React.ReactElement) => {
  const theme = createTheme();
  
  return render(
    <ThemeProvider theme={theme}>
      <AnimationProvider>
        {ui}
      </AnimationProvider>
    </ThemeProvider>
  );
};

/**
 * Create a mock component that uses animations
 * @param testId The test ID to use for finding the component
 * @returns A component that uses animations
 */
export const createAnimatedTestComponent = (testId: string) => {
  const AnimatedDiv = animated.div;
  
  return ({ animationStyle, children }: { animationStyle: any, children?: React.ReactNode }) => (
    <AnimatedDiv 
      data-testid={testId}
      style={animationStyle}
    >
      {children || 'Animated Content'}
    </AnimatedDiv>
  );
};

/**
 * Waits for an animation to complete
 * @param testId The test ID of the animated element
 * @param propertyName The CSS property to check (e.g., 'opacity')
 * @param expectedValue The expected final value
 * @param timeout Optional timeout in ms (default: 1000)
 */
export const waitForAnimation = async (
  testId: string, 
  propertyName: string, 
  expectedValue: string | number,
  timeout = 1000
) => {
  const element = screen.getByTestId(testId);
  
  await waitFor(() => {
    const style = window.getComputedStyle(element);
    const currentValue = style[propertyName as any];
    expect(currentValue).toBe(String(expectedValue));
  }, { timeout });
};

/**
 * Tests if an animation sequence completes in the correct order
 * @param testIds Array of test IDs in the expected animation sequence
 * @param propertyName The CSS property to check (e.g., 'opacity')
 * @param expectedValue The expected final value
 * @param timeout Optional timeout in ms for each animation (default: 1000)
 */
export const testAnimationSequence = async (
  testIds: string[],
  propertyName: string,
  expectedValue: string | number,
  timeout = 1000
) => {
  for (const testId of testIds) {
    await waitForAnimation(testId, propertyName, expectedValue, timeout);
  }
};

/**
 * Creates a test for staggered animations
 * @param parentTestId The test ID of the parent element
 * @param childrenCount The number of children to expect
 * @param childTestIdPrefix The prefix for child test IDs
 * @param propertyName The CSS property to check (e.g., 'opacity')
 * @param expectedValue The expected final value
 * @param timeout Optional timeout in ms (default: 2000)
 */
export const testStaggeredAnimations = async (
  parentTestId: string,
  childrenCount: number,
  childTestIdPrefix: string,
  propertyName: string,
  expectedValue: string | number,
  timeout = 2000
) => {
  const parent = screen.getByTestId(parentTestId);
  
  // Verify correct number of children
  expect(parent.children.length).toBe(childrenCount);
  
  // Check that each child eventually reaches the expected animation state
  const testIds = Array.from({ length: childrenCount }, (_, i) => `${childTestIdPrefix}-${i}`);
  
  await testAnimationSequence(testIds, propertyName, expectedValue, timeout);
};