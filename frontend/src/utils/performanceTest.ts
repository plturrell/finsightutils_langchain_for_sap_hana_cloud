/**
 * Animation Performance Testing Utilities
 * This is a wrapper around the shared animation performance testing utilities
 */

import { 
  measureAnimationPerformance, 
  createPerformanceReport, 
  getPerformanceRating,
  PerformanceMetrics,
  PerformanceTestOptions
} from '@finsightdev/ui-animations/dist/utils/animationPerformanceTest';

/**
 * Run a batch of performance tests on animated components
 * @param tests Array of test configurations
 * @returns HTML report with test results
 */
export const runPerformanceTests = async (
  tests: Array<{
    name: string;
    triggerAnimation: () => void;
    duration?: number;
  }>
): Promise<string> => {
  console.log(`Running ${tests.length} animation performance tests...`);
  
  const results: Array<{ name: string; metrics: PerformanceMetrics }> = [];
  
  // Run each test
  for (const test of tests) {
    const metrics = await measureAnimationPerformance({
      testName: test.name,
      triggerAnimation: test.triggerAnimation,
      duration: test.duration || 2000,
      targetFps: 55
    });
    
    results.push({
      name: test.name,
      metrics
    });
    
    console.log(`Test "${test.name}" completed:`);
    console.log(`  FPS: ${metrics.fps.toFixed(1)}`);
    console.log(`  Smoothness: ${metrics.smoothnessPercentage.toFixed(1)}%`);
    console.log(`  Result: ${metrics.meetsStandards ? 'PASS' : 'FAIL'}`);
  }
  
  // Generate report
  return createPerformanceReport(results);
};

export {
  measureAnimationPerformance,
  createPerformanceReport,
  getPerformanceRating,
  type PerformanceMetrics,
  type PerformanceTestOptions
};