/**
 * Animation Performance Testing Utilities
 * Tools for measuring and analyzing animation performance in React
 */

/**
 * Interface for performance metrics
 */
export interface PerformanceMetrics {
  /** Average FPS during animation */
  fps: number;
  /** Duration of the animation in milliseconds */
  duration: number;
  /** Number of frames rendered */
  frameCount: number;
  /** Maximum time between frames (ms) */
  maxFrameTime: number;
  /** Average time between frames (ms) */
  averageFrameTime: number;
  /** Frames that took longer than 16ms (60fps threshold) to render */
  longFrames: number;
  /** Percentage of frames that took longer than 16ms */
  longFramesPercentage: number;
  /** Percentage of time the animation maintained 60fps */
  smoothnessPercentage: number;
  /** Whether the animation meets performance standards */
  meetsStandards: boolean;
}

/**
 * Interface for performance test options
 */
export interface PerformanceTestOptions {
  /** Test name for reporting */
  testName: string;
  /** Function to trigger the animation */
  triggerAnimation: () => void;
  /** Duration to measure performance (ms) */
  duration?: number;
  /** Target FPS threshold for 'meets standards' */
  targetFps?: number;
  /** Function to call after test completes */
  onComplete?: (metrics: PerformanceMetrics) => void;
  /** Function to call on each frame */
  onFrame?: (frameTime: number) => void;
}

/**
 * Measures animation performance
 * @param options Test configuration options
 * @returns Promise that resolves with performance metrics
 */
export async function measureAnimationPerformance(
  options: PerformanceTestOptions
): Promise<PerformanceMetrics> {
  const {
    testName,
    triggerAnimation,
    duration = 2000, // Default 2 seconds
    targetFps = 55, // Default target FPS (slightly below 60 to account for variations)
    onComplete,
    onFrame,
  } = options;

  console.log(`Starting animation performance test: ${testName}`);
  
  // Variables to track performance
  let startTime = performance.now();
  let frames = 0;
  let frameTimes: number[] = [];
  let lastFrameTime = startTime;
  let testActive = true;
  
  // Function to measure each frame
  const frameCallback = () => {
    const now = performance.now();
    const frameTime = now - lastFrameTime;
    
    // Record frame data
    frames++;
    frameTimes.push(frameTime);
    lastFrameTime = now;
    
    // Call frame callback if provided
    if (onFrame) {
      onFrame(frameTime);
    }
    
    // Continue measuring if test is still active
    if (testActive && now - startTime < duration) {
      requestAnimationFrame(frameCallback);
    } else if (testActive) {
      // Finalize test
      testActive = false;
      const metrics = calculateMetrics(startTime, now, frames, frameTimes, targetFps);
      console.log(`Animation performance test completed: ${testName}`);
      console.table(metrics);
      
      if (onComplete) {
        onComplete(metrics);
      }
    }
  };
  
  // Start the test
  return new Promise((resolve) => {
    // Trigger the animation
    triggerAnimation();
    
    // Start measuring frames
    requestAnimationFrame(() => {
      startTime = performance.now();
      lastFrameTime = startTime;
      requestAnimationFrame(frameCallback);
      
      // Ensure the test completes even if animation stalls
      setTimeout(() => {
        if (testActive) {
          testActive = false;
          const now = performance.now();
          const metrics = calculateMetrics(startTime, now, frames, frameTimes, targetFps);
          console.log(`Animation performance test completed (timeout): ${testName}`);
          console.table(metrics);
          
          if (onComplete) {
            onComplete(metrics);
          }
          
          resolve(metrics);
        }
      }, duration + 500); // Add buffer time to ensure test completes
    });
  });
}

/**
 * Calculate performance metrics from raw frame data
 */
function calculateMetrics(
  startTime: number,
  endTime: number,
  frames: number,
  frameTimes: number[],
  targetFps: number
): PerformanceMetrics {
  const duration = endTime - startTime;
  const fps = frames / (duration / 1000);
  const longFrameThreshold = 1000 / 60; // ~16.67ms per frame at 60fps
  
  // Calculate frame statistics
  const maxFrameTime = Math.max(...frameTimes);
  const averageFrameTime = frameTimes.reduce((sum, time) => sum + time, 0) / frames;
  const longFrames = frameTimes.filter(time => time > longFrameThreshold).length;
  const longFramesPercentage = (longFrames / frames) * 100;
  const smoothnessPercentage = 100 - longFramesPercentage;
  
  // Determine if performance meets standards
  const meetsStandards = fps >= targetFps && longFramesPercentage < 10;
  
  return {
    fps,
    duration,
    frameCount: frames,
    maxFrameTime,
    averageFrameTime,
    longFrames,
    longFramesPercentage,
    smoothnessPercentage,
    meetsStandards
  };
}

/**
 * Create a performance test report for a set of tests
 * @param metrics Array of performance metrics from different tests
 * @returns HTML string with a formatted report
 */
export function createPerformanceReport(metrics: Array<{ name: string; metrics: PerformanceMetrics }>): string {
  // Calculate overall statistics
  const overallFps = metrics.reduce((sum, m) => sum + m.metrics.fps, 0) / metrics.length;
  const overallSmoothness = metrics.reduce((sum, m) => sum + m.metrics.smoothnessPercentage, 0) / metrics.length;
  const passingTests = metrics.filter(m => m.metrics.meetsStandards).length;
  const passingPercentage = (passingTests / metrics.length) * 100;
  
  // Generate HTML report
  return `
    <html>
      <head>
        <title>Animation Performance Report</title>
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
          }
          h1, h2, h3 {
            color: #0066B3;
          }
          .summary {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 5px solid #0066B3;
          }
          .metric-card {
            display: inline-block;
            width: 30%;
            margin: 1%;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
          }
          .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
          }
          .green { color: #4caf50; }
          .red { color: #f44336; }
          .amber { color: #ff9800; }
          table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
          }
          th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
          }
          th {
            background-color: #f8f9fa;
            font-weight: 500;
          }
          tr:hover {
            background-color: #f8f9fa;
          }
          .pass {
            color: #4caf50;
            font-weight: 500;
          }
          .fail {
            color: #f44336;
            font-weight: 500;
          }
          .bar-container {
            width: 100%;
            background-color: #f1f1f1;
            border-radius: 4px;
            margin: 10px 0;
          }
          .bar {
            height: 10px;
            border-radius: 4px;
          }
          .fps-bar {
            background-color: #0066B3;
          }
          .smoothness-bar {
            background-color: #4caf50;
          }
        </style>
      </head>
      <body>
        <h1>Animation Performance Report</h1>
        <div class="summary">
          <h2>Summary</h2>
          <div class="metric-card">
            <h3>Average FPS</h3>
            <div class="metric-value ${overallFps >= 55 ? 'green' : overallFps >= 45 ? 'amber' : 'red'}">
              ${overallFps.toFixed(1)}
            </div>
            <div class="bar-container">
              <div class="bar fps-bar" style="width: ${Math.min(100, (overallFps / 60) * 100)}%"></div>
            </div>
          </div>
          <div class="metric-card">
            <h3>Smoothness</h3>
            <div class="metric-value ${overallSmoothness >= 90 ? 'green' : overallSmoothness >= 75 ? 'amber' : 'red'}">
              ${overallSmoothness.toFixed(1)}%
            </div>
            <div class="bar-container">
              <div class="bar smoothness-bar" style="width: ${overallSmoothness}%"></div>
            </div>
          </div>
          <div class="metric-card">
            <h3>Passing Tests</h3>
            <div class="metric-value ${passingPercentage >= 90 ? 'green' : passingPercentage >= 75 ? 'amber' : 'red'}">
              ${passingTests}/${metrics.length} (${passingPercentage.toFixed(0)}%)
            </div>
          </div>
        </div>
        
        <h2>Test Results</h2>
        <table>
          <thead>
            <tr>
              <th>Test Name</th>
              <th>FPS</th>
              <th>Duration (ms)</th>
              <th>Frame Count</th>
              <th>Avg Frame Time (ms)</th>
              <th>Max Frame Time (ms)</th>
              <th>Long Frames</th>
              <th>Smoothness</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            ${metrics.map(m => `
              <tr>
                <td>${m.name}</td>
                <td>${m.metrics.fps.toFixed(1)}</td>
                <td>${m.metrics.duration.toFixed(0)}</td>
                <td>${m.metrics.frameCount}</td>
                <td>${m.metrics.averageFrameTime.toFixed(2)}</td>
                <td>${m.metrics.maxFrameTime.toFixed(2)}</td>
                <td>${m.metrics.longFrames} (${m.metrics.longFramesPercentage.toFixed(1)}%)</td>
                <td>${m.metrics.smoothnessPercentage.toFixed(1)}%</td>
                <td class="${m.metrics.meetsStandards ? 'pass' : 'fail'}">
                  ${m.metrics.meetsStandards ? 'PASS' : 'FAIL'}
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
        
        <h2>Recommendations</h2>
        <ul>
          ${metrics.filter(m => !m.metrics.meetsStandards).map(m => `
            <li>
              <strong>${m.name}</strong>: 
              ${m.metrics.fps < 55 ? 'Low FPS, consider optimizing animation complexity or reducing elements.' : ''} 
              ${m.metrics.longFramesPercentage > 10 ? 'Too many long frames, check for expensive operations during animation.' : ''}
              ${m.metrics.maxFrameTime > 50 ? 'Some frames are extremely slow, look for blocking operations.' : ''}
            </li>
          `).join('')}
          ${metrics.filter(m => !m.metrics.meetsStandards).length === 0 ? 
            '<li>All animations meet performance standards! Great job!</li>' : ''}
        </ul>
        
        <h3>General Tips for Animation Performance</h3>
        <ul>
          <li>Use <code>transform</code> and <code>opacity</code> for animations instead of properties that trigger layout</li>
          <li>Add <code>will-change</code> property to elements that will animate</li>
          <li>Batch animations together to reduce render cycles</li>
          <li>Use staggered animations to distribute processing over time</li>
          <li>For complex components, consider using <code>React.memo</code> to prevent unnecessary re-renders</li>
          <li>Avoid expensive calculations during animation phases</li>
          <li>Reduce the number of animated elements on screen at once</li>
          <li>Use CSS animations where possible for simple effects</li>
        </ul>
      </body>
    </html>
  `;
}

/**
 * Load test that creates many instances of a component and measures performance
 * @param options Load test configuration
 * @returns Promise that resolves with performance metrics
 */
export async function runAnimationLoadTest<T>(
  options: {
    testName: string;
    componentFactory: (props: T) => JSX.Element;
    propsFactory: (index: number) => T;
    count: number;
    container: HTMLElement;
    triggerAnimation: () => void;
    duration?: number;
  }
): Promise<PerformanceMetrics> {
  const { testName, componentFactory, propsFactory, count, container, triggerAnimation, duration } = options;
  
  // Create components
  const components = [];
  for (let i = 0; i < count; i++) {
    components.push(componentFactory(propsFactory(i)));
  }
  
  // Measure performance
  return measureAnimationPerformance({
    testName,
    triggerAnimation,
    duration,
  });
}

/**
 * Get a performance rating based on metrics
 * @param metrics Performance metrics
 * @returns Rating on scale of 1-5 (5 being best)
 */
export function getPerformanceRating(metrics: PerformanceMetrics): number {
  // Calculate score based on FPS, smoothness, and frame times
  let score = 0;
  
  // FPS Score (0-2)
  if (metrics.fps >= 58) score += 2;
  else if (metrics.fps >= 50) score += 1.5;
  else if (metrics.fps >= 40) score += 1;
  else if (metrics.fps >= 30) score += 0.5;
  
  // Smoothness Score (0-2)
  if (metrics.smoothnessPercentage >= 95) score += 2;
  else if (metrics.smoothnessPercentage >= 90) score += 1.5;
  else if (metrics.smoothnessPercentage >= 80) score += 1;
  else if (metrics.smoothnessPercentage >= 70) score += 0.5;
  
  // Frame Time Score (0-1)
  if (metrics.maxFrameTime <= 20) score += 1;
  else if (metrics.maxFrameTime <= 33) score += 0.5;
  
  return Math.round(score);
}

/**
 * Exports an object containing all animation performance testing utilities
 */
export default {
  measureAnimationPerformance,
  createPerformanceReport,
  runAnimationLoadTest,
  getPerformanceRating
};