import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Dashboard from '../pages/Dashboard';
import { ErrorProvider } from '../context/ErrorContext';
import * as apiClient from '../api/client';
import { 
  mockHealthStatus, 
  mockGPUInfo, 
  mockRecentQueries,
  mockPerformanceStats,
  mockPerformanceComparison 
} from '../api/mockData';

// Mock the API client module
jest.mock('../api/client', () => ({
  ...jest.requireActual('../api/client'),
  safeApiCall: jest.fn()
}));

// Wrap component with required providers
const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <ErrorProvider>
        {ui}
      </ErrorProvider>
    </BrowserRouter>
  );
};

describe('Dashboard Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.resetAllMocks();
    
    // Setup mock API responses
    (apiClient.safeApiCall as jest.Mock).mockImplementation((fn, options) => {
      // Check the function being called and return appropriate mock data
      const fnString = fn.toString();
      
      if (fnString.includes('healthService.check')) {
        return Promise.resolve({ data: mockHealthStatus });
      } else if (fnString.includes('gpuService.info')) {
        return Promise.resolve({ data: mockGPUInfo });
      } else if (fnString.includes('analyticsService.getRecentQueries')) {
        return Promise.resolve({ data: mockRecentQueries });
      } else if (fnString.includes('analyticsService.getPerformanceStats')) {
        return Promise.resolve({ data: mockPerformanceStats });
      } else if (fnString.includes('analyticsService.getPerformanceComparison')) {
        return Promise.resolve({ data: mockPerformanceComparison });
      }
      
      // Fallback to options.fallback if provided
      return Promise.resolve(options?.fallback);
    });
  });
  
  test('renders dashboard title', async () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });
  
  test('displays system status when data is loaded', async () => {
    renderWithProviders(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('All Systems Operational')).toBeInTheDocument();
    });
  });
  
  test('displays GPU information when available', async () => {
    renderWithProviders(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/GPU/i)).toBeInTheDocument();
      expect(screen.getByText(`${mockGPUInfo.device_count} GPUs Available`)).toBeInTheDocument();
    });
  });
  
  test('displays recent queries', async () => {
    renderWithProviders(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Recent Queries')).toBeInTheDocument();
      // Check if the first query from mock data is displayed
      expect(screen.getByText(mockRecentQueries[0].query)).toBeInTheDocument();
    });
  });
  
  test('displays performance comparison chart', async () => {
    renderWithProviders(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('GPU vs CPU Performance')).toBeInTheDocument();
    });
  });
  
  test('handles refresh button click', async () => {
    renderWithProviders(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });
    
    // Find and click the refresh button
    const refreshButton = screen.getByRole('button', { name: /refresh data/i });
    refreshButton.click();
    
    // Verify that API calls were made
    await waitFor(() => {
      expect(apiClient.safeApiCall).toHaveBeenCalledTimes(5);
    });
  });
});