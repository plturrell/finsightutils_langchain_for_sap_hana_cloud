import { useCallback, useEffect } from 'react';
import { AxiosError } from 'axios';
import { useError } from '../context/ErrorContext';
import { ApiError } from '../components/ErrorHandler';
import { setGlobalErrorHandler } from '../api/client';

// Define more specific error categories for better handling
export enum ErrorType {
  NETWORK = 'network',
  AUTH = 'auth',
  SERVER = 'server',
  VALIDATION = 'validation',
  UNEXPECTED = 'unexpected',
  API_TIMEOUT = 'api_timeout',
  CLIENT = 'client'
}

// Map HTTP status codes to error types for easier handling
const getErrorTypeFromStatus = (status: number): ErrorType => {
  if (status === 0) return ErrorType.NETWORK;
  if (status === 401 || status === 403) return ErrorType.AUTH;
  if (status === 422) return ErrorType.VALIDATION;
  if (status >= 500) return ErrorType.SERVER;
  if (status >= 400) return ErrorType.CLIENT;
  return ErrorType.UNEXPECTED;
};

// Format errors based on their type
const formatErrorByType = (type: ErrorType, message: string): { message: string; suggestions: string[] } => {
  switch (type) {
    case ErrorType.NETWORK:
      return {
        message: message || 'Network connection issue',
        suggestions: [
          'Check your internet connection',
          'Verify that the API server is running',
          'Try again in a few moments'
        ]
      };
    case ErrorType.AUTH:
      return {
        message: message || 'Authentication required',
        suggestions: [
          'Sign in to your account',
          'Your session may have expired',
          'Contact administrator if you should have access'
        ]
      };
    case ErrorType.SERVER:
      return {
        message: message || 'Server error occurred',
        suggestions: [
          'Wait a few minutes and try again',
          'Contact support if the problem persists',
          'Check system status for any outages'
        ]
      };
    case ErrorType.VALIDATION:
      return {
        message: message || 'Invalid data provided',
        suggestions: [
          'Check the form for errors',
          'Ensure all required fields are filled',
          'Verify data format is correct'
        ]
      };
    case ErrorType.API_TIMEOUT:
      return {
        message: message || 'Request timed out',
        suggestions: [
          'The server is taking too long to respond',
          'Try again with a smaller request',
          'Contact support if the problem persists'
        ]
      };
    case ErrorType.CLIENT:
      return {
        message: message || 'Client request error',
        suggestions: [
          'Check your input parameters',
          'Review the API documentation',
          'Verify request headers and format'
        ]
      };
    case ErrorType.UNEXPECTED:
    default:
      return {
        message: message || 'An unexpected error occurred',
        suggestions: [
          'Refresh the page and try again',
          'Clear your browser cache',
          'Contact support if the problem persists'
        ]
      };
  }
};

const useErrorHandler = () => {
  const { setError, clearError } = useError();

  const handleError = useCallback((error: unknown) => {
    if (error instanceof Error) {
      const axiosError = error as AxiosError;
      
      if (axiosError.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        const { status, statusText, data } = axiosError.response;
        const errorType = getErrorTypeFromStatus(status);
        
        // Extract error message
        let errorMessage = 'An error occurred';
        let errorDetails = {};
        
        if (typeof data === 'object' && data !== null) {
          if (data.detail && typeof data.detail === 'object') {
            errorMessage = data.detail.message || errorMessage;
            errorDetails = data.detail;
          } else if (data.message) {
            errorMessage = data.message;
            errorDetails = { message: data.message };
          } else {
            errorDetails = data;
          }
        } else if (typeof data === 'string') {
          errorMessage = data;
          errorDetails = { message: data };
        }
        
        // Get formatted error info based on type
        const { message, suggestions } = formatErrorByType(errorType, errorMessage);
        
        setError({
          status,
          statusText,
          detail: {
            ...errorDetails,
            message,
            suggestions,
            error_type: errorType,
            original_error: JSON.stringify(data)
          }
        });
      } else if (axiosError.request) {
        // The request was made but no response was received
        const isTimeout = axiosError.message.includes('timeout');
        const errorType = isTimeout ? ErrorType.API_TIMEOUT : ErrorType.NETWORK;
        const { message, suggestions } = formatErrorByType(errorType, axiosError.message);
        
        setError({
          status: 0,
          statusText: isTimeout ? 'Request Timeout' : 'No Response',
          detail: {
            message,
            suggestions,
            error_type: errorType,
            original_error: axiosError.message
          }
        });
      } else {
        // Something happened in setting up the request
        const { message, suggestions } = formatErrorByType(ErrorType.CLIENT, axiosError.message);
        
        setError({
          status: 0,
          statusText: 'Request Failed',
          detail: {
            message,
            suggestions,
            error_type: ErrorType.CLIENT,
            original_error: axiosError.message
          }
        });
      }
    } else {
      // Handle non-Error objects
      const { message, suggestions } = formatErrorByType(ErrorType.UNEXPECTED, String(error));
      
      setError({
        status: 0,
        statusText: 'Unknown Error',
        detail: {
          message,
          suggestions,
          error_type: ErrorType.UNEXPECTED,
          original_error: String(error)
        }
      });
    }
  }, [setError]);

  // Register the error handler globally
  useEffect(() => {
    setGlobalErrorHandler(handleError);
    return () => setGlobalErrorHandler(() => {});
  }, [handleError]);

  // Create a "safe" fetch function that automatically handles errors
  const safeFetch = useCallback(async <T>(
    fetchFn: () => Promise<T>,
    options?: {
      fallback?: T;
      silentError?: boolean;
      customErrorHandler?: (err: any) => void;
    }
  ): Promise<T | undefined> => {
    try {
      return await fetchFn();
    } catch (err) {
      // Call custom handler if provided
      if (options?.customErrorHandler) {
        options.customErrorHandler(err);
      }
      
      // Set global error if not silent
      if (!options?.silentError) {
        handleError(err);
      }
      
      // Return fallback value if provided
      return options?.fallback;
    }
  }, [handleError]);

  return { 
    handleError, 
    clearError,
    safeFetch,
    ErrorType 
  };
};

export default useErrorHandler;