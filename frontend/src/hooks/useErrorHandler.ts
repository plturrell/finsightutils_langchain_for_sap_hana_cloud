import { useCallback } from 'react';
import { AxiosError } from 'axios';
import { useError } from '../context/ErrorContext';
import { ApiError } from '../components/ErrorHandler';

const useErrorHandler = () => {
  const { setError, clearError } = useError();

  const handleError = useCallback((error: unknown) => {
    if (error instanceof Error) {
      const axiosError = error as AxiosError;
      
      if (axiosError.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        const { status, statusText, data } = axiosError.response;
        
        setError({
          status: status,
          statusText: statusText,
          detail: typeof data === 'object' ? data.detail : { message: String(data) }
        });
      } else if (axiosError.request) {
        // The request was made but no response was received
        setError({
          status: 0,
          statusText: 'No Response',
          detail: {
            message: 'The server did not respond. Please check your network connection and try again.',
            suggestions: [
              'Verify that the API server is running.',
              'Check your network connection.',
              'Try the operation again after a brief delay.'
            ]
          }
        });
      } else {
        // Something happened in setting up the request
        setError({
          status: 0,
          statusText: 'Request Failed',
          detail: {
            message: axiosError.message || 'An unexpected error occurred.',
            suggestions: [
              'Check your network connection.',
              'Try refreshing the page.',
              'Contact support if the problem persists.'
            ]
          }
        });
      }
    } else {
      // Handle non-Error objects
      setError({
        status: 0,
        statusText: 'Unknown Error',
        detail: {
          message: String(error) || 'An unexpected error occurred.',
          suggestions: [
            'Try refreshing the page.',
            'Clear your browser cache and cookies.',
            'Contact support if the problem persists.'
          ]
        }
      });
    }
  }, [setError]);

  return { handleError, clearError };
};

export default useErrorHandler;