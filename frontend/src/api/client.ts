import axios from 'axios';

// Create a base Axios instance with default config
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Add a request interceptor for handling auth and other preprocessing
apiClient.interceptors.request.use(
  (config) => {
    // You can add authentication tokens here if needed
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor for handling errors and response formatting
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Standardize and log error information
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('API Error Response:', error.response.data);
      
      // Format the error to match our error handler component expectations
      if (error.response.data && typeof error.response.data === 'object') {
        // If the error already has a detail property, make sure it's properly formatted
        if (!error.response.data.detail) {
          error.response.data = { 
            detail: error.response.data 
          };
        }
      } else {
        // If the error response is not an object or doesn't have expected structure
        error.response.data = { 
          detail: { 
            message: typeof error.response.data === 'string' 
              ? error.response.data 
              : 'An unexpected error occurred', 
            original_error: JSON.stringify(error.response.data)
          } 
        };
      }
      
      // Handle specific status codes
      if (error.response.status === 401) {
        // Handle unauthorized error (e.g., redirect to login)
        console.log('Unauthorized access, please login');
        // You could trigger a redirect or auth flow here
      }
    } else if (error.request) {
      // The request was made but no response was received
      console.error('API No Response Error:', error.request);
      error.response = {
        status: 0,
        statusText: 'No Response',
        data: {
          detail: {
            message: 'The server did not respond. Please check your network connection and try again.',
            suggestions: [
              'Verify that the API server is running.',
              'Check your network connection.',
              'Try the operation again after a brief delay.'
            ]
          }
        }
      };
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('API Request Error:', error.message);
      error.response = {
        status: 0,
        statusText: 'Request Failed',
        data: {
          detail: {
            message: error.message || 'An unexpected error occurred.',
            suggestions: [
              'Check your network connection.',
              'Try refreshing the page.',
              'Contact support if the problem persists.'
            ]
          }
        }
      };
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;