# SAP HANA LangChain Integration Frontend

This is the frontend application for the SAP HANA Cloud LangChain integration, providing a visual interface for vector search, development, and debugging.

## Error Handling System

### Overview

The application includes a comprehensive error handling system that transforms backend errors into user-friendly, context-aware messages. This system:

- Captures and formats API errors with consistent structure
- Displays context-specific error messages and suggestions
- Integrates with our backend's error interpretation system
- Provides operation-specific suggestions to help users resolve issues
- Presents technical details for debugging when needed

### Key Components

#### 1. ErrorHandler Component (`/src/components/ErrorHandler.tsx`)

A reusable React component that displays errors with:
- Clear, concise error messages
- Contextual suggestions for resolving the issue
- Common issues related to the operation
- Technical details that can be expanded when needed
- Contextual operation information

#### 2. Error Context (`/src/context/ErrorContext.tsx`)

A React context that:
- Manages global error state
- Provides error setter and clearing functions
- Makes errors accessible throughout the application

#### 3. Error Hook (`/src/hooks/useErrorHandler.ts`)

A custom hook that:
- Standardizes error handling across components
- Integrates with the Error Context
- Formats different types of errors consistently
- Provides handleError and clearError functions

#### 4. API Client Error Interceptor (`/src/api/client.ts`)

Axios interceptor that:
- Standardizes error responses
- Ensures all errors have consistent structure
- Adds helpful information to network and request errors

### How to Use

#### 1. Basic Usage in a Component

```tsx
import React, { useState } from 'react';
import useErrorHandler from '../hooks/useErrorHandler';
import { developerService } from '../api/services';

const MyComponent = () => {
  const { handleError, clearError } = useErrorHandler();
  
  const fetchData = async () => {
    clearError(); // Clear any previous errors
    
    try {
      const response = await developerService.someApiCall();
      // Handle success
    } catch (error) {
      console.error("Error:", error);
      handleError(error);
    }
  };
  
  return (
    <div>
      {/* Component content */}
      <button onClick={fetchData}>Fetch Data</button>
    </div>
  );
};
```

#### 2. Handling Different Error Types

The system automatically handles different types of errors:

- **HTTP Errors**: Status codes, error messages from backend
- **Network Errors**: Connection issues, timeouts
- **Request Errors**: Invalid request parameters

#### 3. Error Display

Errors are automatically displayed in the main layout when using the `useErrorHandler` hook, as the Layout component includes the `ErrorHandler` component which reads from the global error context.

### Backend Integration

This frontend error handling system is designed to work with our backend's error interpretation system in `/api/error_utils.py`, which provides:

- SQL error pattern recognition and interpretation
- Operation-specific context and suggestions
- Detailed error information for debugging

The backend returns errors with this structure:

```json
{
  "detail": {
    "message": "The requested table doesn't exist in the database.",
    "operation": "Vector Search",
    "suggestions": [
      "Check the table name for typos.",
      "Verify that the table has been created.",
      "Ensure you have the correct schema name if using schema.table notation."
    ],
    "common_issues": [
      "Missing or invalid vector index",
      "Incorrect vector dimensionality", 
      "Table doesn't contain the expected vector column"
    ],
    "original_error": "table 'EMBEDDINGS' does not exist"
  }
}
```

The frontend error handler parses this structure and displays it in a user-friendly way.

## Error Examples

### Database Connection Error

![Database Connection Error](images/error-connection.png)

### Vector Search Error

![Vector Search Error](images/error-vector-search.png)

### API Request Error

![API Request Error](images/error-request.png)

## Next Steps

- Add error analytics to track common issues
- Implement error persistence for debugging sessions
- Add more specialized error handlers for different operations