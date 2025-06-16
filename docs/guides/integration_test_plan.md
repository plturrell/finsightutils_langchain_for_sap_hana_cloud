# SAP HANA LangChain Integration Test Plan

## Overview
This test plan outlines the steps for testing the integration between SAP HANA Cloud and LangChain components. The system consists of an API service (backend) and a frontend application.

## Prerequisites
- Docker installed and running
- Access to Docker Hub (for pulling pre-built images)
- Basic understanding of the SAP HANA LangChain integration

## 1. Test API Service

### 1.1 Basic Health Check
- Start the API container:
  ```bash
  docker pull finsightintelligence/langchain-sap-hana:minimal-api
  docker run -d -p 8000:8000 -e TEST_MODE=true finsightintelligence/langchain-sap-hana:minimal-api
  ```
- Test basic health endpoints:
  ```bash
  curl http://localhost:8000/health
  curl http://localhost:8000/health/ping
  ```
- Expected results: Valid JSON responses with status "ok"

### 1.2 GPU Information
- Check GPU availability:
  ```bash
  curl http://localhost:8000/gpu/info
  ```
- Expected results: JSON response with GPU information or indication that GPU is not available

### 1.3 Vector Operations
- Test basic vector operations:
  ```bash
  # Add sample texts
  curl -X POST -H "Content-Type: application/json" -d '{"texts": ["This is a test document", "Another test document"]}' http://localhost:8000/texts
  
  # Query the vector store
  curl -X POST -H "Content-Type: application/json" -d '{"query": "test document", "k": 1}' http://localhost:8000/query
  ```
- Expected results: Successful addition of texts and retrieval of similar documents

### 1.4 Use the HTML Test Page
- Open the test.html file in a browser to perform interactive testing of the API
- Set the API URL to http://localhost:8000
- Test all basic endpoints and verify responses

## 2. Test Frontend

### 2.1 Build Frontend
- Build the frontend container:
  ```bash
  docker build -t finsightintelligence/langchain-hana-frontend:latest -f frontend/Dockerfile frontend/
  ```

### 2.2 Run Frontend
- Start the frontend container:
  ```bash
  docker run -d -p 3000:3000 -e REACT_APP_API_URL=http://localhost:8000 finsightintelligence/langchain-hana-frontend:latest
  ```
- Open browser at http://localhost:3000
- Verify the frontend loads correctly with all components

### 2.3 Frontend Functionality
- Test the dashboard visualization
- Test the search functionality
- Test the vector visualization components

## 3. Integration Tests

### 3.1 Combined Setup
- Start both containers using docker-compose:
  ```bash
  docker-compose -f docker-compose.test.yml up
  ```
- Verify both services are running:
  ```bash
  docker-compose -f docker-compose.test.yml ps
  ```

### 3.2 End-to-End Tests
- Open the frontend at http://localhost:3000
- Perform the following operations:
  1. Check dashboard displays correctly
  2. Run a sample search query
  3. Check API statistics are displayed
  4. Test GPU information display

### 3.3 Performance Tests
- Run basic performance tests using the benchmark endpoints:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"texts": ["This is a test document"], "iterations": 5}' http://localhost:8000/benchmark/embedding
  ```
- Check the response times and ensure they're within acceptable ranges

## 4. Security Tests

### 4.1 API Access Controls
- Verify that sensitive endpoints have proper access controls
- Test CORS settings are applied correctly

### 4.2 Error Handling
- Test API error handling by sending invalid requests:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"invalid": "request"}' http://localhost:8000/query
  ```
- Verify that errors are properly handled and reported

## 5. Cleanup
- Stop and remove all containers:
  ```bash
  docker-compose -f docker-compose.test.yml down
  docker rm -f $(docker ps -a -q --filter "ancestor=finsightintelligence/langchain-sap-hana:minimal-api")
  docker rm -f $(docker ps -a -q --filter "ancestor=finsightintelligence/langchain-hana-frontend:latest")
  ```

## Test Results
Document all test results, including:
- Successful test cases
- Failed test cases with details on the failures
- Performance metrics
- Recommendations for improvements

## Troubleshooting
If you encounter issues during testing:
1. Check container logs:
   ```bash
   docker logs [container_id]
   ```
2. Verify network connectivity between containers
3. Ensure all required environment variables are set correctly
4. Confirm the API URLs are correctly configured in the frontend