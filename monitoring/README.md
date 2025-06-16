# FinSight API Monitoring System

## Overview

We have created a comprehensive API monitoring system for the FinSight services. This system includes:

1. An enhanced API testing framework that tests real endpoints
2. A continuous monitoring service that runs tests at regular intervals
3. A visualization dashboard for monitoring API health

All services are now returning 200 OK responses for their core endpoints, and the tests correctly handle specialized endpoints that return 404.

## Components

### Enhanced API Testing Framework (`enhanced_api_test.py`)

This script performs real tests against all service endpoints and generates detailed reports:

- Tests core endpoints (`/`, `/health`, `/docs`, `/openapi.json`)
- Tests specialized endpoints (with 404 expected)
- Generates detailed Markdown, CSV, and JSON reports
- Includes response data and timing information
- Implements retry logic for reliable testing

### Continuous Monitoring Service (`api_monitor.py`)

This script runs the API tests at regular intervals and tracks service health over time:

- Configurable test interval (default: 5 minutes)
- Stores test history for trend analysis
- Generates alerts when success rates fall below thresholds
- Writes logs and reports to a specified directory
- Can be run once or as a continuous service

### API Health Dashboard (`api_dashboard.html`)

A web-based dashboard for visualizing API health:

- Real-time overview of all service success rates
- Charts showing success rates over time
- Service-specific health details
- Alerts for any services with issues
- Auto-refreshes to show the latest data

### Helper Scripts

- `start_monitoring.sh`: Starts the monitoring service and dashboard server
- `stop_monitoring.sh`: Stops all monitoring services

## Services Being Monitored

The following services are being monitored:

1. **AIQToolkit** (port 8000)
2. **OWL** (port 8001)
3. **Dynamo** (port 8002)
4. **DSPy** (port 8003)
5. **NeMo** (port 8004)
6. **SAP HANA Cloud Utilities** (port 8005)

## Usage Instructions

### Running a Single Test

To run a single test and generate reports:

```bash
python3 enhanced_api_test.py
```

### Starting Continuous Monitoring

To start the continuous monitoring service and dashboard:

```bash
./start_monitoring.sh
```

This will:
- Start the API monitor in the background (runs every 5 minutes)
- Start a web server for the dashboard on port 8080
- Create a `stop_monitoring.sh` script

### Accessing the Dashboard

Once the monitoring service is running, access the dashboard at:

```
http://localhost:8080/api_dashboard.html
```

### Stopping Monitoring

To stop all monitoring services:

```bash
./stop_monitoring.sh
```

## Current Status

All services are healthy and responding with 200 OK status codes for their core endpoints. The specialized endpoints are correctly returning 404 status codes, which is expected and handled by the testing framework.

The monitoring system is successfully collecting real data from all services and providing accurate reports.