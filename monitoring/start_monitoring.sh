#!/bin/bash

# Start the API monitoring service
# This script runs the API monitor in the background and starts a simple HTTP server for the dashboard

# Configuration
MONITOR_INTERVAL=300  # 5 minutes
LOG_DIR="monitor_logs"
HTTP_PORT=8080

# Create the logs directory if it doesn't exist
mkdir -p $LOG_DIR

# Start the API monitor in the background
echo "Starting API monitor with ${MONITOR_INTERVAL} second interval..."
nohup python3 api_monitor.py --interval $MONITOR_INTERVAL --log-dir $LOG_DIR > $LOG_DIR/monitor.log 2>&1 &
MONITOR_PID=$!
echo "API monitor started with PID $MONITOR_PID"
echo $MONITOR_PID > $LOG_DIR/monitor.pid

# Start a simple HTTP server for the dashboard
echo "Starting HTTP server for dashboard on port $HTTP_PORT..."
python3 -m http.server $HTTP_PORT &
HTTP_PID=$!
echo "HTTP server started with PID $HTTP_PID"
echo $HTTP_PID > $LOG_DIR/http.pid

echo "Monitoring services started successfully."
echo "Access the dashboard at http://localhost:$HTTP_PORT/api_dashboard.html"
echo "To stop the services, run: ./stop_monitoring.sh"

# Create a stop script if it doesn't exist
if [ ! -f "stop_monitoring.sh" ]; then
  cat > stop_monitoring.sh << EOF
#!/bin/bash

# Stop the API monitoring services

# Read PIDs from files
if [ -f "monitor_logs/monitor.pid" ]; then
  MONITOR_PID=\$(cat monitor_logs/monitor.pid)
  echo "Stopping API monitor (PID \$MONITOR_PID)..."
  kill \$MONITOR_PID 2>/dev/null || echo "API monitor is not running"
  rm monitor_logs/monitor.pid
fi

if [ -f "monitor_logs/http.pid" ]; then
  HTTP_PID=\$(cat monitor_logs/http.pid)
  echo "Stopping HTTP server (PID \$HTTP_PID)..."
  kill \$HTTP_PID 2>/dev/null || echo "HTTP server is not running"
  rm monitor_logs/http.pid
fi

echo "Monitoring services stopped."
EOF
  chmod +x stop_monitoring.sh
  echo "Created stop_monitoring.sh script"
fi