#!/bin/bash
# Automated testing script for SAP HANA Cloud LangChain integration on T4 GPU

# Default settings
CONFIG_FILE="test_config.json"
RESULTS_DIR="test_results"
RUN_ALL=false
SPECIFIC_SUITE=""
VERBOSE=false

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help             Show this help message and exit"
    echo "  -c, --config FILE      Path to configuration file (default: test_config.json)"
    echo "  -r, --results-dir DIR  Directory to store test results (default: test_results)"
    echo "  -a, --all              Run all test suites"
    echo "  -s, --suite SUITE      Run specific test suite"
    echo "  -l, --list-suites      List available test suites"
    echo "  -v, --verbose          Enable verbose output"
    echo ""
    echo "Available test suites:"
    echo "  environment            Verify environment setup (GPU, drivers, Python packages)"
    echo "  tensorrt               Test TensorRT optimization for T4 GPU"
    echo "  vectorstore            Test vector store functionality"
    echo "  gpu_performance        Benchmark GPU performance for embedding and vector operations"
    echo "  error_handling         Test error handling and recovery"
    echo "  api                    Test API endpoints"
    echo ""
    echo "Example:"
    echo "  $0 --all                     Run all test suites"
    echo "  $0 --suite gpu_performance   Run only GPU performance tests"
}

# Function to create a default configuration file if it doesn't exist
create_default_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Creating default configuration file: $CONFIG_FILE"
        cat > "$CONFIG_FILE" << EOF
{
    "api_base_url": "https://jupyter0-513syzm60.brevlab.com",
    "results_dir": "$RESULTS_DIR",
    "test_timeout": 300,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "precision": "fp16",
    "batch_sizes": [1, 8, 32, 64, 128],
    "auth": {
        "enabled": false,
        "username": "",
        "password": ""
    },
    "hana_connection": {
        "address": "",
        "port": 0,
        "user": "",
        "password": ""
    }
}
EOF
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -s|--suite)
            SPECIFIC_SUITE="$2"
            shift 2
            ;;
        -l|--list-suites)
            python run_automated_tests.py --list-suites
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Create default configuration file if it doesn't exist
create_default_config

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found in PATH"
    exit 1
fi

# Check for required Python packages
echo "Checking for required Python packages..."
REQUIRED_PACKAGES="requests numpy"
for package in $REQUIRED_PACKAGES; do
    if ! python3 -c "import $package" &> /dev/null; then
        echo "Installing required package: $package"
        python3 -m pip install $package
    fi
done

# Run the tests
if [ "$RUN_ALL" = true ]; then
    echo "Running all test suites..."
    if [ "$VERBOSE" = true ]; then
        python3 run_automated_tests.py --all --config "$CONFIG_FILE" --results-dir "$RESULTS_DIR"
    else
        python3 run_automated_tests.py --all --config "$CONFIG_FILE" --results-dir "$RESULTS_DIR" 2>&1 | grep -v "DEBUG"
    fi
elif [ -n "$SPECIFIC_SUITE" ]; then
    echo "Running test suite: $SPECIFIC_SUITE"
    if [ "$VERBOSE" = true ]; then
        python3 run_automated_tests.py --suite "$SPECIFIC_SUITE" --config "$CONFIG_FILE" --results-dir "$RESULTS_DIR"
    else
        python3 run_automated_tests.py --suite "$SPECIFIC_SUITE" --config "$CONFIG_FILE" --results-dir "$RESULTS_DIR" 2>&1 | grep -v "DEBUG"
    fi
else
    echo "No test suite specified. Use --all or --suite SUITE"
    show_help
    exit 1
fi

# Check the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n\033[0;32mTests completed successfully!\033[0m"
elif [ $EXIT_CODE -eq 2 ]; then
    echo -e "\n\033[0;33mTests completed with some failures. See logs for details.\033[0m"
else
    echo -e "\n\033[0;31mTests failed with errors. See logs for details.\033[0m"
fi

# Generate HTML report if any tests were run
if [ "$RUN_ALL" = true ] || [ -n "$SPECIFIC_SUITE" ]; then
    echo "Generating HTML report..."
    python3 -c "
import json
import os
import sys

# HTML template
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>SAP HANA Cloud LangChain Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #0066cc; }
        h2 { color: #333; margin-top: 20px; }
        .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .status-success { color: green; }
        .status-partial { color: orange; }
        .status-error, .status-failure { color: red; }
        .status-simulated { color: blue; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .recommendations { background-color: #e6f7ff; padding: 15px; border-radius: 5px; }
        .performance { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>SAP HANA Cloud LangChain Integration Test Report</h1>
    <div class='summary'>
        <h2>Summary</h2>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>API Base URL:</strong> {api_base_url}</p>
        <p><strong>Total Tests:</strong> {total_tests}</p>
        <p><strong>Passed:</strong> <span class='status-success'>{passed}</span></p>
        <p><strong>Simulated:</strong> <span class='status-simulated'>{simulated}</span></p>
        <p><strong>Failed:</strong> <span class='status-failure'>{failed}</span></p>
        <p><strong>Error:</strong> <span class='status-error'>{error}</span></p>
    </div>
    
    <h2>Test Suite Status</h2>
    <table>
        <tr>
            <th>Suite</th>
            <th>Status</th>
        </tr>
        {suite_status_rows}
    </table>
    
    <div class='performance'>
        <h2>Performance Metrics</h2>
        <p><strong>Average GPU Speedup:</strong> {gpu_speedup}x</p>
        <p><strong>Optimal Batch Size:</strong> {optimal_batch_size}</p>
        <p><strong>MMR Speedup:</strong> {mmr_speedup}x</p>
    </div>
    
    <div class='recommendations'>
        <h2>Recommendations</h2>
        <ul>
            {recommendations}
        </ul>
    </div>
</body>
</html>
'''

# Read the report data
report_path = os.path.join('$RESULTS_DIR', 'test_report.json')
if not os.path.exists(report_path):
    print('No test report found')
    sys.exit(1)

with open(report_path, 'r') as f:
    report = json.load(f)

# Generate suite status rows
suite_status_rows = ''
for suite_name, status in report.get('suite_status', {}).items():
    suite_status_rows += f'<tr><td>{suite_name}</td><td class=\"status-{status}\">{status}</td></tr>'

# Generate recommendations list
recommendations_html = ''
for recommendation in report.get('recommendations', []):
    recommendations_html += f'<li>{recommendation}</li>'

# Format performance metrics
gpu_speedup = f\"{report.get('performance', {}).get('gpu_speedup', 'N/A'):.2f}\" if report.get('performance', {}).get('gpu_speedup') else 'N/A'
mmr_speedup = f\"{report.get('performance', {}).get('mmr_speedup', 'N/A'):.2f}\" if report.get('performance', {}).get('mmr_speedup') else 'N/A'
optimal_batch_size = report.get('performance', {}).get('optimal_batch_size', 'N/A')

# Generate HTML
html = html_template.format(
    timestamp=report.get('timestamp', 'N/A'),
    api_base_url=report.get('api_base_url', 'N/A'),
    total_tests=report.get('summary', {}).get('total_tests', 0),
    passed=report.get('summary', {}).get('passed', 0),
    simulated=report.get('summary', {}).get('simulated', 0),
    failed=report.get('summary', {}).get('failed', 0),
    error=report.get('summary', {}).get('error', 0),
    suite_status_rows=suite_status_rows,
    gpu_speedup=gpu_speedup,
    optimal_batch_size=optimal_batch_size,
    mmr_speedup=mmr_speedup,
    recommendations=recommendations_html
)

# Write HTML report
html_report_path = os.path.join('$RESULTS_DIR', 'test_report.html')
with open(html_report_path, 'w') as f:
    f.write(html)

print(f'HTML report generated: {html_report_path}')
" || echo "Failed to generate HTML report"

    echo "Test results available in: $RESULTS_DIR"
    echo "HTML report: $RESULTS_DIR/test_report.html"
fi

exit $EXIT_CODE