#!/bin/bash
# Run the financial system with FinMTEB/Fin-E5 model

# SAP HANA Cloud connection parameters
HANA_HOST="d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com"
HANA_PORT=443
HANA_USER="DBADMIN"
HANA_PASSWORD="Initial@1"

# Model parameters
MODEL_NAME="FinMTEB/Fin-E5"
MODELS_DIR="./financial_models"

# Check if using fine-tuned model flag is set
USE_FINE_TUNED=false
if [ "$1" == "--use-fine-tuned" ]; then
  USE_FINE_TUNED=true
  shift
  
  # Check if fine-tuned model path exists
  if [ -f "fin_e5_tuned_model_path.txt" ]; then
    MODEL_NAME=$(cat fin_e5_tuned_model_path.txt)
    echo "Using fine-tuned model: $MODEL_NAME"
  else
    echo "Error: Fine-tuned model path not found. Run fine-tuning first."
    exit 1
  fi
fi

# Check if command is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 [--use-fine-tuned] [add|query|metrics|health] [additional options]"
  echo ""
  echo "Examples:"
  echo "  $0 add --input-file documents.json"
  echo "  $0 query --input-file queries.json --output-file results.json"
  echo "  $0 --use-fine-tuned query --input-file queries.json"
  echo "  $0 metrics"
  echo "  $0 health"
  exit 1
fi

COMMAND=$1
shift

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Run the system with the selected model
python run_financial_system.py \
  --host "$HANA_HOST" \
  --port "$HANA_PORT" \
  --user "$HANA_USER" \
  --password "$HANA_PASSWORD" \
  --model-name "$MODEL_NAME" \
  --local-model \
  --models-dir "$MODELS_DIR" \
  --auto-download \
  "$COMMAND" "$@"