#!/bin/bash
# Run fine-tuning for FinMTEB/Fin-E5 model with financial domain data

# Set output directory
OUTPUT_DIR="./fine_tuned_financial_models"
MODELS_DIR="./financial_models"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODELS_DIR"

echo "Starting fine-tuning process for FinMTEB/Fin-E5..."

# Run fine-tuning with prepared datasets
python finetune_fin_e5.py \
  --train-file financial_training_data.json \
  --val-file financial_validation_data.json \
  --training-format pairs \
  --base-model "FinMTEB/Fin-E5" \
  --models-dir "$MODELS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --output-model-name "FinMTEB-Fin-E5-financial-custom" \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --max-seq-length 256 \
  --evaluation-steps 10

echo "Fine-tuning complete! The model path has been saved to fin_e5_tuned_model_path.txt"
echo "To use the fine-tuned model, run:"
echo "  ./run_fin_e5.sh query --model-name \"\$(cat fin_e5_tuned_model_path.txt)\" --input-file queries.json"