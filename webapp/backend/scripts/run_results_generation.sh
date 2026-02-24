#!/bin/bash
# Quick script to generate all results for IEEE paper

echo "=========================================="
echo "Generating Results for IEEE Paper"
echo "=========================================="

# Navigate to scripts directory
cd "$(dirname "$0")"

# Install dependencies if needed
echo ""
echo "Checking dependencies..."
pip install -q -r requirements_results.txt 2>/dev/null || echo "Note: Some dependencies may need manual installation"

# Create results directory
mkdir -p ../../results/figures
mkdir -p ../../results/tables

# Run all generation scripts
echo ""
echo "Running comprehensive results generation..."
python generate_all_results.py \
    --csv ../../oversampled_data.csv \
    --output-dir ../../results \
    --model-dir ../../model

echo ""
echo "=========================================="
echo "Results generation complete!"
echo "Check ../../results/ for all outputs"
echo "=========================================="
