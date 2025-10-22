#!/bin/bash

# FIR-based Grid Search Example
# This script demonstrates how to use FIR-based grid search for GP hyperparameter optimization

echo "========================================"
echo "FIR-based Grid Search Example"
echo "========================================"
echo ""

# Configuration
INPUT_FILES="input/*.mat"
VALIDATION_FILE="input/input_test_20250912_165937.mat"  # Update with your validation file
OUTPUT_DIR="output_fir_grid_search"
KERNEL="rbf"
N_FILES=1
ND=50
FIR_LENGTH=1024

# Grid search configuration
GRID_MAX_COMBINATIONS=500  # Reduced for faster computation

echo "Configuration:"
echo "  Input files: ${INPUT_FILES}"
echo "  Validation file: ${VALIDATION_FILE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Kernel: ${KERNEL}"
echo "  Number of files: ${N_FILES}"
echo "  Frequency points (nd): ${ND}"
echo "  FIR length: ${FIR_LENGTH}"
echo "  Max grid combinations: ${GRID_MAX_COMBINATIONS}"
echo ""

# Check if validation file exists
if [ ! -f "${VALIDATION_FILE}" ]; then
    echo "Error: Validation file not found: ${VALIDATION_FILE}"
    echo "Please update VALIDATION_FILE in this script."
    exit 1
fi

echo "========================================"
echo "Example 1: Traditional Grid Search (NLL-based)"
echo "========================================"
echo ""

python src/unified_pipeline.py ${INPUT_FILES} \
    --n-files ${N_FILES} \
    --nd ${ND} \
    --kernel ${KERNEL} \
    --normalize \
    --log-frequency \
    --optimize \
    --grid-search \
    --grid-search-max-combinations ${GRID_MAX_COMBINATIONS} \
    --extract-fir \
    --fir-length ${FIR_LENGTH} \
    --fir-validation-mat ${VALIDATION_FILE} \
    --out-dir ${OUTPUT_DIR}_traditional

echo ""
echo "Traditional grid search complete!"
echo "Results saved to: ${OUTPUT_DIR}_traditional"
echo ""

echo "========================================"
echo "Example 2: FIR-based Grid Search (RMSE-based)"
echo "========================================"
echo ""

python src/unified_pipeline.py ${INPUT_FILES} \
    --n-files ${N_FILES} \
    --nd ${ND} \
    --kernel ${KERNEL} \
    --normalize \
    --log-frequency \
    --optimize \
    --grid-search \
    --use-fir-grid-search \
    --grid-search-max-combinations ${GRID_MAX_COMBINATIONS} \
    --extract-fir \
    --fir-length ${FIR_LENGTH} \
    --fir-validation-mat ${VALIDATION_FILE} \
    --out-dir ${OUTPUT_DIR}_fir_based

echo ""
echo "FIR-based grid search complete!"
echo "Results saved to: ${OUTPUT_DIR}_fir_based"
echo ""

echo "========================================"
echo "Example 3: Comparison of Different Kernels"
echo "========================================"
echo ""

# Test multiple kernels with FIR-based grid search
KERNELS=("rbf" "matern32" "matern52")

for kernel in "${KERNELS[@]}"; do
    echo "Testing kernel: ${kernel}"

    python src/unified_pipeline.py ${INPUT_FILES} \
        --n-files ${N_FILES} \
        --nd ${ND} \
        --kernel ${kernel} \
        --normalize \
        --log-frequency \
        --optimize \
        --grid-search \
        --use-fir-grid-search \
        --grid-search-max-combinations ${GRID_MAX_COMBINATIONS} \
        --extract-fir \
        --fir-length ${FIR_LENGTH} \
        --fir-validation-mat ${VALIDATION_FILE} \
        --out-dir ${OUTPUT_DIR}_kernel_${kernel}

    echo "Kernel ${kernel} complete!"
    echo ""
done

echo "========================================"
echo "All examples complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_DIR}_traditional (NLL-based)"
echo "  - ${OUTPUT_DIR}_fir_based (RMSE-based)"
for kernel in "${KERNELS[@]}"; do
    echo "  - ${OUTPUT_DIR}_kernel_${kernel} (RMSE-based, ${kernel} kernel)"
done
echo ""
echo "Compare the FIR RMSE values to see the improvement!"
