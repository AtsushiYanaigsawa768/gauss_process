@echo off
REM FIR-based Grid Search Example (Windows)
REM This script demonstrates how to use FIR-based grid search for GP hyperparameter optimization

echo ========================================
echo FIR-based Grid Search Example
echo ========================================
echo.

REM Configuration
set INPUT_FILES=input\*.mat
set VALIDATION_FILE=input\input_test_20250912_165937.mat
set OUTPUT_DIR=output_fir_grid_search
set KERNEL=rbf
set N_FILES=1
set ND=50
set FIR_LENGTH=1024
set GRID_MAX_COMBINATIONS=500

echo Configuration:
echo   Input files: %INPUT_FILES%
echo   Validation file: %VALIDATION_FILE%
echo   Output directory: %OUTPUT_DIR%
echo   Kernel: %KERNEL%
echo   Number of files: %N_FILES%
echo   Frequency points (nd): %ND%
echo   FIR length: %FIR_LENGTH%
echo   Max grid combinations: %GRID_MAX_COMBINATIONS%
echo.

REM Check if validation file exists
if not exist "%VALIDATION_FILE%" (
    echo Error: Validation file not found: %VALIDATION_FILE%
    echo Please update VALIDATION_FILE in this script.
    exit /b 1
)

echo ========================================
echo Example 1: Traditional Grid Search (NLL-based)
echo ========================================
echo.

python src\unified_pipeline.py %INPUT_FILES% ^
    --n-files %N_FILES% ^
    --nd %ND% ^
    --kernel %KERNEL% ^
    --normalize ^
    --log-frequency ^
    --optimize ^
    --grid-search ^
    --grid-search-max-combinations %GRID_MAX_COMBINATIONS% ^
    --extract-fir ^
    --fir-length %FIR_LENGTH% ^
    --fir-validation-mat %VALIDATION_FILE% ^
    --out-dir %OUTPUT_DIR%_traditional

echo.
echo Traditional grid search complete!
echo Results saved to: %OUTPUT_DIR%_traditional
echo.

echo ========================================
echo Example 2: FIR-based Grid Search (RMSE-based)
echo ========================================
echo.

python src\unified_pipeline.py %INPUT_FILES% ^
    --n-files %N_FILES% ^
    --nd %ND% ^
    --kernel %KERNEL% ^
    --normalize ^
    --log-frequency ^
    --optimize ^
    --grid-search ^
    --use-fir-grid-search ^
    --grid-search-max-combinations %GRID_MAX_COMBINATIONS% ^
    --extract-fir ^
    --fir-length %FIR_LENGTH% ^
    --fir-validation-mat %VALIDATION_FILE% ^
    --out-dir %OUTPUT_DIR%_fir_based

echo.
echo FIR-based grid search complete!
echo Results saved to: %OUTPUT_DIR%_fir_based
echo.

echo ========================================
echo Example 3: Comparison of Different Kernels
echo ========================================
echo.

REM Test RBF kernel
echo Testing kernel: rbf
python src\unified_pipeline.py %INPUT_FILES% ^
    --n-files %N_FILES% ^
    --nd %ND% ^
    --kernel rbf ^
    --normalize ^
    --log-frequency ^
    --optimize ^
    --grid-search ^
    --use-fir-grid-search ^
    --grid-search-max-combinations %GRID_MAX_COMBINATIONS% ^
    --extract-fir ^
    --fir-length %FIR_LENGTH% ^
    --fir-validation-mat %VALIDATION_FILE% ^
    --out-dir %OUTPUT_DIR%_kernel_rbf
echo Kernel rbf complete!
echo.

REM Test Matern 3/2 kernel
echo Testing kernel: matern32
python src\unified_pipeline.py %INPUT_FILES% ^
    --n-files %N_FILES% ^
    --nd %ND% ^
    --kernel matern32 ^
    --normalize ^
    --log-frequency ^
    --optimize ^
    --grid-search ^
    --use-fir-grid-search ^
    --grid-search-max-combinations %GRID_MAX_COMBINATIONS% ^
    --extract-fir ^
    --fir-length %FIR_LENGTH% ^
    --fir-validation-mat %VALIDATION_FILE% ^
    --out-dir %OUTPUT_DIR%_kernel_matern32
echo Kernel matern32 complete!
echo.

REM Test Matern 5/2 kernel
echo Testing kernel: matern52
python src\unified_pipeline.py %INPUT_FILES% ^
    --n-files %N_FILES% ^
    --nd %ND% ^
    --kernel matern52 ^
    --normalize ^
    --log-frequency ^
    --optimize ^
    --grid-search ^
    --use-fir-grid-search ^
    --grid-search-max-combinations %GRID_MAX_COMBINATIONS% ^
    --extract-fir ^
    --fir-length %FIR_LENGTH% ^
    --fir-validation-mat %VALIDATION_FILE% ^
    --out-dir %OUTPUT_DIR%_kernel_matern52
echo Kernel matern52 complete!
echo.

echo ========================================
echo All examples complete!
echo ========================================
echo.
echo Results saved to:
echo   - %OUTPUT_DIR%_traditional (NLL-based)
echo   - %OUTPUT_DIR%_fir_based (RMSE-based)
echo   - %OUTPUT_DIR%_kernel_rbf (RMSE-based, rbf kernel)
echo   - %OUTPUT_DIR%_kernel_matern32 (RMSE-based, matern32 kernel)
echo   - %OUTPUT_DIR%_kernel_matern52 (RMSE-based, matern52 kernel)
echo.
echo Compare the FIR RMSE values to see the improvement!
echo.
pause
