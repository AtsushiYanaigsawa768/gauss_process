@echo off
REM Correct Train/Test Split Example
REM This script demonstrates the CORRECT way to separate training and test data

echo ========================================================================
echo CORRECT Train/Test Split Example
echo ========================================================================
echo.

REM Configuration
set INPUT_PATTERN=input\*.mat
set TEST_FILE=input\input_test_20250913_010037.mat
set OUTPUT_DIR=output_correct_split
set N_FILES=5
set ND=100

echo Configuration:
echo   Input pattern: %INPUT_PATTERN% (multiple files)
echo   Test file: %TEST_FILE% (excluded from training)
echo   Number of files to use: %N_FILES%
echo   Frequency points: %ND%
echo.

REM Check if test file exists
if not exist "%TEST_FILE%" (
    echo Error: Test file not found: %TEST_FILE%
    exit /b 1
)

echo ========================================================================
echo Example 1: Grid Search with Proper Train/Test Split
echo ========================================================================
echo.
echo Expected behavior:
echo   - System will select first %N_FILES% files from %INPUT_PATTERN%
echo   - System will automatically EXCLUDE %TEST_FILE% from training
echo   - Training will use %N_FILES% - 1 = 4 files
echo   - Testing will use 1 file: %TEST_FILE%
echo.

python src\unified_pipeline.py %INPUT_PATTERN% ^
    --n-files %N_FILES% ^
    --nd %ND% ^
    --kernel rbf ^
    --normalize ^
    --grid-search ^
    --fir-validation-mat %TEST_FILE% ^
    --extract-fir ^
    --fir-length 1024 ^
    --out-dir %OUTPUT_DIR%\example1

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! Grid search completed with proper train/test split.
    echo Results saved to: %OUTPUT_DIR%\example1
    echo.
) else (
    echo.
    echo FAILED! Check error messages above.
    echo.
    exit /b 1
)

echo ========================================================================
echo Example 2: Using Different Files for Training and Testing
echo ========================================================================
echo.

set TRAIN_FILE=input\input_test_20250913_030050.mat
set TEST_FILE2=input\input_test_20250913_010037.mat

echo   Training file: %TRAIN_FILE%
echo   Test file: %TEST_FILE2%
echo.

python src\unified_pipeline.py %TRAIN_FILE% ^
    --n-files 1 ^
    --time-duration 1800 ^
    --nd %ND% ^
    --kernel rbf ^
    --normalize ^
    --grid-search ^
    --fir-validation-mat %TEST_FILE2% ^
    --extract-fir ^
    --fir-length 1024 ^
    --out-dir %OUTPUT_DIR%\example2

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! Grid search completed with different files.
    echo Results saved to: %OUTPUT_DIR%\example2
    echo.
) else (
    echo.
    echo FAILED! Check error messages above.
    echo.
    exit /b 1
)

echo ========================================================================
echo Example 3: WRONG WAY (This will fail as expected)
echo ========================================================================
echo.
echo This demonstrates what NOT to do:
echo   Training: %TEST_FILE%
echo   Testing:  %TEST_FILE% (SAME FILE - WRONG!)
echo.
echo Expected: ERROR message about data leakage
echo.

python src\unified_pipeline.py %TEST_FILE% ^
    --n-files 1 ^
    --nd %ND% ^
    --kernel rbf ^
    --fir-validation-mat %TEST_FILE% ^
    --out-dir %OUTPUT_DIR%\example3_wrong

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo EXPECTED FAILURE! System correctly rejected same file for train/test.
    echo This is the correct behavior to prevent data leakage.
    echo.
) else (
    echo.
    echo WARNING: This should have failed! Check system configuration.
    echo.
)

echo ========================================================================
echo All Examples Complete
echo ========================================================================
echo.
echo Summary:
echo   Example 1: CORRECT - Multiple files with automatic exclusion
echo   Example 2: CORRECT - Explicit different files
echo   Example 3: EXPECTED FAILURE - Same file (demonstrates error handling)
echo.
echo See TRAIN_TEST_SPLIT_GUIDE.md for more information.
echo.
pause
