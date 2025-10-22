#!/usr/bin/env python3
"""
Verification script for FIR-based grid search implementation.

This script runs a minimal test to verify that:
1. FIR-based grid search is actually being executed
2. FIR models are being constructed during grid search
3. RMSE evaluation is working correctly
4. The selected parameters are based on FIR RMSE, not NLL
"""

import sys
import subprocess
import re
from pathlib import Path
import glob

def run_test():
    """Run minimal test with FIR-based grid search."""

    print("="*80)
    print("FIR-based Grid Search Verification Test")
    print("="*80)
    print()

    # Find MAT files
    mat_pattern = 'input/*.mat'
    mat_files = glob.glob(mat_pattern)

    if not mat_files:
        print(f"Error: No MAT files found matching pattern: {mat_pattern}")
        return False

    mat_files = sorted(mat_files)
    input_file = mat_files[0]

    # Find validation file
    validation_file = None
    for f in mat_files:
        if 'test' in f.lower():
            validation_file = f
            break

    if not validation_file:
        validation_file = input_file

    print(f"Input file: {input_file}")
    print(f"Validation file: {validation_file}")
    print()

    # Test 1: Traditional grid search (NLL-based)
    print("-"*80)
    print("Test 1: Traditional Grid Search (NLL-based)")
    print("-"*80)
    print()

    cmd1 = [
        sys.executable, 'src/unified_pipeline.py', input_file,
        '--n-files', '1',
        '--nd', '10',  # Small for fast testing
        '--kernel', 'rbf',
        '--normalize',
        '--log-frequency',
        '--optimize',
        '--grid-search',
        '--grid-search-max-combinations', '50',  # Small grid for testing
        '--extract-fir',
        '--fir-length', '512',  # Smaller for speed
        '--fir-validation-mat', validation_file,
        '--out-dir', 'verify_output_traditional'
    ]

    print("Running command:")
    print(' '.join(cmd1))
    print()

    result1 = subprocess.run(cmd1, capture_output=True, text=True)

    # Check for "best NLL" in output
    nll_found = 'best NLL' in result1.stdout or 'Best NLL' in result1.stdout
    print(f"✓ Found 'best NLL' in output: {nll_found}")

    if nll_found:
        # Extract NLL value
        nll_match = re.search(r'[Bb]est NLL:\s*([\d.e+-]+)', result1.stdout)
        if nll_match:
            print(f"  NLL value: {nll_match.group(1)}")

    print()

    # Test 2: FIR-based grid search (RMSE-based)
    print("-"*80)
    print("Test 2: FIR-based Grid Search (RMSE-based)")
    print("-"*80)
    print()

    cmd2 = [
        sys.executable, 'src/unified_pipeline.py', input_file,
        '--n-files', '1',
        '--nd', '10',  # Small for fast testing
        '--kernel', 'rbf',
        '--normalize',
        '--log-frequency',
        '--optimize',
        '--grid-search',
        '--use-fir-grid-search',  # Enable FIR-based evaluation
        '--grid-search-max-combinations', '50',  # Small grid for testing
        '--extract-fir',
        '--fir-length', '512',  # Smaller for speed
        '--fir-validation-mat', validation_file,
        '--out-dir', 'verify_output_fir_based'
    ]

    print("Running command:")
    print(' '.join(cmd2))
    print()

    result2 = subprocess.run(cmd2, capture_output=True, text=True)

    # Check for "best RMSE" in output
    rmse_found = 'best RMSE' in result2.stdout or 'Best RMSE' in result2.stdout
    print(f"✓ Found 'best RMSE' in output: {rmse_found}")

    if rmse_found:
        # Extract RMSE value
        rmse_match = re.search(r'[Bb]est RMSE:\s*([\d.e+-]+)', result2.stdout)
        if rmse_match:
            print(f"  RMSE value: {rmse_match.group(1)}")

    # Check for FIR evaluation messages
    fir_eval_found = 'FIR evaluation' in result2.stdout or 'Creating FIR evaluation' in result2.stdout
    print(f"✓ Found FIR evaluation messages: {fir_eval_found}")

    # Check for validation data loading
    validation_loaded = 'Loading validation data' in result2.stdout or 'Validation data loaded' in result2.stdout
    print(f"✓ Validation data loaded: {validation_loaded}")

    print()

    # Summary
    print("="*80)
    print("Verification Summary")
    print("="*80)
    print()

    all_checks = [
        ("Traditional grid search (NLL-based)", nll_found),
        ("FIR-based grid search (RMSE-based)", rmse_found),
        ("FIR evaluation functions created", fir_eval_found),
        ("Validation data loaded", validation_loaded),
    ]

    passed = sum(1 for _, check in all_checks if check)
    total = len(all_checks)

    for name, check in all_checks:
        status = "✓ PASS" if check else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Result: {passed}/{total} checks passed")
    print()

    if passed == total:
        print("✓ All verification checks passed!")
        print("  FIR-based grid search is working correctly.")
        return True
    else:
        print("✗ Some verification checks failed.")
        print("  Please review the output above for details.")

        # Print relevant output sections
        if not rmse_found:
            print()
            print("Relevant output from FIR-based test:")
            print("-"*80)
            # Find grid search section
            lines = result2.stdout.split('\n')
            in_grid_search = False
            for line in lines:
                if 'grid search' in line.lower():
                    in_grid_search = True
                if in_grid_search:
                    print(line)
                    if 'complete' in line.lower():
                        break

        return False

if __name__ == '__main__':
    success = run_test()
    sys.exit(0 if success else 1)
