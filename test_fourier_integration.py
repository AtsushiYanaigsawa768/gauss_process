#!/usr/bin/env python3
"""
test_fourier_integration.py

Test script to demonstrate the Fourier transform integration in the unified pipeline.
This script shows how to use both FRF and Fourier methods for frequency domain analysis.
"""

import os
import sys
import subprocess


def run_test(method_name: str, freq_method: str):
    """Run a test with specified frequency method."""
    print(f"\n{'='*60}")
    print(f"Testing {method_name} (freq_method={freq_method})")
    print('='*60)

    cmd = [
        sys.executable,
        'src/unified_pipeline.py',
        'input/*.mat',
        '--n-files', '1',
        '--nd', '50',
        '--freq-method', freq_method,
        '--kernel', 'rbf',
        '--normalize',
        '--out-dir', f'output_test_{freq_method}',
        '--time-duration', '60.0'
    ]

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Success!")
        print("\nOutput files created:")
        output_dir = f'output_test_{freq_method}'
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                print(f"  - {file}")
    else:
        print("✗ Failed!")
        print(f"Error: {result.stderr}")

    return result.returncode == 0


def main():
    """Main test function."""
    print("Testing Fourier Transform Integration in Unified Pipeline")
    print("="*60)

    # Test 1: Traditional FRF method
    success1 = run_test("Traditional FRF Method", "frf")

    # Test 2: New Fourier transform method
    success2 = run_test("Fourier Transform Method", "fourier")

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary:")
    print(f"  FRF method: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"  Fourier method: {'✓ PASSED' if success2 else '✗ FAILED'}")
    print('='*60)

    return 0 if (success1 and success2) else 1


if __name__ == "__main__":
    sys.exit(main())