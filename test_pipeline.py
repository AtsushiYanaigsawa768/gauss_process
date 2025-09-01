#!/usr/bin/env python3
"""
Quick test script for the GP-FIR pipeline
"""

from gp_fir_pipeline import run_complete_pipeline
from pathlib import Path

if __name__ == "__main__":
    # Test with example data
    print("Testing GP-FIR Pipeline...")
    print("-" * 60)
    
    # Create test output directory
    test_dir = Path("./test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Run pipeline
    results = run_complete_pipeline(
        input_data_path=None,  # Use example data
        output_dir=test_dir
    )
    
    # Check outputs
    print("\nChecking outputs...")
    expected_files = [
        "gp_predictions.csv",
        "gp_results.png", 
        "fir_results.png",
        "fir_coefficients.npz"
    ]
    
    for filename in expected_files:
        filepath = test_dir / filename
        if filepath.exists():
            print(f"[OK] {filename} created successfully")
        else:
            print(f"[FAIL] {filename} not found")
    
    print("\nTest completed!")