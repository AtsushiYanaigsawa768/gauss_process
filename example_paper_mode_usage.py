#!/usr/bin/env python3
"""
Example usage of paper-mode FIR implementation.

This demonstrates how to use the paper-based procedure for GP→FIR conversion.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gp_to_fir_direct_fixed import gp_to_fir_direct_pipeline


def example_basic_usage():
    """Basic example: Simple 1st-order system."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # Generate sample data: 1st order lag G(jω) = 1/(1 + jω/ωc)
    omega = np.linspace(1.0, 200.0, 80)  # 80 points, 1-200 rad/s
    wc = 50.0  # cutoff frequency
    G = 1.0 / (1.0 + 1j * omega / wc)

    print(f"Input: 1st order lag system (ωc = {wc} rad/s)")
    print(f"  {len(omega)} frequency points from {omega[0]} to {omega[-1]} rad/s")

    # Run paper-mode pipeline
    results = gp_to_fir_direct_pipeline(
        omega=omega,
        G=G,
        output_dir=Path("example_output_basic"),
        paper_mode=True,      # Use paper-based procedure
        fir_order=100         # Request 100 taps
    )

    print(f"\nResults:")
    print(f"  Method: {results['method']}")
    print(f"  Nd: {results['Nd']} frequency points")
    print(f"  M (IDFT length): {results['M_idft']}")
    print(f"  FIR order: {results['fir_order']} taps")
    print(f"  Frequency range: {results['omega_range']}")

    return results


def example_with_validation():
    """Example with validation data (requires MAT file)."""
    print("\n" + "="*60)
    print("Example 2: With Validation Data")
    print("="*60)

    # Check if example MAT file exists
    mat_file = Path("fir/data/example_data.mat")  # Adjust path as needed

    if not mat_file.exists():
        print(f"MAT file not found: {mat_file}")
        print("Skipping validation example.")
        print("\nTo use validation:")
        print("  1. Prepare MAT file with [t, y, u] data")
        print("  2. Pass mat_file parameter to pipeline")
        return None

    # Load your frequency response data from CSV or compute from GP
    # For this example, we'll use dummy data
    omega = np.logspace(0, 2.5, 60)  # 60 points, 1-316 rad/s
    # Simulate a simple system
    G = 10.0 / (1.0 + 0.1j*omega)

    print(f"Input: {len(omega)} frequency points")
    print(f"Validation MAT file: {mat_file}")

    results = gp_to_fir_direct_pipeline(
        omega=omega,
        G=G,
        output_dir=Path("example_output_validated"),
        paper_mode=True,
        fir_order=200,
        mat_file=mat_file  # Provide validation data
    )

    print(f"\nResults with validation:")
    print(f"  RMSE: {results.get('rmse', 'N/A')}")
    print(f"  FIT%: {results.get('fit_percent', 'N/A')}")
    print(f"  R²: {results.get('r2', 'N/A')}")

    return results


def example_mode_comparison():
    """Compare paper-mode vs legacy mode."""
    print("\n" + "="*60)
    print("Example 3: Mode Comparison")
    print("="*60)

    # Same system for both modes
    omega = np.linspace(5.0, 300.0, 50)
    wn = 80.0  # natural frequency
    zeta = 0.2  # damping
    G = 1.0 / (1.0 - (omega/wn)**2 + 2j*zeta*(omega/wn))

    print(f"Input: 2nd order system (ωn={wn}, ζ={zeta})")

    # Paper mode
    print("\n--- Paper Mode ---")
    results_paper = gp_to_fir_direct_pipeline(
        omega=omega,
        G=G,
        output_dir=Path("example_output_paper"),
        paper_mode=True,
        fir_order=128
    )

    # Legacy mode
    print("\n--- Legacy Mode ---")
    results_legacy = gp_to_fir_direct_pipeline(
        omega=omega,
        G=G,
        output_dir=Path("example_output_legacy"),
        paper_mode=False,
        fir_length=128,
        N_fft=512
    )

    print("\n" + "-"*60)
    print("Comparison:")
    print(f"  Paper:  Nd={results_paper['Nd']}, M={results_paper['M_idft']}, "
          f"N={results_paper['fir_order']}")
    print(f"  Legacy: N_fft={results_legacy['n_fft']}, "
          f"L={results_legacy['fir_length']}")

    return results_paper, results_legacy


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Paper-Mode FIR Implementation Examples")
    print("="*60)

    # Example 1: Basic usage
    results1 = example_basic_usage()

    # Example 2: With validation (if MAT file available)
    results2 = example_with_validation()

    # Example 3: Mode comparison
    results3 = example_mode_comparison()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nCheck the following directories for outputs:")
    print("  - example_output_basic/")
    print("  - example_output_paper/")
    print("  - example_output_legacy/")
    if results2:
        print("  - example_output_validated/")


if __name__ == "__main__":
    main()