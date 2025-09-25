# Gaussian Process Frequency Response Pipeline

## Pipeline Overview

```
MAT files → frequency_response.py → FRF data → unified_pipeline.py → GP-smoothed FRF → gp_to_fir_direct.py → FIR model
```

## Core Components

### 1. `frequency_response.py`
- Converts time-series .mat files to frequency response functions (FRF)
- Outputs: CSV, MAT files, Bode/Nyquist plots

### 2. `unified_pipeline.py`
Main pipeline with extensible kernel architecture:
- Kernels: RBF, Matern, Rational Quadratic, TC, Exponential
- GP modes: Separate (real/imag) or Polar (mag/phase)
- Features: Hyperparameter optimization, normalization, visualization

### 3. `gp_to_fir_direct.py`
FIR extraction using GP interpolation:
- GP interpolation → Hermitian symmetry → IFFT → 1024 FIR coefficients
- Validates against time-series data (RMSE, R², FIT%)

### 4. `gp_frequency_analysis.py`
Advanced analysis: Spectral mixture kernels, stability analysis, complex GP

### 5. `gp_fir_model.py`
Additional FIR conversion methods and comparisons

## Usage

```bash
# Basic pipeline
python src/unified_pipeline.py input/*.mat --n-files 1 --kernel rbf

# Complete pipeline with FIR extraction
python src/unified_pipeline.py input/*.mat --n-files 1 \
    --kernel rbf --normalize --log-frequency \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat input/input_test_20250912_165937.mat \
    --out-dir output_complete

# Use existing FRF data
python src/unified_pipeline.py --use-existing output/matched_frf.csv --kernel matern --nu 2.5
```

## Extending Kernels

Inherit from `Kernel` base class and implement required methods. Register in `create_kernel()` factory.

## Output Structure

```
output/
├── *_frf.csv, *_frf.mat        # Frequency response data
├── *_bode_*.png, *_nyquist.png # Plots
├── gp_*.png                    # GP visualizations
├── gp_smoothed_frf.csv         # GP-smoothed FRF
├── gp_results.json             # GP parameters/metrics
└── fir_gp/                     # FIR results
    ├── fir_coefficients_gp.npz
    ├── fir_gp_results.json
    └── gp_fir_results.png
```

## FIR Extraction Process

1. **GP Interpolation**: Use fitted GP to interpolate to uniform frequency grid
2. **Hermitian Symmetry**: G(-ω) = conj(G(ω)) for real impulse response
3. **IFFT**: Obtain 1024 FIR coefficients

## Notes

- Input: MAT files with [time, output, input]
- GP interpolation leverages learned kernel structure
- Validation metrics: RMSE, R², FIT%