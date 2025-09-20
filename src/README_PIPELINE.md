# System Identification Pipeline

This pipeline implements the system identification methodology described in Method.tex, consisting of:

1. **Data Loading**: Load .mat files containing time-series input/output data
2. **Frequency Domain Conversion**: Apply FFT to extract frequency response
3. **GP Regression**: Apply Gaussian Process regression for interpolation
4. **FIR Model Creation**: Use IFFT to generate FIR filter coefficients
5. **Validation**: Compare predicted vs actual output and calculate metrics

## Usage

### Quick Test
```bash
cd src
python test_pipeline.py
```

### Full Pipeline
```bash
cd src
python main.py
```

### Configuration

Edit the `Config` class in `main.py` to adjust:

- `NUM_MAT_FILES`: Number of .mat files to load (None = all)
- `MAX_SAMPLES_PER_FILE`: Limit samples per file for memory
- `NUM_FREQ_POINTS`: Initial FFT frequency points
- `NUM_TRAINING_POINTS`: Points for GP training
- `NUM_INTERPOLATION_POINTS`: Dense interpolation grid
- `GP_KERNEL`: Kernel choice (see below)
- `FIR_LENGTH`: FIR filter length

### Available Kernels

All kernels from Method.tex are implemented:

- **RBF**: `"rbf"` - Squared exponential kernel
- **Matérn**: `"matern12"`, `"matern32"`, `"matern52"`
- **Exponential**: `"exp"` or `"exponential"` - BIBO-stable kernel
- **TC**: `"tc"` or `"tuned_correlated"` - Tuned-Correlated kernel
- **DC**: `"dc"` - Diagonal/Correlated kernel
- **DI**: `"di"` - Diagonal/Independent kernel
- **Stable Spline**: `"ss"` or `"ss1"` - First-order stable spline
- **Second-order**: `"ss2"` - Second-order stable spline
- **High-freq**: `"hf"` or `"high_freq"` - High frequency stable spline
- **Complex SS**: `"stable_spline_complex"` - Stable spline with Wiener basis

### Input Data Format

.mat files must contain an 'output' array with shape (3, N):
- Row 0: Time vector
- Row 1: Input signal
- Row 2: Output signal

### Output Files

Results are saved to the configured output directory:
- `*_nyquist.png`: Nyquist diagram of frequency response
- `*_gp_interpolation.png`: GP interpolation results (magnitude/phase)
- `*_fir_results.png`: FIR model validation plots
- `*_metrics.csv`: Performance metrics (RMSE, R², etc.)
- `*_fir_coefficients.npz`: FIR filter coefficients
- `*_config.json`: Configuration used for the run

### Performance Metrics

- **RMSE**: Root Mean Square Error
- **NRMSE**: Normalized RMSE
- **R²**: Coefficient of determination
- **FIT%**: Percentage fit (system identification metric)

## Module Structure

- `main.py`: Entry point and configuration
- `data_loader.py`: Load .mat files
- `freq_domain.py`: FFT and frequency analysis
- `gp_kernels.py`: All GP kernel implementations
- `gp_regressor.py`: GP regression for complex data
- `fir_model.py`: FIR model creation via IFFT
- `visualization.py`: All plotting functions

## Troubleshooting

1. **Memory issues**: Reduce `MAX_SAMPLES_PER_FILE` or `NUM_MAT_FILES`
2. **Poor FIR performance**: Increase `FIR_LENGTH` or `NUM_INTERPOLATION_POINTS`
3. **GP optimization slow**: Reduce `GP_MAXITER` or use simpler kernel
4. **Unicode errors**: Fixed in latest version (R² → R2)