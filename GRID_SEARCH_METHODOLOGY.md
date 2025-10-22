# Grid Search Hyperparameter Optimization for Gaussian Process Regression in Frequency Response Identification

## Abstract

This document describes the grid search methodology employed for hyperparameter optimization in Gaussian Process (GP) regression applied to frequency response function (FRF) identification of a flexible link mechanism. The proposed method utilizes a validation-based approach with high-resolution frequency domain data to select optimal kernel parameters, while keeping noise variance fixed to prevent overfitting.

## 1. Introduction

Gaussian Process regression has been widely applied to system identification tasks due to its ability to provide probabilistic predictions and uncertainty quantification. However, the performance of GP models is highly dependent on the choice of hyperparameters, particularly kernel parameters. This work presents a systematic grid search approach for hyperparameter selection in the context of FRF identification.

## 2. Data Acquisition and Preprocessing

### 2.1 Training Data Generation

**Source**: MAT files containing time-series measurements of input-output data from a flexible link mechanism.

**Location**: `input/*.mat` directory

**Conversion Process**:
1. Time-series data [t, u, y] is loaded from MAT files
2. Frequency response is computed using the Frequency Response Function (FRF) method:
   - **Frequency grid**: Log-scale distribution from 10^(-1.0) to 10^(2.3) Hz
   - **Number of points (N_d)**: Configurable (default: 100)
   - **Method**: Synchronous demodulation with trapezoidal integration
   - **Processing**:
     - Mean subtraction: `subtract_mean = True`
     - Transient removal: `drop_seconds = 0.0`
3. Complex transfer function G(jω) is computed via cross-power averaging:
   ```
   G(jω) = ⟨Y(jω) U*(jω)⟩ / ⟨U(jω) U*(jω)⟩
   ```
4. Decomposition into real and imaginary components:
   ```
   G(jω) = Re{G(jω)} + j·Im{G(jω)}
   ```

**Normalization**: Both input (log₁₀(ω)) and output (Re{G}, Im{G}) are normalized using `StandardScaler`:
```
X_normalized = (X - μ_X) / σ_X
y_normalized = (y - μ_y) / σ_y
```

### 2.2 Validation Data Generation

**Source**: Test MAT file specified by `--fir-validation-mat` argument

**Location**: User-specified (e.g., `test.mat`)

**Conversion Process**:
1. **Fixed resolution**: 200 frequency points (independent of training N_d)
2. **Same frequency grid**: Log-scale from 10^(-1.0) to 10^(2.3) Hz
3. **Same processing pipeline**: Identical to training data generation
4. **Purpose**: Evaluate model generalization to unseen measurements from the same system

**Important Note**: The validation dataset is derived from a different time-series measurement than the training data, but uses the same frequency grid to ensure fair comparison.

## 3. Evaluation Metric

### 3.1 Root Mean Squared Error (RMSE)

The grid search optimization uses **RMSE in the original scale** as the evaluation metric:

```
RMSE = √(1/N_val · Σᵢ₌₁ᴺᵛᵃˡ (y_val,i - ŷ_val,i)²)
```

where:
- `N_val = 200`: Number of validation points
- `y_val,i`: True value at validation point i (original scale)
- `ŷ_val,i`: GP prediction at validation point i (original scale)

### 3.2 Denormalization Procedure

To ensure fair evaluation in the original scale:

1. GP predictions `ŷ_normalized` are computed in normalized space
2. Inverse transformation is applied:
   ```
   ŷ_original = ŷ_normalized · σ_y + μ_y
   y_val,original = y_val,normalized · σ_y + μ_y
   ```
3. RMSE is computed using denormalized values

### 3.3 Independent Optimization

**Real and imaginary parts are optimized independently**:
- Real part GP: Trained on Re{G(jω)}, evaluated on Re{G_val(jω)}
- Imaginary part GP: Trained on Im{G(jω)}, evaluated on Im{G_val(jω)}

This approach allows different kernel parameters for each component, accounting for potentially different smoothness characteristics.

## 4. Grid Search Parameters

### 4.1 Kernel Types Supported

The framework supports multiple kernel functions:

1. **RBF (Radial Basis Function)**: `k(x, x') = σ² exp(-||x - x'||²/(2ℓ²))`
2. **Matérn**: `k(x, x') = σ² · M_ν(√(2ν)||x - x'||/ℓ)`
3. **Rational Quadratic**: `k(x, x') = σ² (1 + ||x - x'||²/(2αℓ²))^(-α)`
4. **Exponential, Stable Spline, and others**: See code for details

### 4.2 Hyperparameter Ranges and Grid Resolution

#### 4.2.1 RBF Kernel

| Parameter | Symbol | Range | Scale | Grid Points | Total Combinations |
|-----------|--------|-------|-------|-------------|-------------------|
| Length scale | ℓ | [10⁻³, 10³] | Logarithmic | 20 | 400 |
| Variance | σ² | [10⁻³, 10³] | Logarithmic | 20 | (20 × 20) |
| **Noise variance** | **σ_n²** | **1×10⁻⁶ (fixed)** | **—** | **—** | **—** |

**Grid generation**:
```python
length_scale_grid = np.logspace(-3, 3, 20)  # [0.001, 0.00215, ..., 1000]
variance_grid = np.logspace(-3, 3, 20)      # [0.001, 0.00215, ..., 1000]
```

**Total combinations**: 20 × 20 = **400 combinations**

#### 4.2.2 Matérn Kernel

| Parameter | Symbol | Range | Scale | Grid Points | Total Combinations |
|-----------|--------|-------|-------|-------------|-------------------|
| Length scale | ℓ | [10⁻³, 10³] | Logarithmic | 20 | 400 |
| Variance | σ² | [10⁻³, 10³] | Logarithmic | 20 | (20 × 20) |
| **Smoothness** | **ν** | **1.5 (fixed)** | **—** | **—** | **—** |
| **Noise variance** | **σ_n²** | **1×10⁻⁶ (fixed)** | **—** | **—** | **—** |

**Total combinations**: 20 × 20 = **400 combinations**

#### 4.2.3 Rational Quadratic Kernel

| Parameter | Symbol | Range | Scale | Grid Points | Total Combinations |
|-----------|--------|-------|-------|-------------|-------------------|
| Length scale | ℓ | [10⁻³, 10³] | Logarithmic | 20 | 8,000 |
| Variance | σ² | [10⁻³, 10³] | Logarithmic | 20 | (20 × 20 × 20) |
| Alpha | α | [10⁻³, 10²] | Logarithmic | 20 | |
| **Noise variance** | **σ_n²** | **1×10⁻⁶ (fixed)** | **—** | **—** | **—** |

**Total combinations**: 20 × 20 × 20 = **8,000 combinations**

Note: Exceeds maximum limit; random sampling applied (see Section 4.3).

### 4.3 Random Sampling for Large Search Spaces

**Maximum combinations limit**: 5,000

When the total number of parameter combinations exceeds 5,000:
1. All combinations are generated
2. **Random sampling** with seed 42 (for reproducibility) selects 5,000 combinations
3. Only the sampled combinations are evaluated

**Affected kernels**: Rational Quadratic (8,000 → 5,000)

### 4.4 Fixed Hyperparameters

**Noise variance (σ_n²)**: Fixed at **1×10⁻⁶**

**Rationale**:
1. **Computational efficiency**: Reduces search space by factor of 12 (from 4,800 to 400 for RBF)
2. **Overfitting prevention**: Avoids fitting to noise characteristics in validation data
3. **Stability**: Prevents numerical instability from extremely small/large noise values
4. **Focus on signal**: Optimizes kernel parameters for capturing system dynamics, not noise

## 5. Grid Search Algorithm

### 5.1 Pseudocode

```
Algorithm: Grid Search Hyperparameter Optimization

Input:
  - X_train: Training frequencies (N_train × 1)
  - y_train: Training response (N_train,) [Re or Im]
  - X_val: Validation frequencies (200 × 1)
  - y_val: Validation response (200,) [Re or Im]
  - y_scaler: StandardScaler for denormalization
  - kernel_type: Kernel function (RBF, Matérn, etc.)
  - σ_n²: Fixed noise variance

Output:
  - θ*: Optimal kernel parameters
  - RMSE*: Best validation RMSE

1: Generate parameter grids based on kernel_type
2: combinations ← Cartesian product of all grids
3: IF len(combinations) > 5000 THEN
4:     combinations ← random_sample(combinations, 5000, seed=42)
5: END IF

6: θ* ← None, RMSE* ← ∞

7: FOR each θ in combinations DO
8:     Set kernel parameters to θ
9:     Compute K ← k(X_train, X_train) + σ_n²·I
10:    Cholesky decomposition: K = L·Lᵀ
11:    Solve for α: α ← L⁻ᵀ·L⁻¹·y_train
12:
13:    Predict on validation: ŷ_val ← k(X_val, X_train)·α
14:
15:    Denormalize:
16:        ŷ_val_orig ← y_scaler.inverse_transform(ŷ_val)
17:        y_val_orig ← y_scaler.inverse_transform(y_val)
18:
19:    RMSE ← √(mean((y_val_orig - ŷ_val_orig)²))
20:
21:    IF RMSE < RMSE* THEN
22:        θ* ← θ, RMSE* ← RMSE
23:    END IF
24: END FOR

25: RETURN θ*, RMSE*
```

### 5.2 Computational Complexity

**Per iteration**:
- Cholesky decomposition: O(N_train³)
- Forward/backward substitution: O(N_train²)
- Prediction: O(N_val · N_train²)
- RMSE computation: O(N_val)

**Total**: O(n_combinations · N_val · N_train²)

**Typical values**:
- N_train = 100
- N_val = 200
- n_combinations = 400 (RBF)

**Estimated time**: ~1-5 minutes per component (Real/Imag) on standard CPU

## 6. Diagnostic Outputs

### 6.1 Progress Monitoring

During grid search, the following diagnostics are reported:

1. **Initial performance** (before optimization):
   ```
   Initial (pre-grid-search) RMSE: X.XXe-XX
   Initial params: [ℓ_init, σ²_init], noise: 1.000e-06 (fixed)
   ```

2. **Search progress** (every 10%):
   ```
   Progress: 40/400 (10.0%), best RMSE: X.XXe-XX
   Progress: 80/400 (20.0%), best RMSE: X.XXe-XX
   ...
   ```

3. **Final results**:
   ```
   Grid search complete. Best RMSE: X.XXe-XX
   Optimal params: [ℓ*, σ²*], noise: 1.000e-06 (fixed)
   ```

### 6.2 Generalization Check

**Training vs. Validation RMSE**:
```
Training RMSE: X.XXe-XX, Validation RMSE: Y.YYe-YY
```

**Overfitting warning**: If validation RMSE > 1.5 × training RMSE:
```
WARNING: Validation RMSE is Z.ZZx higher than training RMSE
         This may indicate poor generalization or data mismatch
```

## 7. Data Leakage Considerations

### 7.1 Current Implementation

**Issue**: The same file (`--fir-validation-mat`) is used for:
1. Grid search validation (hyperparameter selection)
2. Final FIR model evaluation (performance reporting)

**Consequence**: Hyperparameters may be optimized to this specific file, potentially reducing generalization to truly unseen data.

### 7.2 Recommendations

For rigorous evaluation, three datasets are recommended:

1. **Training set**: For GP model fitting
2. **Validation set**: For hyperparameter selection (grid search)
3. **Test set**: For final performance evaluation (FIR RMSE)

**Future work**: Implement separate `--validation-mat` and `--fir-validation-mat` arguments.

## 8. Comparison with Alternative Methods

### 8.1 Gradient-Based Optimization

**Previous method**: Maximum likelihood estimation via L-BFGS-B
- Optimizes both kernel parameters AND noise variance
- Uses negative log marginal likelihood (NLL) as objective
- Requires multiple random restarts (default: 3)

**Grid search advantages**:
1. **Global exploration**: Systematic coverage of parameter space
2. **Validation-based**: Directly optimizes for prediction accuracy
3. **Reproducibility**: Fixed grid eliminates sensitivity to initialization
4. **Interpretability**: Clear visualization of objective landscape

**Grid search disadvantages**:
1. **Computational cost**: Higher for large grids (mitigated by random sampling)
2. **Curse of dimensionality**: Limited to 2-3 kernel parameters

### 8.2 Performance Comparison

| Method | Metric | Parameters Optimized | Typical Time |
|--------|--------|---------------------|--------------|
| Gradient (NLL) | Log marginal likelihood | Kernel + Noise | 10-30 seconds |
| **Grid search (RMSE)** | **Validation RMSE** | **Kernel only** | **1-5 minutes** |

## 9. Implementation Details

### 9.1 Command-Line Usage

```bash
python src/unified_pipeline.py input/*.mat \
  --n-files 1 \
  --nd 100 \
  --kernel rbf \
  --normalize \
  --log-frequency \
  --grid-search \
  --grid-search-max-combinations 5000 \
  --fir-validation-mat test.mat \
  --extract-fir \
  --fir-length 1024 \
  --out-dir output_grid_search
```

**Key arguments**:
- `--grid-search`: Enable grid search (default: False)
- `--grid-search-max-combinations`: Maximum combinations before random sampling (default: 5000)
- `--fir-validation-mat`: Validation file for grid search AND FIR evaluation
- `--nd`: Number of training frequency points (default: 100)

### 9.2 Code Location

**File**: `src/unified_pipeline.py`

**Key functions**:
- `GaussianProcessRegressor._grid_search_hyperparameters()` (Line 632-799)
- `generate_validation_data_from_mat()` (Line 1029-1092)
- `run_gp_pipeline()` (Line 1098-1757)

## 10. Results Interpretation

### 10.1 Optimal Hyperparameters

Typical optimal values for flexible link FRF:

**Real part**:
- Length scale (ℓ): 0.5 - 2.0 (log-frequency domain)
- Variance (σ²): 0.1 - 10.0
- RMSE: 1×10⁻³ - 1×10⁻²

**Imaginary part**:
- Length scale (ℓ): 0.3 - 1.5
- Variance (σ²): 0.1 - 5.0
- RMSE: 1×10⁻³ - 1×10⁻²

**Interpretation**:
- **Length scale**: Controls smoothness (larger = smoother predictions)
- **Variance**: Controls amplitude of variations (larger = more flexible fit)
- **Different for Re/Im**: Reflects different frequency characteristics

### 10.2 Validation Curve Analysis

**Underfitting** (high RMSE):
- Length scale too large → over-smoothing
- Variance too small → insufficient flexibility

**Overfitting** (low training RMSE, high validation RMSE):
- Length scale too small → fitting noise
- Validation RMSE > 1.5 × Training RMSE

**Optimal** (balanced):
- Validation RMSE ≈ 1.0-1.3 × Training RMSE
- Smooth predictions that generalize well

## 11. Limitations and Future Work

### 11.1 Current Limitations

1. **Noise variance fixed**: May not be optimal for all datasets
2. **Single validation set**: No cross-validation
3. **Data leakage**: Validation = Test file
4. **Computational cost**: O(n³) scaling with training data size

### 11.2 Future Improvements

1. **Hierarchical optimization**: Coarse grid → fine grid refinement
2. **Bayesian optimization**: Gaussian Process for hyperparameter optimization
3. **Cross-validation**: K-fold validation for robust selection
4. **Adaptive grids**: Focus sampling on promising regions
5. **Parallel evaluation**: GPU/multi-core acceleration
6. **Time-domain validation**: Optimize for FIR RMSE directly

## 12. Conclusion

This grid search methodology provides a systematic, reproducible approach to hyperparameter optimization for Gaussian Process regression in FRF identification. By using high-resolution validation data (200 points) and RMSE in the original scale, the method directly optimizes for prediction accuracy while maintaining computational tractability through fixed noise variance and random sampling for large search spaces.

The independent optimization of real and imaginary components, combined with diagnostic outputs for overfitting detection, enables robust model selection for downstream FIR model extraction and system identification tasks.

## References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
2. Ljung, L. (1999). *System Identification: Theory for the User*. Prentice Hall.
3. Pintelon, R., & Schoukens, J. (2012). *System Identification: A Frequency Domain Approach*. IEEE Press.

---

**Document version**: 1.0
**Last updated**: 2025-01-17
**Author**: Auto-generated from implementation in `unified_pipeline.py`
