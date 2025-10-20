# FIR Grid Search - Real/Imaginary Coupling Issue

## 重大な問題: Real/Imaginary Partの独立最適化

### 問題の概要

グリッドサーチでFIRモデルを構築する際、Real partとImaginary partを**独立に**最適化していますが、FIR構築には**両方が必要**です。

### 現在の実装（問題あり）

#### Real Part最適化時（line 1359-1364）
```python
if is_real_part:
    y_pred_real = gp_model.predict(X_gp_normalized)  # 最適化中のGP
    if config.normalize:
        y_pred_real = y_real_scaler.inverse_transform(y_pred_real.reshape(-1, 1)).ravel()
    # Use current imaginary prediction (or zero for initial evaluation)
    y_pred_imag = np.zeros_like(y_pred_real)  # ← ゼロ！！！
```

**問題**: Imaginary partがゼロ → FIRモデルが不完全

#### Imaginary Part最適化時（line 1365-1370）
```python
else:
    y_pred_imag = gp_model.predict(X_gp_normalized)  # 最適化中のGP
    if config.normalize:
        y_pred_imag = y_imag_scaler.inverse_transform(y_pred_imag.reshape(-1, 1)).ravel()
    # Use current real prediction (or from G_complex)
    y_pred_real = np.real(G_complex)  # ← 元のデータ（最適化されていない）
```

**問題**: Real partが最適化されたGPではなく元のデータ

### 最終的なFIR構築（正しい実装）

```python
# Real partの最適化されたGP
y_real_pred = gp_real.predict(X_gp_normalized)

# Imaginary partの最適化されたGP
y_imag_pred = gp_imag.predict(X_gp_normalized)

# 両方を使ってFIRを構築
G_pred = y_real_pred + 1j * y_imag_pred
```

### なぜこれが問題か？

1. **不完全な評価**:
   - Real part最適化: Imaginaryがゼロ → FIRモデルが位相情報を持たない
   - Imaginary part最適化: Realが最適化されていない → 不正確なFIRモデル

2. **最適化の矛盾**:
   - グリッドサーチ: 不完全なFIRモデルで評価
   - 最終版: 完全なFIRモデル（Real + Imaginary両方最適化済み）
   - **結果**: 選択されたパラメータが最終モデルで最適でない

3. **実測データとの比較**:
   - グリッドサーチで選択されたReal partのパラメータは、Imaginary=0の条件下で最適
   - しかし最終的には、Imaginary≠0の条件で使用される
   - 条件が異なるため、最適性が保証されない

### 影響度の分析

| シナリオ | Real Part | Imaginary Part | FIR品質 | 影響度 |
|---------|-----------|---------------|---------|-------|
| Real最適化中 | 最適化中 | **ゼロ** | 不完全 | **Critical** |
| Imag最適化中 | 元データ | 最適化中 | 不正確 | **High** |
| 最終モデル | 最適化済み | 最適化済み | 完全 | - |

### 解決策

#### オプション1: 暫定的な解決策（計算コスト低）

Real part最適化時に**元のimaginary data**を使用、Imaginary part最適化時に**最適化済みのreal part**を使用。

```python
# Real part最適化時
if is_real_part:
    y_pred_real = gp_model.predict(X_gp_normalized)
    if config.normalize:
        y_pred_real = y_real_scaler.inverse_transform(y_pred_real.reshape(-1, 1)).ravel()
    # Use ORIGINAL imaginary data (not zero!)
    y_pred_imag = np.imag(G_complex)  # ← 修正

# Imaginary part最適化時
else:
    y_pred_imag = gp_model.predict(X_gp_normalized)
    if config.normalize:
        y_pred_imag = y_imag_scaler.inverse_transform(y_pred_imag.reshape(-1, 1)).ravel()
    # Use OPTIMIZED real part (if available)
    # Need to store optimized real GP somewhere...
    y_pred_real = np.real(G_complex)  # ← まだ問題あり
```

**課題**: Imaginary最適化時、Real partの最適化済みGPにアクセスできない

#### オプション2: 逐次最適化（計算コスト中）

1. Real partを最適化（Imaginaryは元データ）
2. Imaginary partを最適化（Realは最適化済み）
3. （オプション）Real partを再最適化（Imaginaryは最適化済み）

```python
# Phase 1: Optimize Real part (with original Imag)
gp_real.fit(..., use_fir_rmse=True, fir_evaluation_func=fir_eval_real_v1)

# Phase 2: Optimize Imaginary part (with optimized Real)
# Need to pass optimized gp_real to fir_eval_imag
gp_imag.fit(..., use_fir_rmse=True, fir_evaluation_func=fir_eval_imag_v2)

# Phase 3 (optional): Re-optimize Real part (with optimized Imag)
gp_real.fit(..., use_fir_rmse=True, fir_evaluation_func=fir_eval_real_v3)
```

**利点**: 最終的なFIR構築に近い
**欠点**: 実装が複雑、計算コスト増加

#### オプション3: 同時最適化（計算コスト高）

Real/Imaginaryのパラメータ空間を同時に探索。

```python
# Grid: (real_params, real_noise, imag_params, imag_noise)
# For each combination:
#   - Fit Real GP with real_params, real_noise
#   - Fit Imaginary GP with imag_params, imag_noise
#   - Build FIR with both GPs
#   - Evaluate FIR RMSE
```

**利点**: 真の最適化
**欠点**: グリッド点数が爆発的に増加（例: 500^2 = 250,000組み合わせ）

#### オプション4: 複素数GP（計算コスト不明）

Real/Imaginaryを結合した複素数値GPを直接最適化。

**利点**: 理論的に最も正確
**欠点**: 実装が大幅に変わる、計算コスト不明

### 推奨される実装

**短期**: オプション1（暫定的な解決策）
- Real最適化時: Imaginaryは元データ（ゼロではなく）
- Imaginary最適化時: Realは元データ（最適化済みが理想だが、アクセス困難）

**中期**: オプション2（逐次最適化）
- 2回のパス: Real → Imaginary
- または3回のパス: Real → Imaginary → Real

**長期**: オプション3または4（完全な最適化）
- 計算コストと精度のトレードオフを評価

### 現時点での対応

まず**オプション1**を実装して、ゼロを使う問題を修正します。

```python
# Real part最適化時
if is_real_part:
    y_pred_real = gp_model.predict(X_gp_normalized)
    if config.normalize:
        y_pred_real = y_real_scaler.inverse_transform(y_pred_real.reshape(-1, 1)).ravel()
    # Use ORIGINAL imaginary data (better than zero!)
    y_pred_imag = np.imag(G_complex)  # ← 修正！
```

これにより、少なくともFIRモデルが完全な情報（Real + Imaginary）を持つようになります。

### 検証方法

修正後、以下を確認：

1. **グリッドサーチのログ**: "best RMSE"が出力されているか
2. **RMSE値**: ゼロや異常に大きい値でないか
3. **最終FIRとの比較**: グリッドサーチでのRMSEと最終FIRのRMSEが大きく乖離していないか
4. **複数カーネルでの比較**: 異なるカーネルで期待通りの性能差があるか

### まとめ

- **現状**: Real/Imaginary partが独立に最適化されており、FIR評価が不完全
- **影響**: 選択されたパラメータが最終モデルで最適でない可能性
- **対策**: 少なくともゼロではなく元データを使用（オプション1）
- **今後**: 逐次最適化または同時最適化を検討（オプション2/3）
