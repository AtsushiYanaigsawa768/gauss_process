# FIR-based Grid Search for GP Hyperparameter Optimization

## 概要

このドキュメントは、GPのハイパーパラメータグリッドサーチにおいて、評価指標をFIRモデルのRMSEに変更する機能について説明します。

## 従来の方法 vs 新しい方法

### 従来の方法
- **評価指標**: 負の対数周辺尤度 (Negative Log Marginal Likelihood)
- **評価内容**: GPモデルがトレーニングデータにどれだけ適合しているか
- **利点**: 計算が高速
- **欠点**: 最終的な評価指標（FIR RMSE）と直接関係しない

### 新しい方法
- **評価指標**: FIRモデルのRMSE
- **評価内容**: GPモデルから構築したFIRモデルが実際の時系列データにどれだけ適合しているか
- **利点**: 最終評価指標に直接最適化される
- **欠点**: 計算コストが高い（各グリッド点でFIR構築が必要）

## 実装の詳細

### 主な変更点

1. **`GaussianProcessRegressor.fit()`メソッド**
   - `use_fir_rmse`: FIRベースの評価を使用するかどうか
   - `fir_evaluation_func`: FIR評価関数

2. **`_grid_search_hyperparameters()`メソッド**
   - FIRモデル構築と評価のロジックを追加
   - 各グリッド候補に対してFIRモデルを構築し、RMSEを計算

3. **`run_gp_pipeline()`関数**
   - validation dataの読み込み
   - FIR評価関数の作成
   - real/imaginary partそれぞれにFIR評価を適用

4. **コマンドライン引数**
   - `--use-fir-grid-search`: FIRベースのグリッドサーチを有効化

## 使用方法

### 基本的な使用例

```bash
python src/unified_pipeline.py input/*.mat \
    --n-files 1 --nd 50 \
    --kernel rbf --normalize --log-frequency \
    --optimize --grid-search \
    --use-fir-grid-search \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat input/validation.mat \
    --out-dir output
```

### パラメータの説明

| パラメータ | 説明 | 必須 |
|-----------|------|------|
| `--use-fir-grid-search` | FIRベースのグリッドサーチを有効化 | ✓ |
| `--grid-search` | グリッドサーチを使用（勾配ベース最適化の代わり） | ✓ |
| `--fir-validation-mat` | FIR評価用のMATファイル（[t, y, u]形式） | ✓ |
| `--grid-search-max-combinations` | グリッドの最大組み合わせ数（デフォルト: 5000） | × |
| `--optimize` | ハイパーパラメータ最適化を有効化 | ✓ |
| `--extract-fir` | FIRモデル抽出を有効化 | ✓ |

### グリッド点数の推奨設定

FIRベースのグリッドサーチは計算コストが高いため、グリッド点数を調整することを推奨します：

```bash
# 粗いグリッド（高速だが精度低い）
--grid-search-max-combinations 100

# 中程度のグリッド（バランス型）
--grid-search-max-combinations 500

# 細かいグリッド（低速だが精度高い）
--grid-search-max-combinations 2000
```

## 計算コストの比較

| 評価方法 | グリッド点数 | 1点あたりの時間 | 総時間（推定） |
|---------|------------|--------------|--------------|
| 従来方法（NLL） | 5000 | ~0.01秒 | ~50秒 |
| FIRベース | 100 | ~1秒 | ~100秒 |
| FIRベース | 500 | ~1秒 | ~500秒 |
| FIRベース | 2000 | ~1秒 | ~2000秒 |

※実際の時間は、データサイズ、FIR長、システム性能により異なります。

## アルゴリズムの流れ

### FIRベースのグリッドサーチ

```
For each grid combination (kernel_params, noise_var):
    1. Set GP hyperparameters
    2. Fit GP model to training data
    3. Create GP prediction function
    4. Build FIR model from GP predictions
       - Interpolate frequency response to uniform grid
       - Apply IFFT to get impulse response
       - Extract FIR coefficients
    5. Validate FIR model with time-series data
       - Convolve input signal with FIR coefficients
       - Calculate RMSE between predicted and actual output
    6. Store RMSE as evaluation score

Select parameters with minimum RMSE
```

## 実装例：FIR評価関数

```python
def create_fir_evaluator(is_real_part: bool):
    """Create FIR evaluation function for real or imaginary part."""
    def evaluate_fir_rmse(gp_model):
        # 1. Get predictions from current GP model
        y_pred = gp_model.predict(X_normalized)

        # 2. Combine real and imaginary predictions
        G_pred = y_real_pred + 1j * y_imag_pred

        # 3. Build FIR model
        # - Interpolate to uniform frequency grid
        # - Apply IFFT to get impulse response
        g = build_fir_from_frequency_response(omega, G_pred)

        # 4. Validate with time-series data
        y_pred_val = np.convolve(u_val, g, mode="full")[:len(y_val)]

        # 5. Calculate RMSE
        err = y_val - y_pred_val
        rmse = np.sqrt(np.mean(err**2))

        return rmse

    return evaluate_fir_rmse
```

## 注意事項

1. **Validation Dataの必要性**
   - FIRベースのグリッドサーチには、validation用のMATファイルが必須です
   - MATファイルは`[t, y, u]`形式（時間、出力、入力）である必要があります

2. **計算コストの増加**
   - 従来方法と比較して、50～100倍の計算時間がかかります
   - グリッド点数を減らす、または並列処理を検討してください

3. **メモリ使用量**
   - 各グリッド点でFIRモデルを構築するため、メモリ使用量が増加します
   - FIR長を調整することで、メモリ使用量を制御できます

4. **Real/Imaginary Partの独立性**
   - 現在の実装では、realとimaginary partを独立に最適化します
   - より精密な最適化には、両者を同時に最適化する必要があります

## トラブルシューティング

### エラー: "Validation MAT file not found"
**解決策**: `--fir-validation-mat`で指定したファイルが存在することを確認してください。

### エラー: "FIR evaluation failed"
**原因**: FIR構築時のエラー（周波数範囲、データ形式など）
**解決策**:
- Validation dataの形式を確認（[t, y, u]）
- 周波数範囲が適切か確認
- FIR長を調整（`--fir-length`）

### 計算が遅すぎる
**解決策**:
1. グリッド点数を減らす（`--grid-search-max-combinations`）
2. FIR長を短くする（`--fir-length`）
3. データ点数を減らす（`--nd`）

## パフォーマンス最適化のヒント

1. **グリッドの事前絞り込み**
   - まず従来方法で粗いグリッドサーチを実行
   - 良好なパラメータ範囲を特定
   - その範囲でFIRベースのグリッドサーチを実行

2. **並列処理**
   - グリッド点の評価は独立しているため、並列処理が可能
   - 将来のバージョンで実装予定

3. **キャッシング**
   - 同じパラメータでの評価結果をキャッシュ
   - 将来のバージョンで実装予定

## 参考文献

- Gaussian Process Regression
- System Identification
- FIR Filter Design
- Hyperparameter Optimization

## 更新履歴

- 2024-XX-XX: 初版作成
- FIRベースのグリッドサーチ機能を実装
