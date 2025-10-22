# FIR-based Grid Search Implementation - Change Summary

## 変更概要

GPのハイパーパラメータグリッドサーチにおいて、評価指標を**負の対数周辺尤度（NLL）** から **FIRモデルのRMSE** に変更する機能を実装しました。

## 変更ファイル

### `src/unified_pipeline.py`

#### 1. `GaussianProcessRegressor.fit()` メソッド（572-596行目）
```python
def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True, n_restarts: int = 3,
        use_grid_search: bool = False, param_grids: Dict = None, max_grid_combinations: int = 5000,
        use_fir_rmse: bool = False, fir_evaluation_func = None):
```

**追加パラメータ:**
- `use_fir_rmse`: FIRベースの評価を使用するかどうか
- `fir_evaluation_func`: FIR評価関数（GPモデルを受け取りRMSEを返す）

#### 2. `_grid_search_hyperparameters()` メソッド（622-752行目）
```python
def _grid_search_hyperparameters(self, param_grids: Dict = None, max_combinations: int = 5000,
                                 use_fir_rmse: bool = False, fir_evaluation_func = None):
```

**追加機能:**
- `fir_rmse_evaluation()` 関数を追加
  - 各グリッド候補に対してFIRモデルを構築
  - validation dataでRMSEを計算
  - エラーハンドリング

**評価ロジックの変更:**
```python
evaluation_func = fir_rmse_evaluation if use_fir_rmse else neg_log_marginal_likelihood
score_name = "RMSE" if use_fir_rmse else "NLL"

for i, combination in enumerate(all_combinations):
    score = evaluation_func(kernel_params, noise_var)
    if score < best_score:
        best_score = score
        best_params = kernel_params
        best_noise = noise_var
```

#### 3. `run_gp_pipeline()` 関数（1231-1445行目）

**追加機能:**

##### 3.1 Validation Dataの読み込み（1234-1290行目）
```python
use_fir_grid_search = getattr(config, 'use_fir_grid_search', False)
use_grid_search = getattr(config, 'use_grid_search', False)

if use_fir_grid_search and use_grid_search:
    # Load validation MAT file
    # Extract [t, y, u] time-series data
    # Infer sampling time Ts
    validation_data = {'t': T, 'y': y, 'u': u, 'Ts': Ts}
```

##### 3.2 FIR評価関数の作成（1311-1417行目）
```python
def create_fir_evaluator(is_real_part: bool):
    def evaluate_fir_rmse(gp_model):
        # 1. Get predictions from GP model
        # 2. Combine real and imaginary predictions
        # 3. Build FIR model (IFFT-based)
        # 4. Validate with time-series data
        # 5. Calculate RMSE
        return rmse
    return evaluate_fir_rmse

fir_evaluation_func_real = create_fir_evaluator(is_real_part=True)
fir_evaluation_func_imag = create_fir_evaluator(is_real_part=False)
```

##### 3.3 GP Fitの更新（1419-1445行目）
```python
# Real part
gp_real.fit(X_gp_normalized, y_real,
           optimize=config.optimize,
           n_restarts=config.n_restarts,
           use_grid_search=use_grid_search,
           max_grid_combinations=max_grid_combinations,
           use_fir_rmse=use_fir_grid_search,  # NEW
           fir_evaluation_func=fir_evaluation_func_real)  # NEW

# Imaginary part
gp_imag.fit(X_gp_normalized, y_imag,
           optimize=config.optimize,
           n_restarts=config.n_restarts,
           use_grid_search=use_grid_search,
           max_grid_combinations=max_grid_combinations,
           use_fir_rmse=use_fir_grid_search,  # NEW
           fir_evaluation_func=fir_evaluation_func_imag)  # NEW
```

#### 4. `main()` 関数（1794-1890行目）

**追加コマンドライン引数:**
```python
parser.add_argument('--use-fir-grid-search', action='store_true',
                   help='Evaluate grid search candidates using FIR model RMSE (requires --fir-validation-mat and --grid-search)')
```

#### 5. `run_comprehensive_test()` 関数（2076-2204行目）

**追加パラメータ:**
```python
def run_comprehensive_test(mat_files: List[str],
                          output_base_dir: str = 'test_output',
                          fir_validation_mat: Optional[str] = None,
                          nd_values: List[int] = None,
                          freq_method: str = 'frf',
                          use_grid_search: bool = False,
                          grid_search_max_combinations: int = 5000,
                          use_fir_grid_search: bool = False):  # NEW
```

**configオブジェクトに追加:**
```python
config = argparse.Namespace(
    # ... existing parameters ...
    use_fir_grid_search=use_fir_grid_search  # NEW
)
```

## 新しいドキュメント

### 1. `FIR_GRID_SEARCH_README.md`
- 機能の詳細説明
- 使用方法
- パラメータの説明
- パフォーマンス最適化のヒント
- トラブルシューティング

### 2. `example_fir_grid_search.sh` (Linux/Mac)
- 従来方法との比較例
- 複数カーネルでの比較例
- 実行可能なシェルスクリプト

### 3. `example_fir_grid_search.bat` (Windows)
- Windows用バッチファイル
- 同様の比較例

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

### 必須条件

1. `--use-fir-grid-search` を使用する場合:
   - `--grid-search` が必要
   - `--fir-validation-mat` が必要（validation用のMATファイル）
   - `--optimize` が必要

2. Validation MATファイルの形式:
   - `[t, y, u]` 形式（時間、出力、入力）
   - または名前付き変数 `t`, `y`, `u`

## 計算コストの注意

FIRベースのグリッドサーチは、従来方法と比較して **50～100倍の計算時間** がかかります。

**推奨設定:**
- グリッド点数を減らす: `--grid-search-max-combinations 500`（デフォルト: 5000）
- FIR長を短くする: `--fir-length 512`（デフォルト: 1024）
- 周波数点数を減らす: `--nd 30`（デフォルト: 100）

## アルゴリズムの比較

| 側面 | 従来方法（NLL） | 新方法（FIR RMSE） |
|-----|---------------|------------------|
| 評価指標 | 負の対数周辺尤度 | FIRモデルのRMSE |
| 計算速度 | 高速（~0.01秒/点） | 低速（~1秒/点） |
| 最適化目標 | データ適合度 | 最終評価指標 |
| 推奨グリッド点数 | 5000 | 100-500 |

## 期待される改善

1. **最終評価指標への直接最適化**
   - FIRモデルのRMSEが直接最小化される
   - より実用的なハイパーパラメータが選択される

2. **過学習の抑制**
   - validation dataでの評価により、汎化性能が向上

3. **カーネル選択の改善**
   - 最終的なタスクに適したカーネルが選択される

## テスト方法

### 1. 従来方法との比較
```bash
# 従来方法
python src/unified_pipeline.py input/*.mat \
    --n-files 1 --nd 50 --kernel rbf \
    --normalize --log-frequency --optimize --grid-search \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat input/validation.mat \
    --out-dir output_traditional

# 新方法
python src/unified_pipeline.py input/*.mat \
    --n-files 1 --nd 50 --kernel rbf \
    --normalize --log-frequency --optimize --grid-search \
    --use-fir-grid-search \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat input/validation.mat \
    --out-dir output_fir_based

# Compare RMSE values in output directories
```

### 2. 自動テスト
```bash
# Linux/Mac
bash example_fir_grid_search.sh

# Windows
example_fir_grid_search.bat
```

## 今後の改善案

1. **並列処理**
   - グリッド点の評価を並列化
   - マルチプロセス/マルチスレッド対応

2. **キャッシング**
   - 同じパラメータでの評価結果をキャッシュ
   - 計算の重複を避ける

3. **段階的最適化**
   - まず粗いグリッド（NLL-based）
   - 次に細かいグリッド（FIR-based）

4. **Polar modeのサポート**
   - 現在はseparate modeのみ
   - magnitude/phaseでのFIR評価を追加

5. **Real/Imaginaryの同時最適化**
   - 現在は独立に最適化
   - 両者を同時に考慮した最適化

## 連絡先・サポート

質問や問題がある場合は、以下を確認してください：
- `FIR_GRID_SEARCH_README.md`: 詳細なドキュメント
- `example_fir_grid_search.sh/bat`: 使用例
- このファイル: 変更内容のサマリー

## バージョン情報

- 実装日: 2024-XX-XX
- Python要件: 3.7+
- 依存関係:
  - numpy
  - scipy
  - matplotlib
  - pandas
  - sklearn

## ライセンス

このプロジェクトのライセンスに従います。
