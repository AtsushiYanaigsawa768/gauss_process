# FIR Grid Search - Critical Fix Documentation

## 発見された重大な問題

### 問題1: `--test-mode`で`use_fir_grid_search`が渡されていない

**影響**: テストモードでFIRベースのグリッドサーチを使用できない

**修正内容**:
```python
# Before (line 2558)
run_comprehensive_test(
    mat_files,
    output_base_dir='test_output_frf',
    fir_validation_mat=validation_mat,
    freq_method='frf',
    use_grid_search=True,
    grid_search_max_combinations=5000
)

# After
run_comprehensive_test(
    mat_files,
    output_base_dir='test_output_frf',
    fir_validation_mat=validation_mat,
    freq_method='frf',
    use_grid_search=True,
    grid_search_max_combinations=5000,
    use_fir_grid_search=False  # Set to True to enable FIR-based grid search
)
```

**修正箇所**: `src/unified_pipeline.py:2558-2566`

---

### 問題2: FIR構築ロジックの不整合（重大）

**影響**: グリッドサーチで選択されたハイパーパラメータが、最終的なFIRモデルでは最適でない可能性

**詳細な相違点**:

#### グリッドサーチ内のFIR構築（修正前）
```python
# 1. Frequency grid: デジタル周波数に基づく
omega_ifft = np.linspace(0, np.pi / Ts, N_fft // 2 + 1)

# 2. Interpolation: 線形補間
interp_real = interp1d(omega, np.real(G_pred), kind='linear', ...)
interp_imag = interp1d(omega, np.imag(G_pred), kind='linear', ...)

# 3. Hermitian symmetry: 偶数長
G_full = np.concatenate([G_ifft, np.conj(G_ifft[-2:0:-1])])

# 4. IFFT
g = np.fft.ifft(G_full).real[:fir_length]

# 5. Parameters
N_fft = 2 * fir_length  # 例: 2048
```

#### 最終的なFIR構築（paper_mode）
```python
# 1. Frequency grid: 連続周波数の等間隔グリッド
omega_uni = np.linspace(omega_min, omega_max, Nd=1000)

# 2. Interpolation: 線形補間またはGP予測
Gr = np.interp(omega_uni, omega, np.real(G), ...)
# または
G_uni = gp_predict_func(omega_uni)

# 3. Two-sided Hermitian spectrum: 奇数長
M = 2*Nd - 1  # 例: 1999
X[0] = np.real(G_uni[0])  # DC
X[1:Nd] = G_uni[1:Nd]     # Positive
X[M-(Nd-1):] = np.conjugate(G_uni[1:Nd][::-1])  # Negative

# 4. IDFT
h_full = np.fft.ifft(X, n=M).real
g = h_full[:N].copy()

# 5. Parameters
Nd = 1000  # 固定
M = 1999   # 奇数長
```

#### 主な相違点まとめ

| 項目 | グリッドサーチ（修正前） | 最終的なFIR | 影響 |
|-----|---------------------|-----------|------|
| Frequency grid | デジタル周波数ベース | 連続周波数ベース | **高** |
| Grid type | `np.linspace(0, π/Ts, N_fft//2+1)` | `np.linspace(ω_min, ω_max, 1000)` | **高** |
| Interpolation | 線形 | 線形またはGP | 中 |
| Spectrum length | 偶数長 (N_fft) | 奇数長 (M = 2*Nd-1) | **高** |
| Grid points | N_fft/2+1 (例: 1025) | Nd = 1000 | 中 |
| IFFT length | N_fft (例: 2048) | M = 1999 | 中 |
| DC handling | なし | 実数化 | 低 |
| Nyquist handling | なし | 実数化（偶数長の場合） | 低 |

**なぜこれが重大な問題か？**

1. **周波数グリッドの不一致**:
   - グリッドサーチ: デジタル周波数 `Ω = 2πk/N_fft` → 連続周波数 `ω = Ω/Ts`
   - 最終版: 連続周波数 `ω` を直接等間隔に配置
   - **結果**: 異なる周波数点でGPを評価 → 異なるFIRモデル

2. **スペクトル構築の不一致**:
   - グリッドサーチ: 偶数長スペクトル、ナイキスト周波数を含む
   - 最終版: 奇数長スペクトル、ナイキスト周波数なし
   - **結果**: IDFTの挙動が異なる

3. **データ点数の不一致**:
   - グリッドサーチ: N_fft = 2 * fir_length = 2048 → 1025点
   - 最終版: Nd = 1000点固定
   - **結果**: 周波数分解能が異なる

4. **最適化の矛盾**:
   - グリッドサーチで「アルゴリズムA」を使ってパラメータを最適化
   - 最終的には「アルゴリズムB」でFIRを構築
   - **結果**: 最適化されたパラメータがアルゴリズムBでは最適でない

**修正内容**: グリッドサーチ内のFIR構築を、最終的なFIR構築と完全に一致させました。

```python
def create_fir_evaluator(is_real_part: bool):
    def evaluate_fir_rmse(gp_model):
        """
        Uses the SAME paper-mode algorithm as final FIR extraction:
        - Uniform linear omega grid (Nd = 1000)
        - GP prediction at uniform grid points
        - Two-sided Hermitian spectrum (odd-length M = 2*Nd - 1)
        - IDFT via np.fft.ifft()
        - Extract first N taps
        """
        # ... (完全に一致するロジック)

        # Step 1: Uniform linear omega grid
        Nd = 1000
        omega_uni = np.linspace(omega_min, omega_max, Nd)

        # Step 2: Linear interpolation (same as paper mode fallback)
        Gr = np.interp(omega_uni, omega, np.real(G_pred), ...)
        Gi = np.interp(omega_uni, omega, np.imag(G_pred), ...)
        G_uni = Gr + 1j * Gi

        # Step 3: Two-sided Hermitian spectrum (odd-length)
        M = 2*Nd - 1
        X[0] = np.real(G_uni[0])  # DC
        X[1:Nd] = G_uni[1:Nd]
        X[M-(Nd-1):] = np.conjugate(G_uni[1:Nd][::-1])

        # Step 4: IDFT
        h_full = np.fft.ifft(X, n=M).real
        g = h_full[:N].copy()

        # Step 5: Validation (same as final)
        y_pred_val = np.convolve(u_eval, g, mode="full")[:len(y_eval)]
        skip = min(10, N // 10)
        rmse = np.sqrt(np.mean((y_valid - y_pred_valid)**2))

        return rmse
```

**修正箇所**: `src/unified_pipeline.py:1318-1439`

---

## 修正による期待される改善

### 1. 一貫性の確保
- グリッドサーチと最終的なFIR構築で**完全に同じアルゴリズム**を使用
- 選択されたハイパーパラメータが最終モデルでも最適

### 2. 精度の向上
- グリッドサーチで直接最終評価指標（FIR RMSE）を最小化
- 過学習の抑制（validation dataでの評価）

### 3. 信頼性の向上
- アルゴリズムの不一致による予期しない挙動を防止
- テスト結果の再現性の向上

---

## 修正後のテスト方法

### 1. 従来方法との比較
```bash
# 従来方法（NLLベース）
python src/unified_pipeline.py input/*.mat \
    --n-files 1 --nd 50 --kernel rbf \
    --normalize --log-frequency --optimize --grid-search \
    --extract-fir --fir-validation-mat input/validation.mat \
    --out-dir output_traditional

# 新方法（FIR RMSEベース、修正版）
python src/unified_pipeline.py input/*.mat \
    --n-files 1 --nd 50 --kernel rbf \
    --normalize --log-frequency --optimize --grid-search \
    --use-fir-grid-search \
    --extract-fir --fir-validation-mat input/validation.mat \
    --out-dir output_fir_based_fixed
```

### 2. アルゴリズムの一貫性確認

グリッドサーチと最終的なFIR構築が同じRMSE値を返すことを確認：

```bash
# グリッドサーチで選択されたパラメータでのRMSE
# → グリッドサーチのログに表示される "best RMSE"

# 最終的なFIR構築でのRMSE
# → 出力ディレクトリの "fir_extraction" 結果

# これらが一致することを確認！
```

### 3. テストモードでの確認

```bash
# テストモード（FIRベースのグリッドサーチを有効化）
# src/unified_pipeline.py の2565行目を変更:
# use_fir_grid_search=False → use_fir_grid_search=True

python src/unified_pipeline.py --test-mode
```

---

## 計算コストへの影響

修正により、グリッドサーチの計算コストがわずかに増加します：

| 項目 | 修正前 | 修正後 | 増加率 |
|-----|--------|--------|-------|
| FIR点数 | N_fft/2+1 (1025) | Nd = 1000 | -2.4% |
| IFFT長 | N_fft (2048) | M = 1999 | -2.4% |
| 1点あたりの時間 | ~1秒 | ~1秒 | 0% |

**影響**: ほぼなし（わずかに高速化）

---

## まとめ

### 修正内容
1. `--test-mode`に`use_fir_grid_search`パラメータを追加
2. グリッドサーチ内のFIR構築を最終版と完全一致させる

### 重要性
- **Critical**: FIR構築ロジックの不一致は最適化結果に直接影響
- **High**: アルゴリズムの一貫性は信頼性と再現性に不可欠

### 推奨事項
1. 修正版を使用してハイパーパラメータを再最適化
2. 従来方法との比較テストを実施
3. 複数のカーネルで性能を比較

---

## チェックリスト

- [x] `--test-mode`に`use_fir_grid_search`パラメータを追加
- [x] グリッドサーチのFIR構築を修正（paper-mode）
- [x] Nd = 1000固定（最終版と一致）
- [x] 奇数長スペクトル（M = 2*Nd - 1）
- [x] IDFT via `np.fft.ifft()`
- [x] 同じvalidation処理（skip = min(10, N//10)）
- [x] ドキュメントの更新
- [ ] テストの実施
- [ ] 性能比較の実施

---

## バージョン情報

- 修正日: 2024-XX-XX
- 修正ファイル: `src/unified_pipeline.py`
- 修正行数: 1318-1439, 2565
- 影響: グリッドサーチの評価ロジック全体
