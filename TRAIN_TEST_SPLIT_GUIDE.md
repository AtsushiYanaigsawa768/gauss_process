# 訓練データとテストデータの分割ガイド

## 問題

以下のようなエラーが発生する場合：

```
ERROR: Training and test data cannot use the same file!
  Test file: input_test_20250913_010037.mat
  Training files: ['input_test_20250913_010037.mat']
```

これは、訓練データとテストデータに**同じファイル**が指定されているため、適切な評価ができない状態です。

## 解決方法

### 方法1: 複数ファイルを使用（推奨）

訓練データとして複数のファイルを指定し、そのうち1つをテストデータとして使用します。

```bash
# 正しい例
python src/unified_pipeline.py input/*.mat \
    --n-files 5 \
    --nd 100 \
    --kernel rbf \
    --normalize \
    --grid-search \
    --fir-validation-mat input/input_test_20250913_010037.mat \
    --extract-fir \
    --fir-length 1024 \
    --out-dir output_correct
```

**動作:**
- `input/*.mat` で全ファイル（10個）を指定
- `--n-files 5` で最初の5ファイルを選択
- `input_test_20250913_010037.mat` が5ファイルに含まれていれば自動除外
- 残り4ファイルで訓練、1ファイルでテスト

**結果:**
```
======================================================================
DATA SEPARATION: Excluded 1 file(s) from training data
  Test file: input_test_20250913_010037.mat
  Training files remaining: 4
======================================================================
```

### 方法2: 明示的に異なるファイルを指定

訓練用とテスト用で、明示的に異なる日付/時刻のファイルを使用します。

```bash
# 訓練: 2025/09/13 02:00のデータ
# テスト:  2025/09/13 01:00のデータ
python src/unified_pipeline.py input/input_test_20250913_030050.mat \
    --n-files 1 \
    --time-duration 1800 \
    --nd 100 \
    --kernel rbf \
    --normalize \
    --grid-search \
    --fir-validation-mat input/input_test_20250913_010037.mat \
    --extract-fir \
    --fir-length 1024 \
    --out-dir output_different_files
```

## 利用可能なファイル

```
input/input_test_20250913_010037.mat  # 01:00 データ
input/input_test_20250913_030050.mat  # 03:00 データ
input/input_test_20250913_050103.mat  # 05:00 データ
input/input_test_20250913_070119.mat  # 07:00 データ
input/input_test_20250913_090135.mat  # 09:00 データ
input/input_test_20250913_110148.mat  # 11:00 データ
input/input_test_20250913_130201.mat  # 13:00 データ
input/input_test_20250913_150214.mat  # 15:00 データ
input/input_test_20250913_170227.mat  # 17:00 データ
input/input_test_20250913_190241.mat  # 19:00 データ
```

## 推奨される分割戦略

### 戦略A: 時系列分割

```bash
# 訓練: 最初の8ファイル
# テスト:  9番目のファイル
python src/unified_pipeline.py input/*.mat \
    --n-files 8 \
    --fir-validation-mat input/input_test_20250913_170227.mat
```

### 戦略B: ランダム分割（手動）

```bash
# 訓練: 奇数時刻のファイル (01:00, 05:00, 09:00, 13:00, 17:00)
# テスト:  03:00のファイル
python src/unified_pipeline.py \
    input/input_test_20250913_010037.mat \
    input/input_test_20250913_050103.mat \
    input/input_test_20250913_090135.mat \
    input/input_test_20250913_130201.mat \
    input/input_test_20250913_170227.mat \
    --n-files 5 \
    --fir-validation-mat input/input_test_20250913_030050.mat
```

## 間違った例

### ❌ 同じファイルを訓練とテストに使用

```bash
# 間違い: 同じファイルを両方に指定
python src/unified_pipeline.py input/input_test_20250913_010037.mat \
    --n-files 1 \
    --fir-validation-mat input/input_test_20250913_010037.mat
```

**エラー:**
```
ERROR: Training and test data cannot use the same file!
```

## テストスクリプトの修正例

もしテストスクリプト（バッチファイルやPythonスクリプト）で上記のエラーが発生している場合、以下のように修正してください：

### 修正前（間違い）

```python
# test_config.py
configs = [
    {
        'input_files': ['input/input_test_20250913_010037.mat'],
        'validation_file': 'input/input_test_20250913_010037.mat',  # ❌ 同じファイル
        'n_files': 1
    }
]
```

### 修正後（正しい）

```python
# test_config.py
configs = [
    {
        'input_files': ['input/*.mat'],  # ✓ 複数ファイル
        'validation_file': 'input/input_test_20250913_010037.mat',  # ✓ 1つを除外
        'n_files': 5
    }
]
```

または

```python
# test_config.py
configs = [
    {
        'input_files': ['input/input_test_20250913_030050.mat'],  # ✓ 異なるファイル
        'validation_file': 'input/input_test_20250913_010037.mat',  # ✓ 異なるファイル
        'n_files': 1
    }
]
```

## まとめ

1. **訓練データとテストデータは必ず異なるファイルを使用**
2. **複数ファイルがある場合は `input/*.mat` + `--n-files` を使用**
3. **システムは自動的にテストファイルを訓練データから除外**
4. **エラーメッセージの指示に従って修正**

この方法により、適切な訓練/テスト分割が保証され、正しい評価が可能になります。
