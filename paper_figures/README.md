# Paper Figures

論文用の高品質なNyquist図を生成するスクリプトとその生成結果を含むフォルダーです。

## 概要

このフォルダーには、システム同定の結果を論文用に可視化するためのスクリプトが含まれています。
2つの異なる周波数応答推定手法を用いて、統一されたスタイルでNyquist図を生成します。

## 生成される図

### 1. 対数周波数スケールのNyquist図 (`nyquist_log_scale.png/eps`)

- **手法**: 同期復調法（Synchronous Demodulation）
- **周波数グリッド**: 対数スケール（MATLAB形式）
- **データソース**: `src/frequency_response.py` の手法を使用
- **特徴**:
  - 低周波から高周波まで対数的に均等にサンプリング
  - 台形積分による時間重み付き復調
  - クロスパワー推定による伝達関数計算

### 2. 線形周波数スケールのNyquist図 (`nyquist_linear_scale.png/eps`)

- **手法**: フーリエ変換（FFT）
- **周波数グリッド**: 線形スケール
- **データソース**: `src/fourier_transform.py` の手法を使用
- **特徴**:
  - FFTによる直接的な周波数領域解析
  - Hannウィンドウによる窓関数処理
  - 線形周波数スケールでの均等サンプリング

## 図のスタイル

両方の図は、以下の統一されたスタイルで描画されます：

- **マーカー**: 星印（*）のみ、線なし
- **周波数方向インジケーター**:
  - 赤丸（●）: 低周波数点
  - 緑四角（■）: 高周波数点
- **軸**: 等アスペクト比（equal aspect ratio）
- **グリッド**: 破線グリッド
- **フォント**: Serif体、出版物品質
- **解像度**: 300 DPI（PNG）

## 使い方

### 基本的な使い方

```bash
# デフォルト設定で実行（inputフォルダーの最初のファイルを使用）
python paper_figures/generate_nyquist_figures.py

# 特定のMATファイルを指定
python paper_figures/generate_nyquist_figures.py input/input_test_20250912_165937.mat
```

### オプション

```bash
python paper_figures/generate_nyquist_figures.py [MAT_FILE] [OPTIONS]
```

#### 主要なオプション

- `--nd N`: 対数グリッドの周波数点数（デフォルト: 100）
- `--f-low F`: 対数グリッドの下限周波数（log10スケール、デフォルト: -1.0）
- `--f-up F`: 対数グリッドの上限周波数（log10スケール、デフォルト: 2.3）
- `--drop-seconds S`: データ開始からの削除秒数（過渡応答除去、デフォルト: 0.0）
- `--output-dir DIR`: 出力ディレクトリ（デフォルト: paper_figures）
- `--no-eps`: EPSファイルを生成しない
- `--y-col N`: yが2次元の場合の列インデックス（デフォルト: 0）

#### 使用例

```bash
# 周波数点数を200に増やす
python paper_figures/generate_nyquist_figures.py --nd 200

# 初期30秒の過渡応答を除去
python paper_figures/generate_nyquist_figures.py --drop-seconds 30

# EPSファイルなしでPNGのみ生成
python paper_figures/generate_nyquist_figures.py --no-eps

# 周波数範囲を変更（0.01 Hz - 500 Hz）
python paper_figures/generate_nyquist_figures.py --f-low -2.0 --f-up 2.7
```

## 入力ファイル形式

スクリプトは以下の形式のMATファイルを受け付けます：

### 形式A: 個別変数
- `t`: 時間ベクトル [N×1]
- `u`: 入力信号 [N×1]
- `y`: 出力信号 [N×1] または [N×M]（M列の場合、`--y-col`で選択）

### 形式B: 結合配列
- `output` または任意の2次元配列: [3×N] または [N×3]
  - 行/列の順序: [time, output, input] または [t, y, u]

## 出力ファイル

実行後、以下のファイルが生成されます：

```
paper_figures/
├── nyquist_log_scale.png    # 対数スケールNyquist図（PNG、300 DPI）
├── nyquist_log_scale.eps    # 対数スケールNyquist図（EPS、ベクター形式）
├── nyquist_linear_scale.png # 線形スケールNyquist図（PNG、300 DPI）
└── nyquist_linear_scale.eps # 線形スケールNyquist図（EPS、ベクター形式）
```

## 技術的詳細

### 対数周波数グリッド（Method 1）

周波数グリッドはMATLAB形式で生成されます：

```
f = 10^[f_low : step : f_up - step]
ω = 2πf
```

ここで、`step = (f_up - f_low) / N_d`

同期復調による伝達関数計算：

```
C_x(ω) = (2/T) ∫ x(t) e^(-jωt) dt
G(ω) = Y(ω) / U(ω)
```

### 線形周波数グリッド（Method 2）

FFTによる周波数応答計算：

```
U(f) = FFT(u(t)) × dt
Y(f) = FFT(y(t)) × dt
G(f) = Y(f) / U(f)
```

ダウンサンプリング：
- FFT結果（〜90万点）を100点にダウンサンプリング
- 対数グリッドと同じ点数で比較可能

## 論文での使用

### LaTeX

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{paper_figures/nyquist_log_scale.eps}
  \caption{Nyquist plot with logarithmic frequency grid}
  \label{fig:nyquist_log}
\end{figure}
```

### Word/PowerPoint

PNG形式（300 DPI）を使用してください。高解像度なので、拡大しても品質が保たれます。

## トラブルシューティング

### EPSファイルの透明度警告

```
The PostScript backend does not support transparency; partially transparent artists will be rendered opaque.
```

この警告は無視できます。EPSファイルは正常に生成されており、透明度は自動的に不透明としてレンダリングされます。

### メモリエラー

大きなMATファイル（数GB）を処理する場合、メモリ不足が発生する可能性があります。
その場合、`--time-duration`オプション（将来実装予定）を使用してデータの一部のみを処理してください。

### 周波数範囲の調整

システムの特性に応じて、周波数範囲を調整してください：

- 低周波数システム: `--f-low -2.0 --f-up 1.0`
- 高周波数システム: `--f-low 0.0 --f-up 3.0`
- 広帯域システム: `--f-low -1.0 --f-up 2.3`（デフォルト）

## ファイル構成

```
paper_figures/
├── generate_nyquist_figures.py  # メインスクリプト
├── README.md                     # このファイル
├── nyquist_log_scale.png        # 生成された図（対数スケール、PNG）
├── nyquist_log_scale.eps        # 生成された図（対数スケール、EPS）
├── nyquist_linear_scale.png     # 生成された図（線形スケール、PNG）
└── nyquist_linear_scale.eps     # 生成された図（線形スケール、EPS）
```

## 関連ファイル

- `src/frequency_response.py`: 対数周波数グリッドでの周波数応答推定
- `src/fourier_transform.py`: FFTベースの周波数応答推定
- `src/unified_pipeline.py`: 統合パイプライン（両手法を含む）

## ライセンス

このスクリプトは、ガウス過程システム同定プロジェクトの一部です。

## 更新履歴

- 2025-10-12: 初版作成、対数/線形スケールのNyquist図生成機能を実装
