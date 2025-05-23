# 目次

- [概要](#概要)
- [環境構築](#環境構築)
- [ファイルの関係](#ファイルの関係)
- [各ファイルの詳細情報](#各ファイルの詳細情報)
  - [gp ファイル](#gp-ファイル)
  - [GP/dataファイル](#gpdataファイル)
  - [GP/output ファイル](#gpoutput-ファイル)
  - [firファイル](#firファイル)

# 概要
フレキシブルリンクのシステム同定
まず、角周波数と伝達関数のデータに対してガウス過程回帰(GP)を用い、その後それらの情報から、FIRモデルで入力に対するゲイン(フレキシブルリンクがどれだけ動いたのか)を予測する。

# 環境構築



# ファイルの関係

<pre>

gauss_process
├── README.md
├── .gitignore
├── gp
│   ├── output
│   ├── result 
│   ├── data
│   └── *.py -> detailed below
├──fir
│  ├── result 
│  └── *.py -> detailed below
└── adhoc

</pre>

`adhoc` : これまで試行錯誤した際に使用したデータ 実装に関係ない

`fir` :　FIRモデルの構築の際に使用したモデル、データの一覧

`Gauss_Process`: ガウス過程回帰の際に使用したモデル、データの一覧

# 各ファイルの詳細情報

## gp ファイル

#### `gpflow_t_distribution.py`

ガウス過程回帰の際にT分布を仮定したもの

#### `ITGP_robustgp.py`

ガウス過程回帰の一種である、ITGPを実装したもの
レポジトリ：　https://github.com/syrte/robustgp
論文：　　　　https://arxiv.org/abs/2011.11057

#### `k_neighber_gauss_noisey.py`

注意：これはアルゴリズムを実装しただけで、現在の環境に適応させていない。

論文：https://papers.nips.cc/paper_files/paper/2011/file/a8e864d04c95572d1aece099af852d0a-Paper.pdf

#### `linear.py`

ガウス過程回帰ではなく、線形補完を用いたもの。→　`FIR`の場面で用いる

#### `lsqmpmlin.py`

非線形最小二乗法を用いたもの。ガウス過程回帰の比較用に用いた。

#### `sample.py`

`scikit-learn`で標準的に実装されているGPを実装したもの。

## GP/dataファイル

これまで同じ実験機を用いて集められたデータを収集したもの。

## GP/output ファイル

GPファイル内のpyファイルを実行した結果はこのファイルに与えられる。

## firファイル

#### `pure_fir_model.py`

実行するにはmatファイルが必要となる。
詳細はpyファイルを参照すること。