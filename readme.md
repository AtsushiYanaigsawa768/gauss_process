# ファイル関係
model_result: 各実験の結果が置かれています。
`original.png` : 生データを図にしたものです。
`システム同定.png` : 共通課題で作成した、システム同定の結果です。
`remove_noise+RBF.png` : ノイズ除去後に、RBF(Const * RBF)を適応させたものです。
`RBF+RBF.png` : C*RBF + C*RBF というkernelを用いて実行したものです。

result:　`gaussian.py`を実行すると結果が出力されるフォルダです。
`merged.dat`が、生データの保存場所です。

`gaussian.py`が、pythonのscikit-learnを用いてGBFを実行するファイルです。

# Gaussian.py

## Hyper Parameterの設定
`png_name`: 出力されるpngの名前を指定することが出来ます。
`calculate_time` : Trueの場合、どれぐらい実行に時間がかかったかを出力します。

以下のHyper Parameterは、余り変更することを推奨しません。
`noise_filter` : Trueの場合、Hampleフィルターを利用することが出来ます。
`test_set_ration`: テストデータに使われる割合が、どれぐらいかを指定します。データ数ができるだけ少なくしたいので、今は0.8に指定しています。

## kernelの設定
ガウス過程回帰が使用できる、カーネルの種類を表しています。
kernel_setには、単一のカーネルが、
kernels_modelには、複数のカーネルを合したものが入っています。

```
kernel = const * rbf
```
や、

```
kernel = kernels_model[0]
```
のように使用してください。

## RBFの設定
詳しくは`gauss.py`を参照してください。

## 実行方法
`sklearn`や`matplotlib`さえあれば、実行できると思います。
複雑になる場合は別途、決めた方がいいかもしれません。
