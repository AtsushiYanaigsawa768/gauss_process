import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.lib.stride_tricks import sliding_window_view

#––– データ読み込み（おおむね元コードと同じ）––––––––––––––––––––––––––––––––––––––––––––––––––––
io = loadmat('data_hour.mat')
for k,v in io.items():
  if not k.startswith('__'):
    mat = v; break
time = mat[0,:100000]
y    = mat[1,:100000]
u    = mat[2,:100000]
N    = len(u)

#––– モデル長 L を固定（軽量化）––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
L   = 32       # タップ数を 32 に固定
mu  = 0.01     # LMS 学習率（調整要）
h   = np.zeros(L)
yhat = np.zeros(N)
e    = np.zeros(N)

#––– 入力のスライド窓を作っておく –––––––––––––––––––––––––––––––––––––––––––––––––––––––––
u_pad = np.concatenate([np.zeros(L-1), u])
U = sliding_window_view(u_pad, window_shape=L)[ :N, ::-1 ]  # [u[n],u[n-1],…,u[n-L+1]]

#––– LMS アップデートループ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
for n in range(N):
  phi = U[n]            # 長さ L のベクトル
  yhat[n] = phi.dot(h)
  e[n]    = y[n] - yhat[n]
  h += mu * e[n] * phi  # LMS update

#––– プロット––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# plot only data from time ≥ 10s
start = np.searchsorted(time, 10)

plt.figure(figsize=(8,5))
plt.subplot(2,1,1)
plt.plot(time[start:], y[start:], 'k', label='y (meas)')
plt.plot(time[start:], yhat[start:], 'r--', label='yhat (LMS)')
plt.legend()
plt.subplot(2,1,2)
plt.plot(time[start:], e[start:], 'b')
plt.title('error')
plt.tight_layout()
plt.show()
