from scipy.io import loadmat
import matplotlib.pyplot as plt
# ファイルを読み込む
io_data = loadmat('data_flexible.mat')

# 全変数を表示して、最初のデータ配列を取得
for name, arr in io_data.items():
  if not name.startswith('__'):
    print(f'{name} ⇒ shape: {arr.shape}')
    mat = arr
    break
# 時間，y，u を取り出す
time = mat[0, :100000].ravel()
y    = mat[1, :100000].ravel()
u    = mat[2, :100000].ravel()

# y vs time
plt.figure()
plt.plot(time, y, color='C0')
plt.xlabel('time')
plt.ylabel('y')
plt.title('y vs time')
plt.grid(True)

# u vs time
plt.figure()
plt.plot(time, u, color='C1')
plt.xlabel('time')
plt.ylabel('u')
plt.title('u vs time')
plt.grid(True)

plt.show()