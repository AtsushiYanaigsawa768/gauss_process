import numpy as np
import pandas as pd

### ==== 1. 設定（ここだけ変更） ====
input_csv = './result/ITGP_data_fit_t.csv'     # 入力CSVファイル名
output_csv = './result/data_g.csv'             # 出力CSVファイル名

col_omega = 'omega'                   # 周波数列の列名
col_real  = 'Re_G_fit'               # 実部の列名
col_imag  = 'Im_G_fit'               # 虚部の列名

### ==== 2. CSV読み込み ====
df = pd.read_csv(input_csv)
omega = df[col_omega].values
ReG = df[col_real].values
ImG = df[col_imag].values
G_pos = ReG + 1j * ImG

### ==== 3. 周波数間隔と共役スペクトル構築 ====
domega = np.mean(np.diff(omega))
G_neg = np.conj(G_pos[1:][::-1])          # ω = 0 除いて対称に
G_full = np.concatenate([G_neg, G_pos])   # [負周波数, 正周波数]

### ==== 4. 逆フーリエ変換と時間軸構築 ====
g_time = np.fft.ifft(np.fft.ifftshift(G_full))
g_time = np.fft.fftshift(g_time)

T = 2 * np.pi / domega                  # 合計時間幅
dt = T / len(G_full)                   # サンプリング間隔
time = np.linspace(-T/2, T/2 - dt, len(G_full))

### ==== 5. エネルギー計算 ====
energy_re = np.sum(np.real(g_time)**2)
energy_im = np.sum(np.imag(g_time)**2)
energy_ratio = energy_im / energy_re

print(f'実部のエネルギー: {energy_re:.6g}')
print(f'虚部のエネルギー: {energy_im:.6g}')
print(f'虚部/実部の比率: {energy_ratio:.6%}')

### ==== 6. CSV出力 ====
df_out = pd.DataFrame({
    'time': time,
    'g_time_Re': np.real(g_time),
    'g_time_Im': np.imag(g_time)
})
df_out.to_csv(output_csv, index=False)