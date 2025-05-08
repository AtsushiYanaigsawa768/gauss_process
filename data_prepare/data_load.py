import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
# Import data
def data_loader():
  tasks_path = Path.cwd() /  "data_prepare/merged.dat"
  try:
      data = np.genfromtxt(tasks_path, delimiter=',')
  except:
      # If the file doesn't exist, create some dummy data for illustration
      print("Data file not found. Creating dummy data for illustration.")
      omega = np.logspace(-1, 2, 100)
      sys_gain_raw = 10 * (1 / (1 + 1j * omega / 10))
      sys_gain_raw = np.abs(sys_gain_raw) + 0.2 * np.random.randn(len(omega))
      arg_g_raw = np.angle(1 / (1 + 1j * omega / 10)) + 0.1 * np.random.randn(len(omega))
      data = np.vstack((omega, sys_gain_raw, arg_g_raw))
    
  # Transpose if necessary to get data in the right shape
  if data.shape[0] == 3:
    omega = data[0, :]
    sys_gain_raw = data[1, :]
    arg_g_raw = data[2, :]
  else:
    omega = data[:, 0]
    sys_gain_raw = data[:, 1]
    arg_g_raw = data[:, 2]

  # Sort data by frequency
  idx = np.argsort(omega)
  omega = omega[idx]
  sys_gain_raw = sys_gain_raw[idx]
  arg_g_raw = arg_g_raw[idx]

  print(f"Number of data points: {len(omega)}")

  # Remove noise using hampel filter

  sys_gain = sys_gain_raw
  arg_g = arg_g_raw
  G = sys_gain * np.exp(1j * arg_g)


  # Gaussian Process Regression for Gain
  X = np.log10(omega).reshape(-1, 1)
  Y = np.log10(sys_gain)*20

  # Split data into training and test sets (80% test, 20% train)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=20)
  
  return X_train, X_test, Y_train, Y_test,omega,sys_gain_raw

if __name__ == "__main__":
  X_train, X_test, Y_train, Y_test = data_loader()
  print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
  print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")