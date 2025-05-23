import numpy as np

def hampel_filter(data, window_size=10, n_sigmas=3):
  """
  Apply a Hampel filter to remove outliers.
  
  Args:
    data: input data array
    window_size: size of window for median calculation
    n_sigmas: number of standard deviations to use for outlier detection
  
  Returns:
    Filtered data array
  """
  filtered_data = data.copy()
  n = len(data)
  k = window_size // 2
  
  for i in range(n):
    start = max(0, i - k)
    end = min(n, i + k + 1)
    
    window = data[start:end]
    median = np.median(window)
    mad = np.median(np.abs(window - median))  # MAD = Median Absolute Deviation
    
    # Convert MAD to standard deviation (scale factor 1.4826)
    sigma = 1.4826 * mad
    
    # Replace outliers with median
    if np.abs(data[i] - median) > n_sigmas * sigma and sigma > 0:
      filtered_data[i] = median
      
  return filtered_data

def accuracy( Y_train, Y_test,Y_test_pred, Y_train_pred = None):
  """Calculate the accuracy of the model by using mse
  Y_train -> hamepel filter -> (new) Y_train <=> Y_train_pred 
  Y_test -> hamepel filter -> (new) Y_test <=> Y_test_pred
  """
  # Apply Hampel filter to Y_train and Y_test
  Y_train_filtered = hampel_filter(Y_train)
  Y_test_filtered = hampel_filter(Y_test)

  # Calculate MSE for training set
  if Y_train_pred is None:
    mse_train = None
  else:
    Y_train_pred = hampel_filter(Y_train_pred)
    mse_train = np.mean((Y_train_filtered - Y_train_pred)**2)

  # Calculate MSE for test set
  mse_test = np.mean((Y_test_filtered - Y_test_pred)**2)
  if mse_train is  None:
    print(f"Train MSE: {mse_train:.4f}")
  else:
    print(f"Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
  return mse_train, mse_test