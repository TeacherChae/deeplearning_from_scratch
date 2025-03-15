import numpy as np

arr = np.array([0.1, 0.2, 0.3, 0.4])
arr_2d = arr.reshape(1, 4)
print(arr_2d)
brr = np.array([2, 7, 5, 9])
brr_2d = brr.reshape(1, 4)
batch_size = arr_2d.shape[0]
print(arr_2d[np.arange(batch_size), brr_2d])
x = np.log(arr_2d[np.arange(batch_size), brr_2d] + 1e-7)
print(x)
y = -np.sum(x)
print(y)