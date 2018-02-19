import h5py
import numpy as np

X = np.arange(1, 31)

X = X.reshape([5,2,3])
print(X)
X= np.split(X, 3, axis=2)
print(X)

print(X)

y = np.random.rand(1, 1000, 1000).astype('float32')

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('X_train', data=X)
h5f.create_dataset('y_train', data=y)
h5f.close()

# Load hdf5 dataset
h5f = h5py.File('data.h5', 'r')
X = h5f['X_train']
Y = h5f['y_train']
h5f.close()