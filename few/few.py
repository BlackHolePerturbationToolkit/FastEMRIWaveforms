import numpy as np

try:
    from FEW import FastEMRIWaveforms
except ImportError:
    pass

from nn import NN

check = NN()

bias = []
weight = []
dim1 = []
dim2 = []
for i in range(check.num_layers):
    weight.append(check.layers[i].w.flatten())
    dim1.append(check.layers[i].w.shape[0])
    dim2.append(check.layers[i].w.shape[1])
    bias.append(check.layers[i].b)

weight = np.concatenate(weight)
bias = np.concatenate(bias)
dim1 = np.asarray(dim1).astype(np.int32)
dim2 = np.asarray(dim2).astype(np.int32)

transform_matrix = np.asarray(
    np.genfromtxt("few/files/reduced_basis.dat", dtype=np.complex64)
)
trans_dim1, trans_dim2 = transform_matrix.shape
transform_matrix = transform_matrix.flatten()
transform_factor = 1000.0

time_batch_size = 10
few_class = FastEMRIWaveforms(
    time_batch_size,
    check.num_layers,
    dim1,
    dim2,
    weight,
    bias,
    transform_matrix,
    trans_dim1,
    trans_dim2,
    transform_factor,
)

import pdb

pdb.set_trace()
