import numpy as np

try:
    from FEW import FastEMRIWaveforms
except ImportError:
    pass

import argparse
import time

from nn import NN

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--time", "-t", default=0, type=int)
args = parser.parse_args()

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

traj = np.genfromtxt("insp_p12.5_e0.7_tspacing_1M.dat")
input_len = len(traj)
print(input_len)

p = np.asarray(traj[:, 0], dtype=np.float32)
e = np.asarray(traj[:, 1], dtype=np.float32)
Phi_phi = np.asarray(traj[:, 2], dtype=np.float32)
Phi_r = np.asarray(traj[:, 3], dtype=np.float32)

l = np.zeros(2214, dtype=np.int32)
m = np.zeros(2214, dtype=np.int32)
n = np.zeros(2214, dtype=np.int32)

ind = 0
for l_i in range(2, 10 + 1):
    for m_i in range(1, l_i + 1):
        for n_i in range(-20, 20 + 1):
            l[ind] = l_i
            m[ind] = m_i
            n[ind] = n_i
            ind += 1


input_mat = np.concatenate([p, e]).astype(np.float32)


break_index = 80

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
    break_index,
    l,
    m,
    n,
    input_len,
)

check = few_class.run_nn(input_mat, input_len, Phi_phi, Phi_r)
if args.time:
    st = time.perf_counter()
    for _ in range(args.time):
        check = few_class.run_nn(input_mat, input_len, Phi_phi, Phi_r)
    et = time.perf_counter()
    print("time per:", (et - st) / args.time)

import pdb

pdb.set_trace()
