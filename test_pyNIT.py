from pyNIT import NIT
import pdb
import time
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--time", "-t", default=0, type=int)
args = parser.parse_args()

M = 1e5
mu = 1e1
p0 = 13.0
e0 = 0.4
err = 1e-11

t, p, e, Phi_phi, Phi_r = NIT(M, mu, p0, e0, err=err)

for i in range(len(t)):
    print(t[i], p[i], e[i], Phi_phi[i], Phi_r[i])

if args.time:
    st = time.perf_counter()
    for _ in range(args.time):
        t, p, e, Phi_phi, Phi_r = NIT(M, mu, p0, e0, err=err)
    et = time.perf_counter()
    print("time per:", (et - st) / args.time)


pdb.set_trace()
