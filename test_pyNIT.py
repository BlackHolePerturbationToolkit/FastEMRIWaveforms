from pyNIT import NIT
import pdb
import time
import argparse
from few.nit import run_nit

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--time", "-t", default=0, type=int)
args = parser.parse_args()

M = 1e6
mu = 1e-3
p0 = 10.0
e0 = 0.75
err = 1e-11

# return coordinate time non-upsampled vectors
t1, p1, e1, Phi_phi1, Phi_r1 = run_nit(
    M,
    mu,
    p0,
    e0,
    err=err,
    in_coordinate_time=True,
    dt=None,
    T=None,
    new_t=None,
    spline_kwargs={},
)

# return coordinate time upsampled vectors with dt and T
t2, p2, e2, Phi_phi2, Phi_r2 = run_nit(
    M,
    mu,
    p0,
    e0,
    err=err,
    in_coordinate_time=True,
    dt=10.0,
    T=1e7,
    new_t=None,
    spline_kwargs={},
)

# return dimensionless time upsampled vectors with dt and T
# in this case dt and T are also dimensionless
t3, p3, e3, Phi_phi3, Phi_r3 = run_nit(
    M,
    mu,
    p0,
    e0,
    err=err,
    in_coordinate_time=False,
    dt=5.0,
    T=1e6,
    new_t=None,
    spline_kwargs={},
)

# return dimensionless time non-upsampled vectors
t4, p4, e4, Phi_ph4, Phi_r4 = run_nit(
    M,
    mu,
    p0,
    e0,
    err=err,
    in_coordinate_time=False,
    dt=None,
    T=None,
    new_t=None,
    spline_kwargs={},
)

import pdb

pdb.set_trace()

for i in range(len(t)):
    print(t[i], p[i], e[i], Phi_phi[i], Phi_r[i])

if args.time:
    st = time.perf_counter()
    for _ in range(args.time):
        t, p, e, Phi_phi, Phi_r = NIT(M, mu, p0, e0, err=err)
    et = time.perf_counter()
    print("time per:", (et - st) / args.time)


pdb.set_trace()
