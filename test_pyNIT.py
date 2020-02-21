from pyNIT import NIT


p0 = 13.0
e0 = 0.4

t, p, e, Phi_phi, Phi_r = NIT(p0, e0)

for i in range(len(t)):
    print(t[i], p[i], e[i], Phi_phi[i], Phi_r[i])
