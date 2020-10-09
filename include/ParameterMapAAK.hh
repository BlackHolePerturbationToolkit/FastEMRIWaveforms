#ifndef  __PARMAP_H__
#define  __PARMAP_H__

void ParMapVector(double* v_map, double* M_map, double* S_map, double* OmegaPhi, double* OmegaTheta, double* OmegaR,
                  double* p, double* e, double* iota, double M, double s, int length);

void waveform(double* hI, double* hII,
            double* tvec, double* evec, double* vvec,
            double* gimvec, double* Phivec, double* alpvec, double* nuvec, double* gimdotvec, double* OmegaPhi_spin_mapped,
            double M_phys, double mu, double lam, double qS, double phiS, double qK, double phiK, double dist, int length, int nmodes);

#endif // __PARMAP_H__
