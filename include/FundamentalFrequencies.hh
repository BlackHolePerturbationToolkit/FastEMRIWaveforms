#ifndef __FUND_FREQS__
#define __FUND_FREQS__

void KerrGeoMinoFrequencies(double* CapitalGamma_, double* CapitalUpsilonPhi_, double* CapitalUpsilonTheta_, double* CapitalUpsilonr_,
                              double a, double p, double e, double x);

void KerrGeoCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                            double a, double p, double e, double x);

void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);

void SchwarzschildGeoCoordinateFrequencies(double* OmegaPhi, double* OmegaR, double p, double e);

double get_separatrix(double a, double e, double x);
void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length);

#endif // __FUND_FREQS__
