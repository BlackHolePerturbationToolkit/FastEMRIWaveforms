#ifndef __FUND_FREQS__
#define __FUND_FREQS__

void throw_python_error(char* str_in, int status);

int sanity_check(double a, double p, double e, double Y);

void KerrGeoMinoFrequencies(double* CapitalGamma_, double* CapitalUpsilonPhi_, double* CapitalUpsilonTheta_, double* CapitalUpsilonr_,
                              double a, double p, double e, double x);

void KerrGeoCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                            double a, double p, double e, double x);

void KerrGeoEquatorialMinoFrequencies(double* CapitalGamma_, double* CapitalUpsilonPhi_, double* CapitalUpsilonTheta_, double* CapitalUpsilonr_,
                              double a, double p, double e, double x);

void KerrGeoEquatorialCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                            double a, double p, double e, double x);

void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);

void SchwarzschildGeoCoordinateFrequencies(double* OmegaPhi, double* OmegaR, double p, double e);

double get_separatrix(double a, double e, double x);
void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length);

void KerrGeoConstantsOfMotionVectorized(double* E_out, double* L_out, double* Q_out, double* a, double* p, double* e, double* x, int n);
void KerrGeoConstantsOfMotion(double* E_out, double* L_out, double* Q_out, double a, double p, double e, double x);

double Y_to_xI(double a, double p, double e, double Y);
void Y_to_xI_vector(double* x, double* a, double* p, double* e, double* Y, int length);

#endif // __FUND_FREQS__
