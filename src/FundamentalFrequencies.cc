#include "stdio.h"
#include "math.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_ellint.h>

#ifdef __USE_OMP__
#include "omp.h"
#endif


// Define elliptic integrals that use Mathematica's conventions
double EllipticK(double k){
        return gsl_sf_ellint_Kcomp(sqrt(k), GSL_PREC_DOUBLE);
}

double EllipticF(double phi, double k){
        return gsl_sf_ellint_F(phi, sqrt(k), GSL_PREC_DOUBLE) ;
}

double EllipticE(double k){
        return gsl_sf_ellint_Ecomp(sqrt(k), GSL_PREC_DOUBLE);
}

double EllipticEIncomp(double phi, double k){
        return gsl_sf_ellint_E(phi, sqrt(k), GSL_PREC_DOUBLE) ;
}

double EllipticPi(double n, double k){
        return gsl_sf_ellint_Pcomp(sqrt(k), -n, GSL_PREC_DOUBLE);
}

double EllipticPiIncomp(double n, double phi, double k){
        return gsl_sf_ellint_P(phi, sqrt(k), -n, GSL_PREC_DOUBLE);
}


double CapitalDelta(double r, double a)
{
    return pow(r, 2.) - 2. * r + pow(a, 2.);
}

double f(double r, double a, double zm)
{
    return pow(r, 4) + pow(a, 2) * (r * (r + 2) + pow(zm, 2) * CapitalDelta(r, a));
}

double g(double r, double a, double zm)
{
    return 2 * a * r;
}

double h(double r, double a, double zm)
{
    return r * (r - 2) + pow(zm, 2)/(1 - pow(zm, 2)) * CapitalDelta(r, a);
}

double d(double r, double a, double zm)
{
    return (pow(r, 2) + pow(a, 2) * pow(zm, 2)) * CapitalDelta(r, a);
}

double KerrGeoEnergy(double a, double p, double e, double x)
{

	double r1 = p/(1.-e);
	double r2 = p/(1.+e);

	double zm = sqrt(1.-pow(x, 2.));


    double Kappa    = d(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * d(r2, a, zm);
    double Epsilon  = d(r1, a, zm) * g(r2, a, zm) - g(r1, a, zm) * d(r2, a, zm);
    double Rho      = f(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * f(r2, a, zm);
    double Eta      = f(r1, a, zm) * g(r2, a, zm) - g(r1, a, zm) * f(r2, a, zm);
    double Sigma    = g(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * g(r2, a, zm);

	return sqrt((Kappa * Rho + 2 * Epsilon * Sigma - x * 2 * sqrt(Sigma * (Sigma * pow(Epsilon, 2) + Rho * Epsilon * Kappa - Eta * pow(Kappa,2))/pow(x, 2)))/(pow(Rho, 2) + 4 * Eta * Sigma));
}

double KerrGeoAngularMomentum(double a, double p, double e, double x, double En)
{
    double r1 = p/(1-e);

	double zm = sqrt(1-pow(x,2));

    return (-En * g(r1, a, zm) + x * sqrt((-d(r1, a, zm) * h(r1, a, zm) + pow(En, 2) * (pow(g(r1, a, zm), 2) + f(r1, a, zm) * h(r1, a, zm)))/pow(x, 2)))/h(r1, a, zm);
}

double KerrGeoCarterConstant(double a, double p, double e, double x, double En, double L)
{
    double zm = sqrt(1-pow(x,2));

	return pow(zm, 2) * (pow(a, 2) * (1 - pow(En, 2)) + pow(L, 2)/(1 - pow(zm, 2)));
}

void KerrGeoRadialRoots(double* r1_, double*r2_, double* r3_, double* r4_, double a, double p, double e, double x, double En, double Q)
{
    double M = 1.0;
    double r1 = p / (1 - e);
    double r2 = p /(1+e);
    double AplusB = (2 * M)/(1-pow(En, 2)) - (r1 + r2);
    double AB = (pow(a, 2) * Q)/((1-pow(En, 2)) * r1 *  r2);
    double r3 = (AplusB+sqrt(pow(AplusB, 2) - 4 * AB))/2;
    double r4 = AB/r3;

    *r1_ = r1;
    *r2_ = r2;
    *r3_ = r3;
    *r4_ = r4;
}


void KerrGeoMinoFrequencies(double* CapitalGamma_, double* CapitalUpsilonPhi_, double* CapitalUpsilonTheta_, double* CapitalUpsilonr_,
                              double a, double p, double e, double x)
{
    double M = 1.0;

    double En = KerrGeoEnergy(a, p, e, x);
    double L = KerrGeoAngularMomentum(a, p, e, x, En);
    double Q = KerrGeoCarterConstant(a, p, e, x, En, L);

    // get radial roots
    double r1, r2, r3, r4;
    KerrGeoRadialRoots(&r1, &r2, &r3, &r4, a, p, e, x, En, Q);

    double Epsilon0 = pow(a, 2) * (1 - pow(En, 2))/pow(L, 2);
    double zm = 1 - pow(x, 2);
    double a2zp =(pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm))/( (-1 + pow(En, 2)) * (-1 + zm));

    double Epsilon0zp = -((pow(L, 2)+ pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm))/(pow(L, 2) * (-1 + zm)));

    double zmOverZp = zm/((pow(L, 2)+ pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm))/(pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)));

    double kr = sqrt((r1-r2)/(r1-r3) * (r3-r4)/(r2-r4)); //(*Eq.(13)*)
    double kTheta = sqrt(zmOverZp); //(*Eq.(13)*)
    double CapitalUpsilonr = (M_PI * sqrt((1 - pow(En, 2)) * (r1-r3) * (r2-r4)))/(2 * EllipticK(pow(kr, 2))); //(*Eq.(15)*)
    double CapitalUpsilonTheta=(M_PI * L * sqrt(Epsilon0zp))/(2 * EllipticK(pow(kTheta, 2))); //(*Eq.(15)*)

    double rp = M + sqrt(pow(M, 2) - pow(a, 2));
    double rm = M - sqrt(pow(M, 2) - pow(a, 2));

    double hr = (r1 - r2)/(r1 - r3);
    double hp = ((r1 - r2) * (r3 - rp))/((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm))/((r1 - r3) * (r2 - rm));

    // (*Eq. (21)*)
    double CapitalUpsilonPhi = (2 * CapitalUpsilonTheta)/(M_PI * sqrt(Epsilon0zp)) * EllipticPi(zm, pow(kTheta, 2)) + (2 * a * CapitalUpsilonr)/(M_PI * (rp - rm) * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * ((2 * M * En * rp - a * L)/(r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rp) * EllipticPi(hp, pow(kr, 2))) - (2 * M * En * rm - a * L)/(r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rm) * EllipticPi(hm, pow(kr,2))));

    double CapitalGamma = 4 * pow(M, 2) * En + (2 * a2zp * En * CapitalUpsilonTheta)/(M_PI * L * sqrt(Epsilon0zp)) * (EllipticK(pow(kTheta, 2)) -  EllipticE(pow(kTheta, 2))) + (2 * CapitalUpsilonr)/(M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * (En/2 * ((r3 * (r1 + r2 + r3) - r1 * r2) * EllipticK(pow(kr, 2)) + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipticPi(hr,pow(kr, 2)) + (r1 - r3) * (r2 - r4) * EllipticE(pow(kr, 2))) + 2 * M * En * (r3 * EllipticK(pow(kr, 2)) + (r2 - r3) * EllipticPi(hr,pow(kr, 2))) + (2* M)/(rp - rm) * (((4 * pow(M, 2) * En - a * L) * rp - 2 * M * pow(a, 2) * En)/(r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rp) * EllipticPi(hp, pow(kr, 2))) - ((4 * pow(M, 2) * En - a * L) * rm - 2 * M * pow(a, 2) * En)/(r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rm) * EllipticPi(hm,pow(kr, 2)))));

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = CapitalUpsilonTheta;
    *CapitalUpsilonr_ = CapitalUpsilonr;
}


void KerrGeoCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double a, double p, double e, double x)
{

    double CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR;
    KerrGeoMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
                                  a, p, e, x);
    *OmegaPhi_ = CapitalUpsilonPhi / CapitalGamma;
    *OmegaTheta_ = CapitalUpsilonTheta / CapitalGamma;
    *OmegaR_ = CapitalUpsilonR / CapitalGamma;

}

void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length)
{

    #ifdef __USE_OMP__
    #pragma omp parallel for
    #endif
    for (int i = 0; i < length; i += 1)
    {
        KerrGeoCoordinateFrequencies(&OmegaPhi_[i], &OmegaTheta_[i], &OmegaR_[i],
                                      a[i], p[i], e[i], x[i]);
    }

}


/*
int main()
{
    double a = 0.5;
    double p = 10.0;
    double e = 0.2;
    double iota = 0.4;
    double x = cos(iota);

    double temp = KerrGeoMinoFrequencies(a, p, e, x);

    //printf("%e %e %e\n", En, L, C);
}
*/
