#include "stdio.h"
#include "math.h"
#include "global.h"
#include <stdexcept>
#include "Utility.hh"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include "Python.h"

#ifdef __USE_OMP__
#include "omp.h"
#endif

int sanity_check(double a, double p, double e, double Y){
    int res = 0;
    
    if (p<0.0) return 1;
    if ((e>1.0)||(e<0.0)) return 1;
    if ((Y>1.0)||(Y<-1.0)) return 1;
    if ((a>1.0)||(a<0.0)) return 1;
    
    if (res==1){
        printf("a, p, e, Y = %f %f %f %f ",a, p, e, Y);
        // throw std::invalid_argument( "Sanity check wrong");
    }
    return res;
}

// Define elliptic integrals that use Mathematica's conventions
double EllipticK(double k){
    gsl_sf_result result;
    int status = gsl_sf_ellint_Kcomp_e(sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticK failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticF(double phi, double k){
    gsl_sf_result result;
    int status = gsl_sf_ellint_F_e(phi, sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticF failed with arguments phi:%e k: %e", phi, k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticE(double k){
    gsl_sf_result result;
    int status = gsl_sf_ellint_Ecomp_e(sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticE failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticEIncomp(double phi, double k){
    gsl_sf_result result;
    int status = gsl_sf_ellint_E_e(phi, sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticEIncomp failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticPi(double n, double k){
    gsl_sf_result result;
    int status = gsl_sf_ellint_Pcomp_e(sqrt(k), -n, GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        printf("55: %e\n", k);
        sprintf(str, "EllipticPi failed with arguments (k,n): (%e,%e)", k, n);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticPiIncomp(double n, double phi, double k){
    gsl_sf_result result;
    int status = gsl_sf_ellint_P_e(phi, sqrt(k), -n, GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticPiIncomp failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
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

void KerrGeoConstantsOfMotion(double* E_out, double* L_out, double* Q_out, double a, double p, double e, double x)
{
    *E_out = KerrGeoEnergy(a, p , e, x);
    *L_out = KerrGeoAngularMomentum(a, p, e, x, *E_out);
    *Q_out = KerrGeoCarterConstant(a, p, e, x, *E_out, *L_out);
}

void KerrGeoConstantsOfMotionVectorized(double* E_out, double* L_out, double* Q_out, double* a, double* p, double* e, double* x, int n)
{

    for (int i = 0; i < n; i += 1)
    {
        KerrGeoConstantsOfMotion(&E_out[i], &L_out[i], &Q_out[i], a[i], p[i], e[i], x[i]);
    }
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

void KerrGeoEquatorialMinoFrequencies(double* CapitalGamma_, double* CapitalUpsilonPhi_, double* CapitalUpsilonTheta_, double* CapitalUpsilonr_,
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
    //double zm = 0;
    double a2zp =(pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1))/( (-1 + pow(En, 2)) * (-1));

    double Epsilon0zp = -((pow(L, 2)+ pow(a, 2) * (-1 + pow(En, 2)) * (-1))/(pow(L, 2) * (-1)));

    double zp = pow(a,2)* (1 - pow(En,2)) + pow(L,2);

    double kr = sqrt((r1-r2)/(r1-r3) * (r3-r4)/(r2-r4)); //(*Eq.(13)*)
    //double kTheta = 0; //(*Eq.(13)*)
    double CapitalUpsilonr = (M_PI * sqrt((1 - pow(En, 2)) * (r1-r3) * (r2)))/(2 * EllipticK(pow(kr, 2))); //(*Eq.(15)*)
    double CapitalUpsilonTheta= x * pow(zp,0.5); //(*Eq.(15)*)

    double rp = M + sqrt(pow(M, 2) - pow(a, 2));
    double rm = M - sqrt(pow(M, 2) - pow(a, 2));

    double hr = (r1 - r2)/(r1 - r3);
    double hp = ((r1 - r2) * (r3 - rp))/((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm))/((r1 - r3) * (r2 - rm));

    // (*Eq. (21)*)
    double CapitalUpsilonPhi = (CapitalUpsilonTheta)/(sqrt(Epsilon0zp)) + (2 * a * CapitalUpsilonr)/(M_PI * (rp - rm) * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * ((2 * M * En * rp - a * L)/(r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rp) * EllipticPi(hp, pow(kr, 2))) - (2 * M * En * rm - a * L)/(r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rm) * EllipticPi(hm, pow(kr,2))));

    double CapitalGamma = 4 * pow(M, 2) * En+ (2 * CapitalUpsilonr)/(M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * (En/2 * ((r3 * (r1 + r2 + r3) - r1 * r2) * EllipticK(pow(kr, 2)) + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipticPi(hr,pow(kr, 2)) + (r1 - r3) * (r2 - r4) * EllipticE(pow(kr, 2))) + 2 * M * En * (r3 * EllipticK(pow(kr, 2)) + (r2 - r3) * EllipticPi(hr,pow(kr, 2))) + (2* M)/(rp - rm) * (((4 * pow(M, 2) * En - a * L) * rp - 2 * M * pow(a, 2) * En)/(r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rp) * EllipticPi(hp, pow(kr, 2))) - ((4 * pow(M, 2) * En - a * L) * rm - 2 * M * pow(a, 2) * En)/(r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3)/(r2 - rm) * EllipticPi(hm,pow(kr, 2)))));

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = abs(CapitalUpsilonTheta);
    *CapitalUpsilonr_ = CapitalUpsilonr;
}

void KerrGeoEquatorialCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double a, double p, double e, double x)
{
    double CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR;
    
    KerrGeoEquatorialMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
                                  a, p, e, x);

    *OmegaPhi_ = CapitalUpsilonPhi / CapitalGamma;
    *OmegaTheta_ = CapitalUpsilonTheta / CapitalGamma;
    *OmegaR_ = CapitalUpsilonR / CapitalGamma;

}


void SchwarzschildGeoCoordinateFrequencies(double* OmegaPhi, double* OmegaR, double p, double e)
{
    // Need to evaluate 4 different elliptic integrals here. Cache them first to avoid repeated calls.
	double EllipE 	= EllipticE(4*e/(p-6.0+2*e));
	double EllipK 	= EllipticK(4*e/(p-6.0+2*e));;
	double EllipPi1 = EllipticPi(16*e/(12.0 + 8*e - 4*e*e - 8*p + p*p), 4*e/(p-6.0+2*e));
	double EllipPi2 = EllipticPi(2*e*(p-4)/((1.0+e)*(p-6.0+2*e)), 4*e/(p-6.0+2*e));

    *OmegaPhi = (2*Power(p,1.5))/(Sqrt(-4*Power(e,2) + Power(-2 + p,2))*(8 + ((-2*EllipPi2*(6 + 2*e - p)*(3 + Power(e,2) - p)*Power(p,2))/((-1 + e)*Power(1 + e,2)) - (EllipE*(-4 + p)*Power(p,2)*(-6 + 2*e + p))/(-1 + Power(e,2)) +
          (EllipK*Power(p,2)*(28 + 4*Power(e,2) - 12*p + Power(p,2)))/(-1 + Power(e,2)) + (4*(-4 + p)*p*(2*(1 + e)*EllipK + EllipPi2*(-6 - 2*e + p)))/(1 + e) + 2*Power(-4 + p,2)*(EllipK*(-4 + p) + (EllipPi1*p*(-6 - 2*e + p))/(2 + 2*e - p)))/
        (EllipK*Power(-4 + p,2))));

    *OmegaR = (p*Sqrt((-6 + 2*e + p)/(-4*Power(e,2) + Power(-2 + p,2)))*Pi)/
   (8*EllipK + ((-2*EllipPi2*(6 + 2*e - p)*(3 + Power(e,2) - p)*Power(p,2))/((-1 + e)*Power(1 + e,2)) - (EllipE*(-4 + p)*Power(p,2)*(-6 + 2*e + p))/(-1 + Power(e,2)) +
        (EllipK*Power(p,2)*(28 + 4*Power(e,2) - 12*p + Power(p,2)))/(-1 + Power(e,2)) + (4*(-4 + p)*p*(2*(1 + e)*EllipK + EllipPi2*(-6 - 2*e + p)))/(1 + e) + 2*Power(-4 + p,2)*(EllipK*(-4 + p) + (EllipPi1*p*(-6 - 2*e + p))/(2 + 2*e - p)))/
      Power(-4 + p,2));
}

void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length)
{


    for (int i = 0; i < length; i += 1)
    {
        if (a[i] != 0.0)
        {
            if(abs(x[i]) != 1.)
            {
                KerrGeoCoordinateFrequencies(&OmegaPhi_[i], &OmegaTheta_[i], &OmegaR_[i],
                                      a[i], p[i], e[i], x[i]);
            }
            else
            {
                KerrGeoEquatorialCoordinateFrequencies(&OmegaPhi_[i], &OmegaTheta_[i], &OmegaR_[i],
                                      a[i], p[i], e[i], x[i]);
            }
            
        }
        else
        {
            SchwarzschildGeoCoordinateFrequencies(&OmegaPhi_[i], &OmegaR_[i], p[i], e[i]);
            OmegaTheta_[i] = OmegaPhi_[i];
        }

    }

}


struct params_holder
  {
    double a, p, e, x, Y;
  };

double separatrix_polynomial_full(double p, void *params_in)
{

    struct params_holder* params = (struct params_holder*)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (-4*(3 + e)*Power(p,11) + Power(p,12) + Power(a,12)*Power(-1 + e,4)*Power(1 + e,8)*Power(-1 + x,4)*Power(1 + x,4) - 4*Power(a,10)*(-3 + e)*Power(-1 + e,3)*Power(1 + e,7)*p*Power(-1 + Power(x,2),4) - 4*Power(a,8)*(-1 + e)*Power(1 + e,5)*Power(p,3)*Power(-1 + x,3)*Power(1 + x,3)*(7 - 7*Power(x,2) - Power(e,2)*(-13 + Power(x,2)) + Power(e,3)*(-5 + Power(x,2)) + 7*e*(-1 + Power(x,2))) + 8*Power(a,6)*(-1 + e)*Power(1 + e,3)*Power(p,5)*Power(-1 + Power(x,2),2)*(3 + e + 12*Power(x,2) + 4*e*Power(x,2) + Power(e,3)*(-5 + 2*Power(x,2)) + Power(e,2)*(1 + 2*Power(x,2))) - 8*Power(a,4)*Power(1 + e,2)*Power(p,7)*(-1 + x)*(1 + x)*(-3 + e + 15*Power(x,2) - 5*e*Power(x,2) + Power(e,3)*(-5 + 3*Power(x,2)) + Power(e,2)*(-1 + 3*Power(x,2))) + 4*Power(a,2)*Power(p,9)*(-7 - 7*e + Power(e,3)*(-5 + 4*Power(x,2)) + Power(e,2)*(-13 + 12*Power(x,2))) + 2*Power(a,8)*Power(-1 + e,2)*Power(1 + e,6)*Power(p,2)*Power(-1 + Power(x,2),3)*(2*Power(-3 + e,2)*(-1 + Power(x,2)) + Power(a,2)*(Power(e,2)*(-3 + Power(x,2)) - 3*(1 + Power(x,2)) + 2*e*(1 + Power(x,2)))) - 2*Power(p,10)*(-2*Power(3 + e,2) + Power(a,2)*(-3 + 6*Power(x,2) + Power(e,2)*(-3 + 2*Power(x,2)) + e*(-2 + 4*Power(x,2)))) + Power(a,6)*Power(1 + e,4)*Power(p,4)*Power(-1 + Power(x,2),2)*(-16*Power(-1 + e,2)*(-3 - 2*e + Power(e,2))*(-1 + Power(x,2)) + Power(a,2)*(15 + 6*Power(x,2) + 9*Power(x,4) + Power(e,2)*(26 + 20*Power(x,2) - 2*Power(x,4)) + Power(e,4)*(15 - 10*Power(x,2) + Power(x,4)) + 4*Power(e,3)*(-5 - 2*Power(x,2) + Power(x,4)) - 4*e*(5 + 2*Power(x,2) + 3*Power(x,4)))) - 4*Power(a,4)*Power(1 + e,2)*Power(p,6)*(-1 + x)*(1 + x)*(-2*(11 - 14*Power(e,2) + 3*Power(e,4))*(-1 + Power(x,2)) + Power(a,2)*(5 - 5*Power(x,2) - 9*Power(x,4) + 4*Power(e,3)*Power(x,2)*(-2 + Power(x,2)) + Power(e,4)*(5 - 5*Power(x,2) + Power(x,4)) + Power(e,2)*(6 - 6*Power(x,2) + 4*Power(x,4)))) + Power(a,2)*Power(p,8)*(-16*Power(1 + e,2)*(-3 + 2*e + Power(e,2))*(-1 + Power(x,2)) + Power(a,2)*(15 - 36*Power(x,2) + 30*Power(x,4) + Power(e,4)*(15 - 20*Power(x,2) + 6*Power(x,4)) + 4*Power(e,3)*(5 - 12*Power(x,2) + 6*Power(x,4)) + 4*e*(5 - 12*Power(x,2) + 10*Power(x,4)) + Power(e,2)*(26 - 72*Power(x,2) + 44*Power(x,4)))));

}

double separatrix_polynomial_polar(double p, void *params_in)
{
    struct params_holder* params = (struct params_holder*)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (Power(a,6)*Power(-1 + e,2)*Power(1 + e,4) + Power(p,5)*(-6 - 2*e + p) + Power(a,2)*Power(p,3)*(-4*(-1 + e)*Power(1 + e,2) + (3 + e*(2 + 3*e))*p) - Power(a,4)*Power(1 + e,2)*p*(6 + 2*Power(e,3) + 2*e*(-1 + p) - 3*p - 3*Power(e,2)*(2 + p)));
}


double separatrix_polynomial_equat(double p, void *params_in)
{
    struct params_holder* params = (struct params_holder*)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (Power(a,4)*Power(-3 - 2*e + Power(e,2),2) + Power(p,2)*Power(-6 - 2*e + p,2) - 2*Power(a,2)*(1 + e)*p*(14 + 2*Power(e,2) + 3*p - e*p));
}



double
solver (struct params_holder* params, double (*func)(double, void*), double x_lo, double x_hi)
{
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 1000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 0, r_expected = sqrt (5.0);
    gsl_function F;

    F.function = func;
    F.params = params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    gsl_root_fsolver_set (s, &F, x_lo, x_hi);

    // printf("-----------START------------------- \n");
    // printf("xlo xhi %f %f\n", x_lo, x_hi);
    // double epsrel=0.001;
    double epsrel = 0.00001; // Decreased tolorance
    do
      {
        iter++;
        status = gsl_root_fsolver_iterate (s);
        r = gsl_root_fsolver_root (s);
        x_lo = gsl_root_fsolver_x_lower (s);
        x_hi = gsl_root_fsolver_x_upper (s);
        status = gsl_root_test_interval (x_lo, x_hi, 0.0, epsrel);
       
        // printf("result %f %f %f \n", r, x_lo, x_hi);
      }
    while (status == GSL_CONTINUE && iter < max_iter);
    
    // printf("result %f %f %f \n", r, x_lo, x_hi);
    // printf("stat, iter, GSL_SUCCESS %d %d %d\n", status, iter, GSL_SUCCESS);
    // printf("-----------END------------------- \n");

    if (status != GSL_SUCCESS)
    {
        // warning if it did not converge otherwise throw error
        if (iter == max_iter){
            printf("WARNING: Maximum iteration reached in Utility.cc in Brent root solver.\n");
            printf("Result=%f, x_low=%f, x_high=%f \n", r, x_lo, x_hi);
            printf("a, p, e, Y, sep = %f %f %f %f %f\n", params->a, params->p, params->e, params->Y, get_separatrix(params->a, params->e, r));
        }
        else
        {
            throw std::invalid_argument( "In Utility.cc Brent root solver failed");
        }
    }

    gsl_root_fsolver_free (s);
    return r;
}

double get_separatrix(double a, double e, double x)
{
    double p_sep, z1, z2;
    double sign;
    if (a == 0.0)
    {
        p_sep = 6.0 + 2.0 * e;
        return p_sep;
    }
    else if ((e == 0.0) & (abs(x) == 1.0))
    {
        z1 = 1. + pow((1. - pow(a,  2)), 1./3.) * (pow((1. + a), 1./3.)
             + pow((1. - a), 1./3.));

        z2 = sqrt(3. * pow(a, 2) + pow(z1, 2));

        // prograde
        if (x > 0.0)
        {
            sign = -1.0;
        }
        // retrograde
        else
        {
            sign = +1.0;
        }

        p_sep = (3. + z2 + sign * sqrt((3. - z1) * (3. + z1 + 2. * z2)));
        return p_sep;
    }
    else if (x == 1.0) // Eccentric Prograde Equatorial
    {
        // fills in p and Y with zeros
        struct params_holder params = {a, 0.0, e, x, 0.0};
        double x_lo, x_hi;

        x_lo = 1.0 + e;
        x_hi = 6 + 2. * e;

        p_sep = solver (&params, &separatrix_polynomial_equat, x_lo, x_hi);
        return p_sep;
    }
    else if (x == -1.0) // Eccentric Retrograde Equatorial
    {
        // fills in p and Y with zeros
        struct params_holder params = {a, 0.0, e, x, 0.0};
        double x_lo, x_hi;

        x_lo = 6 + 2. * e;
        x_hi = 5+e+4 *Sqrt(1+e); 

        p_sep = solver (&params, &separatrix_polynomial_equat, x_lo, x_hi);
        return p_sep;
    }
    else
    {
        // fills in p and Y with zeros
        struct params_holder params = {a, 0.0, e, x, 0.0};
        double x_lo, x_hi;

        // solve for polar p_sep
        x_lo = 1.0 + sqrt(3.0) + sqrt(3.0 + 2.0 * sqrt(3.0));
        x_hi = 8.0;



        double polar_p_sep = solver (&params, &separatrix_polynomial_polar, x_lo, x_hi);
        if (x == 0.0) return polar_p_sep;

        double equat_p_sep;
        if (x > 0.0)
        {
            x_lo = 1.0 + e;
            x_hi = 6 + 2. * e;

            equat_p_sep = solver (&params, &separatrix_polynomial_equat, x_lo, x_hi);

            x_lo = equat_p_sep;
            x_hi = polar_p_sep;
        } else
        {
            x_lo = polar_p_sep;
            x_hi = 12.0;
        }

        p_sep = solver (&params, &separatrix_polynomial_full, x_lo, x_hi);

        return p_sep;
    }
}

void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length)
{


    for (int i = 0; i < length; i += 1)
    {
        separatrix[i] = get_separatrix(a[i], e[i], x[i]);
    }

}

double Y_to_xI_eq(double x, void *params_in)
{
    struct params_holder* params = (struct params_holder*)params_in;

    double a = params->a;
    double p = params->p;
    double e = params->e;
    double Y = params->Y;

    double E, L, Q;

    // get constants of motion
    KerrGeoConstantsOfMotion(&E, &L, &Q, a, p, e, x);
    double Y_ = L / sqrt(pow(L, 2) + Q);

    return Y - Y_;
}

#define YLIM 0.998
double Y_to_xI(double a, double p, double e, double Y)
{
    // TODO: check this
    if (abs(Y) > YLIM) return Y;
    // fills in x with 0.0
    struct params_holder params = {a, p, e, 0.0, Y};
    double x_lo, x_hi;

    // set limits
    // assume Y is close to x
    x_lo = Y - 0.15;
    x_hi = Y + 0.15;

    x_lo = x_lo > -YLIM? x_lo : -YLIM;
    x_hi = x_hi < YLIM? x_hi : YLIM;

    double x = solver (&params, &Y_to_xI_eq, x_lo, x_hi);
   
    return x;
}

void Y_to_xI_vector(double* x, double* a, double* p, double* e, double* Y, int length)
{
    for (int i = 0; i < length; i += 1)
    {
        x[i] = Y_to_xI(a[i], p[i], e[i], Y[i]);
    }

}
