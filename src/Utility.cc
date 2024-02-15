#include "stdio.h"
#include "math.h"
#include "global.h"
#include <stdexcept>
#include "Utility.hh"
#include <iostream>
#include <chrono>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include "Python.h"

#ifdef __USE_OMP__
#include "omp.h"
#endif

using namespace std;
using namespace std::chrono;

int sanity_check(double a, double p, double e, double Y)
{
    int res = 0;

    if (p < 0.0)
        return 1;
    if ((e > 1.0) || (e < 0.0))
        return 1;
    if ((Y > 1.0) || (Y < -1.0))
        return 1;
    if ((a > 1.0) || (a < 0.0))
        return 1;

    if (res == 1)
    {
        printf("a, p, e, Y = %f %f %f %f ", a, p, e, Y);
        // throw std::invalid_argument( "Sanity check wrong");
    }
    return res;
}

// Define elliptic integrals that use Mathematica's conventions
double EllipticK(double k)
{
    gsl_sf_result result;
    // cout << "CHECK1" << endl;
    int status = gsl_sf_ellint_Kcomp_e(sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticK failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticF(double phi, double k)
{
    gsl_sf_result result;
    // cout << "CHECK2" << endl;
    int status = gsl_sf_ellint_F_e(phi, sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticF failed with arguments phi:%e k: %e", phi, k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticE(double k)
{
    gsl_sf_result result;
    // cout << "CHECK3 " << k << endl;
    int status = gsl_sf_ellint_Ecomp_e(sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticE failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticEIncomp(double phi, double k)
{
    gsl_sf_result result;
    // cout << "CHECK4" << endl;
    int status = gsl_sf_ellint_E_e(phi, sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticEIncomp failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticPi(double n, double k)
{
    // cout << "CHECK6" << endl;
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

double EllipticPiIncomp(double n, double phi, double k)
{
    // cout << "CHECK7" << endl;
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
    return r * (r - 2) + pow(zm, 2) / (1 - pow(zm, 2)) * CapitalDelta(r, a);
}

double d(double r, double a, double zm)
{
    return (pow(r, 2) + pow(a, 2) * pow(zm, 2)) * CapitalDelta(r, a);
}

double KerrGeoEnergy(double a, double p, double e, double x)
{

    double r1 = p / (1. - e);
    double r2 = p / (1. + e);

    double zm = sqrt(1. - pow(x, 2.));

    double Kappa = d(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * d(r2, a, zm);
    double Epsilon = d(r1, a, zm) * g(r2, a, zm) - g(r1, a, zm) * d(r2, a, zm);
    double Rho = f(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * f(r2, a, zm);
    double Eta = f(r1, a, zm) * g(r2, a, zm) - g(r1, a, zm) * f(r2, a, zm);
    double Sigma = g(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * g(r2, a, zm);

    return sqrt((Kappa * Rho + 2 * Epsilon * Sigma - x * 2 * sqrt(Sigma * (Sigma * pow(Epsilon, 2) + Rho * Epsilon * Kappa - Eta * pow(Kappa, 2)) / pow(x, 2))) / (pow(Rho, 2) + 4 * Eta * Sigma));
}

double KerrGeoAngularMomentum(double a, double p, double e, double x, double En)
{
    double r1 = p / (1 - e);

    double zm = sqrt(1 - pow(x, 2));

    return (-En * g(r1, a, zm) + x * sqrt((-d(r1, a, zm) * h(r1, a, zm) + pow(En, 2) * (pow(g(r1, a, zm), 2) + f(r1, a, zm) * h(r1, a, zm))) / pow(x, 2))) / h(r1, a, zm);
}

double KerrGeoCarterConstant(double a, double p, double e, double x, double En, double L)
{
    double zm = sqrt(1 - pow(x, 2));

    return pow(zm, 2) * (pow(a, 2) * (1 - pow(En, 2)) + pow(L, 2) / (1 - pow(zm, 2)));
}

void KerrGeoConstantsOfMotion(double *E_out, double *L_out, double *Q_out, double a, double p, double e, double x)
{
    *E_out = KerrGeoEnergy(a, p, e, x);
    *L_out = KerrGeoAngularMomentum(a, p, e, x, *E_out);
    *Q_out = KerrGeoCarterConstant(a, p, e, x, *E_out, *L_out);
}

void KerrGeoConstantsOfMotionVectorized(double *E_out, double *L_out, double *Q_out, double *a, double *p, double *e, double *x, int n)
{
#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i += 1)
    {
        KerrGeoConstantsOfMotion(&E_out[i], &L_out[i], &Q_out[i], a[i], p[i], e[i], x[i]);
    }
}

void KerrGeoRadialRoots(double *r1_, double *r2_, double *r3_, double *r4_, double a, double p, double e, double x, double En, double Q)
{
    double M = 1.0;
    double r1 = p / (1 - e);
    double r2 = p / (1 + e);
    double AplusB = (2 * M) / (1 - pow(En, 2)) - (r1 + r2);
    double AB = (pow(a, 2) * Q) / ((1 - pow(En, 2)) * r1 * r2);
    double r3 = (AplusB + sqrt(pow(AplusB, 2) - 4 * AB)) / 2;
    double r4 = AB / r3;

    *r1_ = r1;
    *r2_ = r2;
    *r3_ = r3;
    *r4_ = r4;
}

void KerrGeoMinoFrequencies(double *CapitalGamma_, double *CapitalUpsilonPhi_, double *CapitalUpsilonTheta_, double *CapitalUpsilonr_,
                            double a, double p, double e, double x)
{
    double M = 1.0;

    double En = KerrGeoEnergy(a, p, e, x);
    double L = KerrGeoAngularMomentum(a, p, e, x, En);
    double Q = KerrGeoCarterConstant(a, p, e, x, En, L);

    // get radial roots
    double r1, r2, r3, r4;
    KerrGeoRadialRoots(&r1, &r2, &r3, &r4, a, p, e, x, En, Q);

    double Epsilon0 = pow(a, 2) * (1 - pow(En, 2)) / pow(L, 2);
    double zm = 1 - pow(x, 2);
    double a2zp = (pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)) / ((-1 + pow(En, 2)) * (-1 + zm));

    double Epsilon0zp = -((pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)) / (pow(L, 2) * (-1 + zm)));

    double zmOverZp = zm / ((pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)) / (pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)));

    double kr = sqrt((r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4));                                                //(*Eq.(13)*)
    double kTheta = sqrt(zmOverZp);                                                                                 //(*Eq.(13)*)
    double CapitalUpsilonr = (M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) / (2 * EllipticK(pow(kr, 2))); //(*Eq.(15)*)
    double CapitalUpsilonTheta = (M_PI * L * sqrt(Epsilon0zp)) / (2 * EllipticK(pow(kTheta, 2)));                   //(*Eq.(15)*)

    double rp = M + sqrt(pow(M, 2) - pow(a, 2));
    double rm = M - sqrt(pow(M, 2) - pow(a, 2));

    double hr = (r1 - r2) / (r1 - r3);
    double hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm));

    // (*Eq. (21)*)
    double CapitalUpsilonPhi = (2 * CapitalUpsilonTheta) / (M_PI * sqrt(Epsilon0zp)) * EllipticPi(zm, pow(kTheta, 2)) + (2 * a * CapitalUpsilonr) / (M_PI * (rp - rm) * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * ((2 * M * En * rp - a * L) / (r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rp) * EllipticPi(hp, pow(kr, 2))) - (2 * M * En * rm - a * L) / (r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rm) * EllipticPi(hm, pow(kr, 2))));

    double CapitalGamma = 4 * pow(M, 2) * En + (2 * a2zp * En * CapitalUpsilonTheta) / (M_PI * L * sqrt(Epsilon0zp)) * (EllipticK(pow(kTheta, 2)) - EllipticE(pow(kTheta, 2))) + (2 * CapitalUpsilonr) / (M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * (En / 2 * ((r3 * (r1 + r2 + r3) - r1 * r2) * EllipticK(pow(kr, 2)) + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipticPi(hr, pow(kr, 2)) + (r1 - r3) * (r2 - r4) * EllipticE(pow(kr, 2))) + 2 * M * En * (r3 * EllipticK(pow(kr, 2)) + (r2 - r3) * EllipticPi(hr, pow(kr, 2))) + (2 * M) / (rp - rm) * (((4 * pow(M, 2) * En - a * L) * rp - 2 * M * pow(a, 2) * En) / (r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rp) * EllipticPi(hp, pow(kr, 2))) - ((4 * pow(M, 2) * En - a * L) * rm - 2 * M * pow(a, 2) * En) / (r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rm) * EllipticPi(hm, pow(kr, 2)))));

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = CapitalUpsilonTheta;
    *CapitalUpsilonr_ = CapitalUpsilonr;
}

void KerrCircularMinoFrequencies(double *CapitalGamma_, double *CapitalUpsilonPhi_, double *CapitalUpsilonTheta_, double *CapitalUpsilonr_,
                                 double a, double p, double e, double x)
{
    double CapitalUpsilonr = sqrt((p * (-2 * pow(a, 2) + 6 * a * sqrt(p) + (-5 + p) * p + (pow(a - sqrt(p), 2) * (pow(a, 2) - 4 * a * sqrt(p) - (-4 + p) * p)) / abs(pow(a, 2) - 4 * a * sqrt(p) - (-4 + p) * p))) / (2 * a * sqrt(p) + (-3 + p) * p));
    double CapitalUpsilonTheta = abs((pow(p, 0.25) * sqrt(3 * pow(a, 2) - 4 * a * sqrt(p) + pow(p, 2))) / sqrt(2 * a + (-3 + p) * sqrt(p)));
    double CapitalUpsilonPhi = pow(p, 1.25) / sqrt(2 * a + (-3 + p) * sqrt(p));
    double CapitalGamma = (pow(p, 1.25) * (a + pow(p, 1.5))) / sqrt(2 * a + (-3 + p) * sqrt(p));

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = CapitalUpsilonTheta;
    *CapitalUpsilonr_ = CapitalUpsilonr;
}

void KerrGeoCoordinateFrequencies(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                  double a, double p, double e, double x)
{
    // printf("here p e %f %f %f %f \n", a, p, e, x);
    double CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR;

    KerrGeoMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
                           a, p, e, x);

    if ((CapitalUpsilonPhi != CapitalUpsilonPhi) || (CapitalGamma != CapitalGamma) || (CapitalUpsilonR != CapitalUpsilonR))
    {
        printf("(a, p, e, x) = (%f , %f , %f , %f) \n", a, p, e, x);
        throw std::invalid_argument("Nan in fundamental frequencies");
    }
    // printf("here xhi %f %f\n", CapitalUpsilonPhi, CapitalGamma);
    *OmegaPhi_ = CapitalUpsilonPhi / CapitalGamma;
    *OmegaTheta_ = CapitalUpsilonTheta / CapitalGamma;
    *OmegaR_ = CapitalUpsilonR / CapitalGamma;
}

void KerrGeoEquatorialMinoFrequencies(double *CapitalGamma_, double *CapitalUpsilonPhi_, double *CapitalUpsilonTheta_, double *CapitalUpsilonr_,
                                      double a, double p, double e, double x)
{
    double M = 1.0;

    double En = KerrGeoEnergy(a, p, e, x);
    double L = KerrGeoAngularMomentum(a, p, e, x, En);
    double Q = KerrGeoCarterConstant(a, p, e, x, En, L);

    // get radial roots
    double r1, r2, r3, r4;
    KerrGeoRadialRoots(&r1, &r2, &r3, &r4, a, p, e, x, En, Q);

    double Epsilon0 = pow(a, 2) * (1 - pow(En, 2)) / pow(L, 2);
    // double zm = 0;
    double a2zp = (pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1)) / ((-1 + pow(En, 2)) * (-1));

    double Epsilon0zp = -((pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1)) / (pow(L, 2) * (-1)));

    double zp = pow(a, 2) * (1 - pow(En, 2)) + pow(L, 2);

    double arg_kr = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4);

    // double kr = sqrt(arg_kr); //(*Eq.(13)*)
    // double kTheta = 0; //(*Eq.(13)*)
    double kr2 = abs(arg_kr);

    if (kr2>1.0){
        printf("kr %e %e \n", arg_kr, (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4));
        printf("r1 r2 r3 r4 %e %e %e %e\n", r1, r2, r3, r4);
        printf("a p e %e %e %e\n", a,p,e);
    }
    double CapitalUpsilonr = (M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2))) / (2 * EllipticK(kr2)); //(*Eq.(15)*)
    double CapitalUpsilonTheta = x * pow(zp, 0.5);                                                             //(*Eq.(15)*)

    double rp = M + sqrt(pow(M, 2) - pow(a, 2));
    double rm = M - sqrt(pow(M, 2) - pow(a, 2));

    // this check was introduced to avoid round off errors
    // if (r3 - rp==0.0){
    // printf("round off error %e %e %e\n", r3 - rp, L, 2*rp*En/a);
    // }

    double hr = (r1 - r2) / (r1 - r3);
    double hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm));

    // (*Eq. (21)*)
    // This term is zero when r3 - rp == 0.0
    double prob1 = (2 * M * En * rp - a * L) * (EllipticK(kr2) - (r2 - r3) / (r2 - rp) * EllipticPi(hp, kr2));
    if (abs(prob1) != 0.0)
    {
        prob1 = prob1 / (r3 - rp);
    }
    double CapitalUpsilonPhi = (CapitalUpsilonTheta) / (sqrt(Epsilon0zp)) + (2 * a * CapitalUpsilonr) / (M_PI * (rp - rm) * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * (prob1 - (2 * M * En * rm - a * L) / (r3 - rm) * (EllipticK(kr2) - (r2 - r3) / (r2 - rm) * EllipticPi(hm, kr2)));

    // This term is zero when r3 - rp == 0.0
    double prob2 = ((4 * pow(M, 2) * En - a * L) * rp - 2 * M * pow(a, 2) * En) * (EllipticK(kr2) - (r2 - r3) / (r2 - rp) * EllipticPi(hp, kr2));
    if (abs(prob2) != 0.0)
    {
        prob2 = prob2 / (r3 - rp);
    }
    double CapitalGamma = 4 * pow(M, 2) * En + (2 * CapitalUpsilonr) / (M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * (En / 2 * ((r3 * (r1 + r2 + r3) - r1 * r2) * EllipticK(kr2) + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipticPi(hr, kr2) + (r1 - r3) * (r2 - r4) * EllipticE(kr2)) + 2 * M * En * (r3 * EllipticK(kr2) + (r2 - r3) * EllipticPi(hr, kr2)) + (2 * M) / (rp - rm) * (prob2 - ((4 * pow(M, 2) * En - a * L) * rm - 2 * M * pow(a, 2) * En) / (r3 - rm) * (EllipticK(kr2) - (r2 - r3) / (r2 - rm) * EllipticPi(hm, kr2))));

    // This check makes sure that the problematic terms are zero when r3-rp is zero
    // if (r3 - rp==0.0){
    // printf("prob %e %e\n", prob1, prob2);
    // diff_r3_rp = 1e10;
    // }

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = abs(CapitalUpsilonTheta);
    *CapitalUpsilonr_ = CapitalUpsilonr;
}

void KerrGeoEquatorialCoordinateFrequencies(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                            double a, double p, double e, double x)
{
    double CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR;

    // printf("(a, p, e, x) = (%f , %f , %f , %f) \n", a, p, e, x);
    // if (e=0.0){
    //     KerrCircularMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
    //                               a, p, e, x);
    // }
    // else{
    KerrGeoEquatorialMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
                                     a, p, e, x);
    // }

    *OmegaPhi_ = CapitalUpsilonPhi / CapitalGamma;
    *OmegaTheta_ = CapitalUpsilonTheta / CapitalGamma;
    *OmegaR_ = CapitalUpsilonR / CapitalGamma;
}

void SchwarzschildGeoCoordinateFrequencies(double *OmegaPhi, double *OmegaR, double p, double e)
{
    // Need to evaluate 4 different elliptic integrals here. Cache them first to avoid repeated calls.
    // cout << "TEMPTEMP " << p << " " << e << endl;
    double EllipE = EllipticE(4 * e / (p - 6.0 + 2 * e));
    double EllipK = EllipticK(4 * e / (p - 6.0 + 2 * e));
    ;
    double EllipPi1 = EllipticPi(16 * e / (12.0 + 8 * e - 4 * e * e - 8 * p + p * p), 4 * e / (p - 6.0 + 2 * e));
    double EllipPi2 = EllipticPi(2 * e * (p - 4) / ((1.0 + e) * (p - 6.0 + 2 * e)), 4 * e / (p - 6.0 + 2 * e));

    *OmegaPhi = (2 * Power(p, 1.5)) / (Sqrt(-4 * Power(e, 2) + Power(-2 + p, 2)) * (8 + ((-2 * EllipPi2 * (6 + 2 * e - p) * (3 + Power(e, 2) - p) * Power(p, 2)) / ((-1 + e) * Power(1 + e, 2)) - (EllipE * (-4 + p) * Power(p, 2) * (-6 + 2 * e + p)) / (-1 + Power(e, 2)) +
                                                                                         (EllipK * Power(p, 2) * (28 + 4 * Power(e, 2) - 12 * p + Power(p, 2))) / (-1 + Power(e, 2)) + (4 * (-4 + p) * p * (2 * (1 + e) * EllipK + EllipPi2 * (-6 - 2 * e + p))) / (1 + e) + 2 * Power(-4 + p, 2) * (EllipK * (-4 + p) + (EllipPi1 * p * (-6 - 2 * e + p)) / (2 + 2 * e - p))) /
                                                                                            (EllipK * Power(-4 + p, 2))));

    *OmegaR = (p * Sqrt((-6 + 2 * e + p) / (-4 * Power(e, 2) + Power(-2 + p, 2))) * Pi) /
              (8 * EllipK + ((-2 * EllipPi2 * (6 + 2 * e - p) * (3 + Power(e, 2) - p) * Power(p, 2)) / ((-1 + e) * Power(1 + e, 2)) - (EllipE * (-4 + p) * Power(p, 2) * (-6 + 2 * e + p)) / (-1 + Power(e, 2)) +
                             (EllipK * Power(p, 2) * (28 + 4 * Power(e, 2) - 12 * p + Power(p, 2))) / (-1 + Power(e, 2)) + (4 * (-4 + p) * p * (2 * (1 + e) * EllipK + EllipPi2 * (-6 - 2 * e + p))) / (1 + e) + 2 * Power(-4 + p, 2) * (EllipK * (-4 + p) + (EllipPi1 * p * (-6 - 2 * e + p)) / (2 + 2 * e - p))) /
                                Power(-4 + p, 2));
}

void KerrGeoCoordinateFrequenciesVectorized(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                            double *a, double *p, double *e, double *x, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        if (a[i] != 0.0)
        {
            if (abs(x[i]) != 1.)
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

double periodic_acos(double x) {
    // Ensure x is within the range [-1, 1]
    x = fmod(x, 2.0);
    if (x < -1.0)
        x += 2.0;
    else if (x > 1.0)
        x -= 2.0;

    return acos(x);
}

void solveCubic(double A2, double A1, double A0,double *rp, double *ra, double *r3) {
    // Coefficients
    double a = 1.; // coefficient of r^3
    double b = A2; // coefficient of r^2
    double c = A1; // coefficient of r^1
    double d = A0; // coefficient of r^0
    
    // Calculate p and q
    double p = (3.*a*c - b*b) / (3.*a*a);
    double q = (2.*b*b*b - 9.*a*b*c + 27.*a*a*d) / (27.*a*a*a);

    // Calculate discriminant
    double discriminant = q*q/4. + p*p*p/27.;

    if (discriminant >= 0) {
        // One real root and two complex conjugate roots
        double u = cbrt(-q/2. + sqrt(discriminant));
        double v = cbrt(-q/2. - sqrt(discriminant));
        double root = u + v - b/(3.*a);
        // cout << "Real Root: " << root << endl;

        complex<double> imaginaryPart(-sqrt(3.0) / 2.0 * (u - v), 0.5 * (u + v));
        complex<double> root2 = -0.5 * (u + v) - b / (3. * a) + imaginaryPart;
        complex<double> root3 = -0.5 * (u + v) - b / (3. * a) - imaginaryPart;
        // cout << "Complex Root 1: " << root2 << endl;
        // cout << "Complex Root 2: " << root3 << endl;
        *ra = -0.5 * (u + v) - b / (3. * a);
        *rp = -0.5 * (u + v) - b / (3. * a);
        *r3 = root;
    // } else if (discriminant == 0) {
    //     // All roots are real and at least two are equal
    //     double u = cbrt(-q/2.);
    //     double v = cbrt(-q/2.);
    //     double root = u + v - b/(3.*a);
    //     // cout << "Real Root: " << root << endl;
    //     // cout << "Real Root (equal to above): " << root << endl;
    //     // complex<double> root2 = -0.5 * (u + v) - b / (3 * a);
    //     // cout << "Complex Root: " << root2 << endl;
    //     *ra = -0.5 * (u + v) - b / (3. * a);
    //     *rp = -0.5 * (u + v) - b / (3. * a);
    //     *r3 = root;
    } else {
        // All three roots are real and different
        double r = sqrt(-p/3.);
        double theta = acos(-q/(2.*r*r*r));
        double root1 = 2. * r * cos(theta/3.) - b / (3. * a);
        double root2 = 2. * r * cos((theta + 2.*M_PI) / 3.) - b / (3. * a);
        double root3 = 2. * r * cos((theta - 2.*M_PI) / 3.) - b / (3. * a);
        // ra = -2.*rtQnr*cos((theta + 2.*M_PI)/3.) - A2/3.;
        // rp = -2.*rtQnr*cos((theta - 2.*M_PI)/3.) - A2/3.;
        *ra = root1;
        *rp = root3;
        *r3 = root2;
    }
    // cout << "ra: " << *ra << endl;
    // cout << "rp: " << *rp << endl;
    // cout << "r3: " << *r3 << endl;
}

void ELQ_to_pex(double *p, double *e, double *xI, double a, double E, double Lz, double Q)
//
// pexI_of_aELzQ.cc: implements the mapping from orbit integrals
// (E, Lz, Q) to orbit geometry (p, e, xI).  Also provides the
// roots r3 and r4 of the Kerr radial geodesic function.
//
// Scott A. Hughes (sahughes@mit.edu); code extracted from Gremlin
// and converted to standalone form 13 Jan 2024.
//
{
  if (Q < 1.e-14) { // equatorial
    
    double E2m1 = E*E - 1.;//(E - 1.)*(E + 1.);
    double A2 = 2./E2m1;
    double A1 = a*a - Lz*Lz/E2m1;//(a*a*E2m1 - Lz*Lz)/E2m1;
    double A0 = 2.*(a*E - Lz)*(a*E - Lz)/E2m1;
    double rp,ra,r3;
    solveCubic(A2,A1,A0,&rp,&ra,&r3);
    //
    // double Qnr = (A2*A2 - 3.*A1)/9.;
    // double rtQnr = sqrt(Qnr);
    // double Rnr = (A2*(2.*A2*A2 - 9.*A1) + 27.*A0)/54.;
    // double argacos = Rnr/(rtQnr*rtQnr*rtQnr);
    // double theta = acos(argacos);
    // ra = -2.*rtQnr*cos((theta + 2.*M_PI)/3.) - A2/3.;
    // rp = -2.*rtQnr*cos((theta - 2.*M_PI)/3.) - A2/3.;
    // cout << "Scott ra: " << ra << endl;
    // cout << "Scott rp: " << rp << endl;

    *p = 2.*ra*rp/(ra + rp);
    *e = (ra - rp)/(ra + rp);
    // cout << " p: " << *p << endl;
    // cout << " e: " << *e << endl;
    
    // r3 = -2.*rtQnr*cos(theta/3.) - A2/3.;
    // r4 = 0.;
    //
    
    // if (isnan(*p)||isnan(*e)){
    //     cout << "beginning" << " E =" << E  << "\t" << "L=" <<  Lz << "\t" << "Q=" << Q << endl;
    //     cout << "beginning" << " a =" << a  << "\t" << "p=" <<  *p << "\t" << "e=" << *e << "\t" <<  "arg of acos=" <<Rnr/(rtQnr*rtQnr*rtQnr) << endl;
    //     throw std::exception();
    // }

    if (Lz > 0.) *xI = 1.;
    else *xI = -1.;
  } else { // non-equatorial
    double a2 = a*a;
    double E2m1= (E - 1)*(E + 1.);
    double aEmLz = a*E - Lz;
    //
    // The quartic: r^4 + A3 r^3 + A2 r^2 + A1 r + A0 == 0.
    // Kerr radial function divided by E^2 - 1.
    //
    double A0 = -a2*Q/E2m1;
    double A1 = 2.*(Q + aEmLz*aEmLz)/E2m1;
    double A2 = (a2*E2m1 - Lz*Lz - Q)/E2m1;
    double A3 = 2./E2m1;
    //
    // Definitions following Wolters (https://quarticequations.com)
    //
    double B0 = A0 + A3*(-0.25*A1 + A3*(0.0625*A2 - 0.01171875*A3*A3));
    double B1 = A1 + A3*(-0.5*A2 + 0.125*A3*A3);
    double B2 = A2 - 0.375*A3*A3;
    //
    // Definitions needed for the resolvent cubic: z^3 + C2 z^2 + C1 z + C0 == 0;
    //
    double C0 = -0.015625*B1*B1;
    double C1 = 0.0625*B2*B2 - 0.25*B0;
    double C2 = 0.5*B2;
    //
    double rtQnr = sqrt(C2*C2/9. - C1/3.);
    double Rnr = C2*(C2*C2/27. - C1/6.) + C0/2.;
    double theta = acos(Rnr/(rtQnr*rtQnr*rtQnr));
    //
    // zN = cubic zero N
    //
    double rtz1 = sqrt(-2.*rtQnr*cos((theta + 2.*M_PI)/3.) - C2/3.);
    double z2 = -2.*rtQnr*cos((theta - 2.*M_PI)/3.) - C2/3.;
    double z3 = -2.*rtQnr*cos(theta/3.) - C2/3.;
    double rtz2z3 = sqrt(z2*z3);
    //
    // Now assemble the roots of the quartic.  Note that M/(2(1 - E^2)) = -0.25*A3.
    //
    double sgnB1 = (B1 > 0 ? 1. : -1.);
    double rttermmin = sqrt(z2 + z3 - 2.*sgnB1*rtz2z3);
    double rttermplus = sqrt(z2 + z3 + 2.*sgnB1*rtz2z3);
    double ra = -0.25*A3 + rtz1 + rttermmin;
    double rp = -0.25*A3 + rtz1 - rttermmin;
    // r3 = -0.25*A3 - rtz1 + rttermplus;
    // r4 = -0.25*A3 - rtz1 - rttermplus;
    //
    *p = 2.*ra*rp/(ra + rp);
    *e = (ra - rp)/(ra + rp);
    //
    // Note that omE2 = 1 - E^2 = -E2m1 = -(E^2 - 1)
    //
    double QpLz2ma2omE2 = Q + Lz*Lz + a2*E2m1;
    double denomsqr = QpLz2ma2omE2 + sqrt(QpLz2ma2omE2*QpLz2ma2omE2 - 4.*Lz*Lz*a2*E2m1);
    *xI = sqrt(2.)*Lz/sqrt(denomsqr);
  }
    
}

void ELQ_to_pexVectorised(double *p, double *e, double *x, double *a, double *E, double *Lz, double *Q, int length)
{
#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        ELQ_to_pex(&p[i], &e[i], &x[i], a[i], E[i], Lz[i], Q[i]);
    }
}


struct params_holder
{
    double a, p, e, x, Y;
};

double separatrix_polynomial_full(double p, void *params_in)
{

    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (-4 * (3 + e) * Power(p, 11) + Power(p, 12) + Power(a, 12) * Power(-1 + e, 4) * Power(1 + e, 8) * Power(-1 + x, 4) * Power(1 + x, 4) - 4 * Power(a, 10) * (-3 + e) * Power(-1 + e, 3) * Power(1 + e, 7) * p * Power(-1 + Power(x, 2), 4) - 4 * Power(a, 8) * (-1 + e) * Power(1 + e, 5) * Power(p, 3) * Power(-1 + x, 3) * Power(1 + x, 3) * (7 - 7 * Power(x, 2) - Power(e, 2) * (-13 + Power(x, 2)) + Power(e, 3) * (-5 + Power(x, 2)) + 7 * e * (-1 + Power(x, 2))) + 8 * Power(a, 6) * (-1 + e) * Power(1 + e, 3) * Power(p, 5) * Power(-1 + Power(x, 2), 2) * (3 + e + 12 * Power(x, 2) + 4 * e * Power(x, 2) + Power(e, 3) * (-5 + 2 * Power(x, 2)) + Power(e, 2) * (1 + 2 * Power(x, 2))) - 8 * Power(a, 4) * Power(1 + e, 2) * Power(p, 7) * (-1 + x) * (1 + x) * (-3 + e + 15 * Power(x, 2) - 5 * e * Power(x, 2) + Power(e, 3) * (-5 + 3 * Power(x, 2)) + Power(e, 2) * (-1 + 3 * Power(x, 2))) + 4 * Power(a, 2) * Power(p, 9) * (-7 - 7 * e + Power(e, 3) * (-5 + 4 * Power(x, 2)) + Power(e, 2) * (-13 + 12 * Power(x, 2))) + 2 * Power(a, 8) * Power(-1 + e, 2) * Power(1 + e, 6) * Power(p, 2) * Power(-1 + Power(x, 2), 3) * (2 * Power(-3 + e, 2) * (-1 + Power(x, 2)) + Power(a, 2) * (Power(e, 2) * (-3 + Power(x, 2)) - 3 * (1 + Power(x, 2)) + 2 * e * (1 + Power(x, 2)))) - 2 * Power(p, 10) * (-2 * Power(3 + e, 2) + Power(a, 2) * (-3 + 6 * Power(x, 2) + Power(e, 2) * (-3 + 2 * Power(x, 2)) + e * (-2 + 4 * Power(x, 2)))) + Power(a, 6) * Power(1 + e, 4) * Power(p, 4) * Power(-1 + Power(x, 2), 2) * (-16 * Power(-1 + e, 2) * (-3 - 2 * e + Power(e, 2)) * (-1 + Power(x, 2)) + Power(a, 2) * (15 + 6 * Power(x, 2) + 9 * Power(x, 4) + Power(e, 2) * (26 + 20 * Power(x, 2) - 2 * Power(x, 4)) + Power(e, 4) * (15 - 10 * Power(x, 2) + Power(x, 4)) + 4 * Power(e, 3) * (-5 - 2 * Power(x, 2) + Power(x, 4)) - 4 * e * (5 + 2 * Power(x, 2) + 3 * Power(x, 4)))) - 4 * Power(a, 4) * Power(1 + e, 2) * Power(p, 6) * (-1 + x) * (1 + x) * (-2 * (11 - 14 * Power(e, 2) + 3 * Power(e, 4)) * (-1 + Power(x, 2)) + Power(a, 2) * (5 - 5 * Power(x, 2) - 9 * Power(x, 4) + 4 * Power(e, 3) * Power(x, 2) * (-2 + Power(x, 2)) + Power(e, 4) * (5 - 5 * Power(x, 2) + Power(x, 4)) + Power(e, 2) * (6 - 6 * Power(x, 2) + 4 * Power(x, 4)))) + Power(a, 2) * Power(p, 8) * (-16 * Power(1 + e, 2) * (-3 + 2 * e + Power(e, 2)) * (-1 + Power(x, 2)) + Power(a, 2) * (15 - 36 * Power(x, 2) + 30 * Power(x, 4) + Power(e, 4) * (15 - 20 * Power(x, 2) + 6 * Power(x, 4)) + 4 * Power(e, 3) * (5 - 12 * Power(x, 2) + 6 * Power(x, 4)) + 4 * e * (5 - 12 * Power(x, 2) + 10 * Power(x, 4)) + Power(e, 2) * (26 - 72 * Power(x, 2) + 44 * Power(x, 4)))));
}

double separatrix_polynomial_polar(double p, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (Power(a, 6) * Power(-1 + e, 2) * Power(1 + e, 4) + Power(p, 5) * (-6 - 2 * e + p) + Power(a, 2) * Power(p, 3) * (-4 * (-1 + e) * Power(1 + e, 2) + (3 + e * (2 + 3 * e)) * p) - Power(a, 4) * Power(1 + e, 2) * p * (6 + 2 * Power(e, 3) + 2 * e * (-1 + p) - 3 * p - 3 * Power(e, 2) * (2 + p)));
}

double separatrix_polynomial_equat(double p, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (Power(a, 4) * Power(-3 - 2 * e + Power(e, 2), 2) + Power(p, 2) * Power(-6 - 2 * e + p, 2) - 2 * Power(a, 2) * (1 + e) * p * (14 + 2 * Power(e, 2) + 3 * p - e * p));
}

double derivative_polynomial_equat(double p, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;
    return -2 * Power(a, 2) * (1 + e) * (14 + 2 * Power(e, 2) - e * p + 6 * p) + 4 * p * (18 + 2 * Power(e, 2) - 3 * e * (-4 + p) - 9 * p + Power(p, 2));
}

void eq_pol_fdf(double p, void *params_in, double *y, double *dy)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;
    *y = (Power(a, 4) * Power(-3 - 2 * e + Power(e, 2), 2) + Power(p, 2) * Power(-6 - 2 * e + p, 2) - 2 * Power(a, 2) * (1 + e) * p * (14 + 2 * Power(e, 2) + 3 * p - e * p));
    *dy = -2 * Power(a, 2) * (1 + e) * (14 + 2 * Power(e, 2) - e * p + 6 * p) + 4 * p * (18 + 2 * Power(e, 2) - 3 * e * (-4 + p) - 9 * p + Power(p, 2));
}

double solver(struct params_holder *params, double (*func)(double, void *), double x_lo, double x_hi)
{
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 1000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 0, r_expected = sqrt(5.0);
    gsl_function F;

    F.function = func;
    F.params = params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    // printf("-----------START------------------- \n");
    // printf("xlo xhi %f %f\n", x_lo, x_hi);
    // double epsrel=0.001;
    double epsrel = 1e-11; // Decreased tolorance

    do
    {
        iter++;
        status = gsl_root_fsolver_iterate(s);
        r = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi, 0.0, epsrel);

        // printf("result %f %f %f \n", r, x_lo, x_hi);
    } while (status == GSL_CONTINUE && iter < max_iter);

    // printf("result %f %f %f \n", r, x_lo, x_hi);
    // printf("stat, iter, GSL_SUCCESS %d %d %d\n", status, iter, GSL_SUCCESS);
    // printf("-----------END------------------- \n");

    if (status != GSL_SUCCESS)
    {
        // warning if it did not converge otherwise throw error
        if (iter == max_iter)
        {
            printf("a, p, e, Y = %e %e %e %e\n", params->a, params->p, params->e, params->Y);
            throw std::invalid_argument("In Utility.cc Brent root solver failed");
            printf("WARNING: Maximum iteration reached in Utility.cc in Brent root solver.\n");
            printf("Result=%f, x_low=%f, x_high=%f \n", r, x_lo, x_hi);
            printf("a, p, e, Y, sep = %f %f %f %f %f\n", params->a, params->p, params->e, params->Y, get_separatrix(params->a, params->e, r));
            
        }
        else
        {
            throw std::invalid_argument("In Utility.cc Brent root solver failed");
        }
    }

    gsl_root_fsolver_free(s);
    return r;
}

double solver_derivative(struct params_holder *params, double x_lo, double x_hi)
{
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 100;
    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    double x0, x = 0.8 * (x_lo + x_hi);
    gsl_function_fdf FDF;

    FDF.f = &separatrix_polynomial_equat;
    FDF.df = &derivative_polynomial_equat;
    FDF.fdf = &eq_pol_fdf;
    FDF.params = params;

    T = gsl_root_fdfsolver_steffenson;
    s = gsl_root_fdfsolver_alloc(T);
    gsl_root_fdfsolver_set(s, &FDF, x);

    do
    {
        iter++;
        status = gsl_root_fdfsolver_iterate(s);
        x0 = x;
        x = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, 0, 1e-7);

    } while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS)
    {
        // warning if it did not converge otherwise throw error
        if (iter == max_iter)
        {
            printf("WARNING: Maximum iteration reached in Utility.cc in Brent root solver.\n");
            printf("Result=%f, x_low=%f, x_high=%f \n", x, x_lo, x_hi);
            printf("a, p, e, Y, sep = %f %f %f %f \n", params->a, params->p, params->e, params->Y);
        }
        else
        {
            throw std::invalid_argument("In Utility.cc Brent root solver failed");
        }
    }

    gsl_root_fdfsolver_free(s);

    return x;
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
    else if ((e < 0.0) & (abs(x) == 1.0))
    {
        z1 = 1. + pow((1. - pow(a, 2)), 1. / 3.) * (pow((1. + a), 1. / 3.) + pow((1. - a), 1. / 3.));

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
        struct params_holder params = {a, 0.0, e, x, x};
        double x_lo, x_hi;

        x_lo = 1.0 + e;
        x_hi = 6 + 2. * e;

        p_sep = solver(&params, &separatrix_polynomial_equat, x_lo, x_hi); // separatrix_KerrEquatorial(a, e);//
        return p_sep;
    }
    else if (x == -1.0) // Eccentric Retrograde Equatorial
    {
        // fills in p and Y with zeros
        struct params_holder params = {a, 0.0, e, x, x};
        double x_lo, x_hi;

        x_lo = 6 + 2. * e;
        x_hi = 5 + e + 4 * Sqrt(1 + e);

        p_sep = solver(&params, &separatrix_polynomial_equat, x_lo, x_hi);
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

        double polar_p_sep = solver(&params, &separatrix_polynomial_polar, x_lo, x_hi);
        if (x == 0.0)
            return polar_p_sep;

        double equat_p_sep;
        if (x > 0.0)
        {
            x_lo = 1.0 + e;
            x_hi = 6 + 2. * e;

            equat_p_sep = solver(&params, &separatrix_polynomial_equat, x_lo, x_hi);

            x_lo = equat_p_sep;
            x_hi = polar_p_sep;
        }
        else
        {
            x_lo = polar_p_sep;
            x_hi = 12.0;
        }

        p_sep = solver(&params, &separatrix_polynomial_full, x_lo, x_hi);

        return p_sep;
    }
}

void get_separatrix_vector(double *separatrix, double *a, double *e, double *x, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        separatrix[i] = get_separatrix(a[i], e[i], x[i]);
    }
}

double Y_to_xI_eq(double x, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

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
    if (abs(Y) > YLIM)
        return Y;
    // fills in x with 0.0
    struct params_holder params = {a, p, e, 0.0, Y};
    double x_lo, x_hi;

    // set limits
    // assume Y is close to x
    x_lo = Y - 0.15;
    x_hi = Y + 0.15;

    x_lo = x_lo > -YLIM ? x_lo : -YLIM;
    x_hi = x_hi < YLIM ? x_hi : YLIM;

    double x = solver(&params, &Y_to_xI_eq, x_lo, x_hi);

    return x;
}

void Y_to_xI_vector(double *x, double *a, double *p, double *e, double *Y, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        x[i] = Y_to_xI(a[i], p[i], e[i], Y[i]);
    }
}

void set_threads(int num_threads)
{
#ifdef __USE_OMP__
    omp_set_num_threads(num_threads);
#else
    throw std::invalid_argument("Attempting to set threads for openMP, but FEW was not installed with openMP due to the use of the flag --no_omp used during installation.");
#endif
}

int get_threads()
{
#ifdef __USE_OMP__
    int num_threads;
#pragma omp parallel for
    for (int i = 0; i < 1; i += 1)
    {
        num_threads = omp_get_num_threads();
    }

    return num_threads;
#else
    return 0;
#endif // __USE_OMP__
}

double separatrix_KerrEquatorial(const double a, const double e)
{
    double result;
    double Compile_$2 = -0.875 + a;
    double Compile_$3 = pow(Compile_$2, 2);
    double Compile_$9 = pow(Compile_$2, 3);
    double Compile_$15 = pow(Compile_$2, 4);
    double Compile_$21 = pow(Compile_$2, 5);
    double Compile_$27 = pow(Compile_$2, 6);
    double Compile_$38 = pow(Compile_$2, 7);
    double Compile_$45 = pow(Compile_$2, 8);
    double Compile_$55 = pow(Compile_$2, 9);
    double Compile_$63 = pow(Compile_$2, 10);
    double Compile_$5 = 3200. * Compile_$3;
    double Compile_$6 = -1. + Compile_$5;
    double Compile_$92 = -0.4 + e;
    double Compile_$93 = pow(Compile_$92, 2);
    double Compile_$94 = 12.5 * Compile_$93;
    double Compile_$95 = -1. + Compile_$94;
    double Compile_$8 = -120. * Compile_$2;
    double Compile_$10 = 256000. * Compile_$9;
    double Compile_$11 = Compile_$10 + Compile_$8;
    double Compile_$14 = -12800. * Compile_$3;
    double Compile_$16 = 2.048e7 * Compile_$15;
    double Compile_$17 = 1. + Compile_$14 + Compile_$16;
    double Compile_$19 = 200. * Compile_$2;
    double Compile_$20 = -1.28e6 * Compile_$9;
    double Compile_$22 = 1.6384e9 * Compile_$21;
    double Compile_$23 = Compile_$19 + Compile_$20 + Compile_$22;
    double Compile_$25 = 28800. * Compile_$3;
    double Compile_$26 = -1.2288e8 * Compile_$15;
    double Compile_$28 = 1.31072e11 * Compile_$27;
    double Compile_$29 = -1. + Compile_$25 + Compile_$26 + Compile_$28;
    double Compile_$31 = -280. * Compile_$2;
    double Compile_$34 = 3.584e6 * Compile_$9;
    double Compile_$35 = -1.14688e10 * Compile_$21;
    double Compile_$39 = 1.048576e13 * Compile_$38;
    double Compile_$40 = Compile_$31 + Compile_$34 + Compile_$35 + Compile_$39;
    double Compile_$42 = -51200. * Compile_$3;
    double Compile_$43 = 4.096e8 * Compile_$15;
    double Compile_$44 = -1.048576e12 * Compile_$27;
    double Compile_$46 = 8.388608e14 * Compile_$45;
    double Compile_$47 = 1. + Compile_$42 + Compile_$43 + Compile_$44 + Compile_$46;
    double Compile_$51 = 360. * Compile_$2;
    double Compile_$52 = -7.68e6 * Compile_$9;
    double Compile_$53 = 4.42368e10 * Compile_$21;
    double Compile_$54 = -9.437184e13 * Compile_$38;
    double Compile_$56 = 6.7108864e16 * Compile_$55;
    double Compile_$57 = Compile_$51 + Compile_$52 + Compile_$53 + Compile_$54 + Compile_$56;
    double Compile_$59 = 80000. * Compile_$3;
    double Compile_$60 = -1.024e9 * Compile_$15;
    double Compile_$61 = 4.58752e12 * Compile_$27;
    double Compile_$62 = -8.388608e15 * Compile_$45;
    double Compile_$64 = 5.36870912e18 * Compile_$63;
    double Compile_$65 = -1. + Compile_$59 + Compile_$60 + Compile_$61 + Compile_$62 + Compile_$64;
    double Compile_$67 = -440. * Compile_$2;
    double Compile_$70 = 1.408e7 * Compile_$9;
    double Compile_$71 = -1.261568e11 * Compile_$21;
    double Compile_$74 = 4.6137344e14 * Compile_$38;
    double Compile_$75 = -7.38197504e17 * Compile_$55;
    double Compile_$78 = pow(Compile_$2, 11);
    double Compile_$79 = 4.294967296e20 * Compile_$78;
    double Compile_$80 = Compile_$67 + Compile_$70 + Compile_$71 + Compile_$74 + Compile_$75 + Compile_$79;
    double Compile_$82 = -115200. * Compile_$3;
    double Compile_$83 = 2.1504e9 * Compile_$15;
    double Compile_$84 = -1.4680064e13 * Compile_$27;
    double Compile_$85 = 4.52984832e16 * Compile_$45;
    double Compile_$86 = -6.442450944e19 * Compile_$63;
    double Compile_$87 = pow(Compile_$2, 12);
    double Compile_$88 = 3.4359738368e22 * Compile_$87;
    double Compile_$89 = 1. + Compile_$82 + Compile_$83 + Compile_$84 + Compile_$85 + Compile_$86 + Compile_$88;
    double Compile_$109 = -7.5 * Compile_$92;
    double Compile_$110 = pow(Compile_$92, 3);
    double Compile_$111 = 62.5 * Compile_$110;
    double Compile_$112 = Compile_$109 + Compile_$111;
    double Compile_$126 = -50. * Compile_$93;
    double Compile_$127 = pow(Compile_$92, 4);
    double Compile_$128 = 312.5 * Compile_$127;
    double Compile_$129 = 1. + Compile_$126 + Compile_$128;
    double Compile_$143 = 12.5 * Compile_$92;
    double Compile_$144 = -312.5 * Compile_$110;
    double Compile_$145 = pow(Compile_$92, 5);
    double Compile_$146 = 1562.5 * Compile_$145;
    double Compile_$147 = Compile_$143 + Compile_$144 + Compile_$146;
    double Compile_$161 = 112.5 * Compile_$93;
    double Compile_$162 = -1875. * Compile_$127;
    double Compile_$163 = pow(Compile_$92, 6);
    double Compile_$164 = 7812.5 * Compile_$163;
    double Compile_$165 = -1. + Compile_$161 + Compile_$162 + Compile_$164;
    double Compile_$179 = -17.5 * Compile_$92;
    double Compile_$180 = 875. * Compile_$110;
    double Compile_$181 = -10937.5 * Compile_$145;
    double Compile_$182 = pow(Compile_$92, 7);
    double Compile_$183 = 39062.5 * Compile_$182;
    double Compile_$184 = Compile_$179 + Compile_$180 + Compile_$181 + Compile_$183;
    double Compile_$198 = -200. * Compile_$93;
    double Compile_$199 = 6250. * Compile_$127;
    double Compile_$200 = -62500. * Compile_$163;
    double Compile_$201 = pow(Compile_$92, 8);
    double Compile_$202 = 195312.5 * Compile_$201;
    double Compile_$203 = 1. + Compile_$198 + Compile_$199 + Compile_$200 + Compile_$202;
    double Compile_$217 = 22.5 * Compile_$92;
    double Compile_$218 = -1875. * Compile_$110;
    double Compile_$219 = 42187.5 * Compile_$145;
    double Compile_$220 = -351562.5 * Compile_$182;
    double Compile_$221 = pow(Compile_$92, 9);
    double Compile_$222 = 976562.5 * Compile_$221;
    double Compile_$223 = Compile_$217 + Compile_$218 + Compile_$219 + Compile_$220 + Compile_$222;
    double Compile_$237 = 312.5 * Compile_$93;
    double Compile_$238 = -15625. * Compile_$127;
    double Compile_$239 = 273437.5 * Compile_$163;
    double Compile_$240 = -1.953125e6 * Compile_$201;
    double Compile_$241 = pow(Compile_$92, 10);
    double Compile_$242 = 4.8828125e6 * Compile_$241;
    double Compile_$243 = -1. + Compile_$237 + Compile_$238 + Compile_$239 + Compile_$240 + Compile_$242;
    double Compile_$257 = -27.5 * Compile_$92;
    double Compile_$258 = 3437.5 * Compile_$110;
    double Compile_$259 = -120312.5 * Compile_$145;
    double Compile_$260 = 1.71875e6 * Compile_$182;
    double Compile_$261 = -1.07421875e7 * Compile_$221;
    double Compile_$262 = pow(Compile_$92, 11);
    double Compile_$263 = 2.44140625e7 * Compile_$262;
    double Compile_$264 = Compile_$257 + Compile_$258 + Compile_$259 + Compile_$260 + Compile_$261 + Compile_$263;
    double Compile_$278 = -450. * Compile_$93;
    double Compile_$279 = 32812.5 * Compile_$127;
    double Compile_$280 = -875000. * Compile_$163;
    double Compile_$281 = 1.0546875e7 * Compile_$201;
    double Compile_$282 = -5.859375e7 * Compile_$241;
    double Compile_$283 = pow(Compile_$92, 12);
    double Compile_$284 = 1.220703125e8 * Compile_$283;
    double Compile_$285 = 1. + Compile_$278 + Compile_$279 + Compile_$280 + Compile_$281 + Compile_$282 + Compile_$284;
    double Compile_$299 = 32.5 * Compile_$92;
    double Compile_$300 = -5687.5 * Compile_$110;
    double Compile_$301 = 284375. * Compile_$145;
    double Compile_$302 = -6.09375e6 * Compile_$182;
    double Compile_$303 = 6.34765625e7 * Compile_$221;
    double Compile_$304 = -3.173828125e8 * Compile_$262;
    double Compile_$305 = pow(Compile_$92, 13);
    double Compile_$306 = 6.103515625e8 * Compile_$305;
    double Compile_$307 = Compile_$299 + Compile_$300 + Compile_$301 + Compile_$302 + Compile_$303 + Compile_$304 + Compile_$306;
    double Compile_$321 = 612.5 * Compile_$93;
    double Compile_$322 = -61250. * Compile_$127;
    double Compile_$323 = 2.296875e6 * Compile_$163;
    double Compile_$324 = -4.1015625e7 * Compile_$201;
    double Compile_$325 = 3.759765625e8 * Compile_$241;
    double Compile_$326 = -1.708984375e9 * Compile_$283;
    double Compile_$327 = pow(Compile_$92, 14);
    double Compile_$328 = 3.0517578125e9 * Compile_$327;
    double Compile_$329 = -1. + Compile_$321 + Compile_$322 + Compile_$323 + Compile_$324 + Compile_$325 + Compile_$326 + Compile_$328;
    double Compile_$343 = -37.5 * Compile_$92;
    double Compile_$344 = 8750. * Compile_$110;
    double Compile_$345 = -590625. * Compile_$145;
    double Compile_$346 = 1.7578125e7 * Compile_$182;
    double Compile_$347 = -2.685546875e8 * Compile_$221;
    double Compile_$348 = 2.197265625e9 * Compile_$262;
    double Compile_$349 = -9.1552734375e9 * Compile_$305;
    double Compile_$350 = pow(Compile_$92, 15);
    double Compile_$351 = 1.52587890625e10 * Compile_$350;
    double Compile_$352 = Compile_$343 + Compile_$344 + Compile_$345 + Compile_$346 + Compile_$347 + Compile_$348 + Compile_$349 + Compile_$351;
    double Compile_$366 = -800. * Compile_$93;
    double Compile_$367 = 105000. * Compile_$127;
    double Compile_$368 = -5.25e6 * Compile_$163;
    double Compile_$369 = 1.2890625e8 * Compile_$201;
    double Compile_$370 = -1.71875e9 * Compile_$241;
    double Compile_$371 = 1.26953125e10 * Compile_$283;
    double Compile_$372 = -4.8828125e10 * Compile_$327;
    double Compile_$373 = pow(Compile_$92, 16);
    double Compile_$374 = 7.62939453125e10 * Compile_$373;
    double Compile_$375 = 1. + Compile_$366 + Compile_$367 + Compile_$368 + Compile_$369 + Compile_$370 + Compile_$371 + Compile_$372 + Compile_$374;
    double Compile_$389 = 42.5 * Compile_$92;
    double Compile_$390 = -12750. * Compile_$110;
    double Compile_$391 = 1.115625e6 * Compile_$145;
    double Compile_$392 = -4.3828125e7 * Compile_$182;
    double Compile_$393 = 9.130859375e8 * Compile_$221;
    double Compile_$394 = -1.0791015625e10 * Compile_$262;
    double Compile_$395 = 7.26318359375e10 * Compile_$305;
    double Compile_$396 = -2.593994140625e11 * Compile_$350;
    double Compile_$397 = pow(Compile_$92, 17);
    double Compile_$398 = 3.814697265625e11 * Compile_$397;
    double Compile_$399 = Compile_$389 + Compile_$390 + Compile_$391 + Compile_$392 + Compile_$393 + Compile_$394 + Compile_$395 + Compile_$396 + Compile_$398;
    result = 2.91352319406094061986091651 - 0.00016284618369671501891938 * Compile_$11 - 0.00344098151801864926312442 * Compile_$112 - 8.454686467988030343e-7 * Compile_$11 * Compile_$112 + 0.00045975534530061279176401 * Compile_$129 + 3.522747836810062931e-7 * Compile_$11 * Compile_$129 - 0.00005090800216967482739708 * Compile_$147 - 7.96680630602420129e-8 * Compile_$11 * Compile_$147 + 3.90759688820813564179e-6 * Compile_$165 + 9.9464546613873438e-9 * Compile_$11 * Compile_$165 - 0.00001036956920747591485328 * Compile_$17 - 6.05515515618870803e-8 * Compile_$112 * Compile_$17 + 2.95811040821036276e-8 * Compile_$129 * Compile_$17 - 7.6325749584423864e-9 * Compile_$147 * Compile_$17 + 1.1334766240463451e-9 * Compile_$165 * Compile_$17 + 6.75482658581192536e-9 * Compile_$184 + 4.194186368161293e-10 * Compile_$11 * Compile_$184 + 6.3597769255516e-12 * Compile_$17 * Compile_$184 - 6.836183650071831449198908 * Compile_$2 - 0.011281349177524942504919 * Compile_$112 * Compile_$2 + 0.002439547820661392521926 * Compile_$129 * Compile_$2 - 0.000353272121843407841173 * Compile_$147 * Compile_$2 + 0.000027083682671818370053 * Compile_$165 * Compile_$2 + 2.823622629738801065e-6 * Compile_$184 * Compile_$2 - 7.215425898783023565e-8 * Compile_$203 - 5.952739996652805e-10 * Compile_$11 * Compile_$203 - 6.16649499750823e-11 * Compile_$17 * Compile_$203 - 1.56641183425492541e-6 * Compile_$2 * Compile_$203 + 1.562145286028911668e-8 * Compile_$223 + 1.7698887560237489e-10 * Compile_$11 * Compile_$223 + 2.123360153492348e-11 * Compile_$17 * Compile_$223 + 3.44279714744578915e-7 * Compile_$2 * Compile_$223 - 7.3822347684245341617e-7 * Compile_$23 - 4.5919311457885436e-9 * Compile_$112 * Compile_$23 + 2.5486136701661831e-9 * Compile_$129 * Compile_$23 - 7.288413520527012e-10 * Compile_$147 * Compile_$23 + 1.237400953795063e-10 * Compile_$165 * Compile_$23 - 3.0271966594178e-12 * Compile_$184 * Compile_$23 - 6.0820994338773e-12 * Compile_$203 * Compile_$23 + 2.40921404748626e-12 * Compile_$223 * Compile_$23 - 1.90937539524066598e-9 * Compile_$243 - 3.16610644745792e-11 * Compile_$11 * Compile_$243 - 4.2907763119221e-12 * Compile_$17 * Compile_$243 - 4.6980853020381921e-8 * Compile_$2 * Compile_$243 - 5.426687040614e-13 * Compile_$23 * Compile_$243 + 6.092314040983428e-11 * Compile_$264 + 2.6475141631424e-12 * Compile_$11 * Compile_$264 + 4.446061205118e-13 * Compile_$17 * Compile_$264 + 2.328725998042019e-9 * Compile_$2 * Compile_$264 + 6.68602106062e-14 * Compile_$23 * Compile_$264 + 3.755679334800628e-11 * Compile_$285 + 4.85180811662e-13 * Compile_$11 * Compile_$285 + 5.12583218294e-14 * Compile_$17 * Compile_$285 + 8.32367746170678e-10 * Compile_$2 * Compile_$285 + 4.2859071672e-15 * Compile_$23 * Compile_$285 - 5.624451895467586345e-8 * Compile_$29 - 3.615522469933619e-10 * Compile_$112 * Compile_$29 + 2.237066529670028e-10 * Compile_$129 * Compile_$29 - 6.95599024288007e-11 * Compile_$147 * Compile_$29 + 1.31423226871835e-11 * Compile_$165 * Compile_$29 - 6.543769582531e-13 * Compile_$184 * Compile_$29 - 5.799163645474e-13 * Compile_$203 * Compile_$29 + 2.6330235649347e-13 * Compile_$223 * Compile_$29 - 6.53491058985e-14 * Compile_$243 * Compile_$29 + 9.2764160197e-15 * Compile_$264 * Compile_$29 + 2.249016927e-16 * Compile_$285 * Compile_$29 - 1.180067182287779e-11 * Compile_$307 - 2.647460879526e-13 * Compile_$11 * Compile_$307 - 3.76202707199e-14 * Compile_$17 * Compile_$307 - 3.08643897220732e-10 * Compile_$2 * Compile_$307 - 4.8536712587e-15 * Compile_$23 * Compile_$307 - 5.83776826e-16 * Compile_$29 * Compile_$307 + 2.11679433740239e-12 * Compile_$329 + 6.40218339363e-14 * Compile_$11 * Compile_$329 + 1.01707468927e-14 * Compile_$17 * Compile_$329 + 6.0512123410093e-11 * Compile_$2 * Compile_$329 + 1.4671537978e-15 * Compile_$23 * Compile_$329 + 1.973409169e-16 * Compile_$29 * Compile_$329 - 2.3628336763492e-13 * Compile_$352 - 9.4413005311e-15 * Compile_$11 * Compile_$352 - 1.6758649055e-15 * Compile_$17 * Compile_$352 - 7.188180962268e-12 * Compile_$2 * Compile_$352 - 2.683294083e-16 * Compile_$23 * Compile_$352 - 3.9760166e-17 * Compile_$29 * Compile_$352 + 3.51033130781e-15 * Compile_$375 + 3.895051472e-16 * Compile_$11 * Compile_$375 + 1.020133463e-16 * Compile_$17 * Compile_$375 + 8.0325437025e-14 * Compile_$2 * Compile_$375 + 2.1601097e-17 * Compile_$23 * Compile_$375 + 3.9552016e-18 * Compile_$29 * Compile_$375 + 5.53515655667e-15 * Compile_$399 + 2.368402158e-16 * Compile_$11 * Compile_$399 + 3.59615932e-17 * Compile_$17 * Compile_$399 + 2.00624761053e-13 * Compile_$2 * Compile_$399 + 4.629843e-18 * Compile_$23 * Compile_$399 + 5.069367e-19 * Compile_$29 * Compile_$399 - 4.48566934374887613e-9 * Compile_$40 - 2.92345842669529e-11 * Compile_$112 * Compile_$40 + 1.99174687012693e-11 * Compile_$129 * Compile_$40 - 6.6440448007201e-12 * Compile_$147 * Compile_$40 + 1.3707428416706e-12 * Compile_$165 * Compile_$40 - 9.81798450497e-14 * Compile_$184 * Compile_$40 - 5.38553136e-14 * Compile_$203 * Compile_$40 + 2.802307661463e-14 * Compile_$223 * Compile_$40 - 7.5910209293e-15 * Compile_$243 * Compile_$40 + 1.2128758817e-15 * Compile_$264 * Compile_$40 - 9.7656393e-18 * Compile_$285 * Compile_$40 - 6.6495426e-17 * Compile_$307 * Compile_$40 + 2.51737791e-17 * Compile_$329 * Compile_$40 - 5.5482871e-18 * Compile_$352 * Compile_$40 + 6.525777e-19 * Compile_$375 * Compile_$40 + 4.48739e-20 * Compile_$399 * Compile_$40 - 3.6972136561018969e-10 * Compile_$47 - 2.4110616577454e-12 * Compile_$112 * Compile_$47 + 1.7934539240431e-12 * Compile_$129 * Compile_$47 - 6.355168311345e-13 * Compile_$147 * Compile_$47 + 1.412069842755e-13 * Compile_$165 * Compile_$47 - 1.28229706304e-14 * Compile_$184 * Compile_$47 - 4.8874745419e-15 * Compile_$203 * Compile_$47 + 2.92476132954e-15 * Compile_$223 * Compile_$47 - 8.581034423e-16 * Compile_$243 * Compile_$47 + 1.5165262e-16 * Compile_$264 * Compile_$47 - 5.3281875e-18 * Compile_$285 * Compile_$47 - 7.2426115e-18 * Compile_$307 * Compile_$47 + 3.0809479e-18 * Compile_$329 * Compile_$47 - 7.38143e-19 * Compile_$352 * Compile_$47 + 9.96894e-20 * Compile_$375 * Compile_$47 + 2.4049e-21 * Compile_$399 * Compile_$47 - 3.124047100581606e-11 * Compile_$57 - 2.018883439823e-13 * Compile_$112 * Compile_$57 + 1.629821726888e-13 * Compile_$129 * Compile_$57 - 6.08919144825e-14 * Compile_$147 * Compile_$57 + 1.44199305544e-14 * Compile_$165 * Compile_$57 - 1.5562573913e-15 * Compile_$184 * Compile_$57 - 4.336694924e-16 * Compile_$203 * Compile_$57 + 3.0074864819e-16 * Compile_$223 * Compile_$57 - 9.49802385e-17 * Compile_$243 * Compile_$57 + 1.83209563e-17 * Compile_$264 * Compile_$57 - 1.0714238e-18 * Compile_$285 * Compile_$57 - 7.585738e-19 * Compile_$307 * Compile_$57 + 3.648095e-19 * Compile_$329 * Compile_$57 - 9.44454e-20 * Compile_$352 * Compile_$57 + 1.43867e-20 * Compile_$375 * Compile_$57 - 1.176e-22 * Compile_$399 * Compile_$57 - 0.00318656888430662736679277 * Compile_$6 - 0.000013049320300835375528 * Compile_$112 * Compile_$6 + 4.3614564389187135878e-6 * Compile_$129 * Compile_$6 - 8.250286658015210544e-7 * Compile_$147 * Compile_$6 + 8.17852618284975678e-8 * Compile_$165 * Compile_$6 + 7.022679612923859e-9 * Compile_$184 * Compile_$6 - 5.294518727817022e-9 * Compile_$203 * Compile_$6 + 1.34849498326554545e-9 * Compile_$223 * Compile_$6 - 2.108677853260902e-10 * Compile_$243 * Compile_$6 + 1.36235584964043e-11 * Compile_$264 * Compile_$6 + 3.688732220644e-12 * Compile_$285 * Compile_$6 - 1.6125171774961e-12 * Compile_$307 * Compile_$6 + 3.495943851344e-13 * Compile_$329 * Compile_$6 - 4.6040699803e-14 * Compile_$352 * Compile_$6 + 1.0777724061e-15 * Compile_$375 * Compile_$6 + 1.2695985222e-15 * Compile_$399 * Compile_$6 - 2.69154391416394e-12 * Compile_$65 - 1.71074951826e-14 * Compile_$112 * Compile_$65 + 1.49250002201e-14 * Compile_$129 * Compile_$65 - 5.8447704206e-15 * Compile_$147 * Compile_$65 + 1.4632843663e-15 * Compile_$165 * Compile_$65 - 1.805372286e-16 * Compile_$184 * Compile_$65 - 3.75502146e-17 * Compile_$203 * Compile_$65 + 3.056710268e-17 * Compile_$223 * Compile_$65 - 1.03397205e-17 * Compile_$243 * Compile_$65 + 2.1543832e-18 * Compile_$264 * Compile_$65 - 1.702773e-19 * Compile_$285 * Compile_$65 - 7.6653e-20 * Compile_$307 * Compile_$65 + 4.20781e-20 * Compile_$329 * Compile_$65 - 1.15797e-20 * Compile_$352 * Compile_$65 + 2.0288e-21 * Compile_$375 * Compile_$65 - 1.13e-22 * Compile_$399 * Compile_$65 - 2.3552636742849e-13 * Compile_$80 - 1.4633663483e-15 * Compile_$112 * Compile_$80 + 1.375534921e-15 * Compile_$129 * Compile_$80 - 5.619666536e-16 * Compile_$147 * Compile_$80 + 1.477846906e-16 * Compile_$165 * Compile_$80 - 2.03154943e-17 * Compile_$184 * Compile_$80 - 3.1575075e-18 * Compile_$203 * Compile_$80 + 3.07742986e-18 * Compile_$223 * Compile_$80 - 1.1105829e-18 * Compile_$243 * Compile_$80 + 2.478883e-19 * Compile_$264 * Compile_$80 - 2.41396e-20 * Compile_$285 * Compile_$80 - 7.4715e-21 * Compile_$307 * Compile_$80 + 4.7679e-21 * Compile_$329 * Compile_$80 - 1.3123e-21 * Compile_$352 * Compile_$80 + 3.137e-22 * Compile_$375 * Compile_$80 - 3.96e-23 * Compile_$399 * Compile_$80 - 2.070955450059e-14 * Compile_$89 - 1.251908176e-16 * Compile_$112 * Compile_$89 + 1.263900432e-16 * Compile_$129 * Compile_$89 - 5.36302393e-17 * Compile_$147 * Compile_$89 + 1.472689e-17 * Compile_$165 * Compile_$89 - 2.2118417e-18 * Compile_$184 * Compile_$89 - 2.541341e-19 * Compile_$203 * Compile_$89 + 3.0441636e-19 * Compile_$223 * Compile_$89 - 1.16742e-19 * Compile_$243 * Compile_$89 + 2.77259e-20 * Compile_$264 * Compile_$89 - 3.1536e-21 * Compile_$285 * Compile_$89 - 7.157e-22 * Compile_$307 * Compile_$89 + 5.482e-22 * Compile_$329 * Compile_$89 - 1.005e-22 * Compile_$352 * Compile_$89 + 5.35e-23 * Compile_$375 * Compile_$89 - 1.713e-23 * Compile_$399 * Compile_$89 + 1.1514396684417654056222758 * Compile_$92 - 0.0000205468004819699230606 * Compile_$11 * Compile_$92 - 9.959977615612570796e-7 * Compile_$17 * Compile_$92 - 1.44727270401310843905687 * Compile_$2 * Compile_$92 - 5.2972955270696964e-8 * Compile_$23 * Compile_$92 - 2.886813350228537e-9 * Compile_$29 * Compile_$92 - 1.509603862795322e-10 * Compile_$40 * Compile_$92 - 6.6764891137085e-12 * Compile_$47 * Compile_$92 - 1.276317195595e-13 * Compile_$57 * Compile_$92 - 0.0005297205449556790753947 * Compile_$6 * Compile_$92 + 2.30972109316e-14 * Compile_$65 * Compile_$92 + 4.7508181712e-15 * Compile_$80 * Compile_$92 + 6.359512167e-16 * Compile_$89 * Compile_$92 + 0.02256216328824196672745628 * Compile_$95 - 7.965134025673000616e-7 * Compile_$11 * Compile_$95 - 7.87548456943294908e-8 * Compile_$17 * Compile_$95 + 0.025019571872207194992245 * Compile_$2 * Compile_$95 - 7.4839816170359549e-9 * Compile_$23 * Compile_$95 - 7.040293939633263e-10 * Compile_$29 * Compile_$95 - 6.61481821465013e-11 * Compile_$40 * Compile_$95 - 6.2267155461927e-12 * Compile_$47 * Compile_$95 - 5.878718666362e-13 * Compile_$57 * Compile_$95 - 6.6986500531244527943e-6 * Compile_$6 * Compile_$95 - 5.56821074926e-14 * Compile_$65 * Compile_$95 - 5.2906566763e-15 * Compile_$80 * Compile_$95 - 4.996962178e-16 * Compile_$89 * Compile_$95;
    return result;
}

// Secondary Spin functions

double P(double r, double a, double En, double xi)
{
    return En * r * r - a * xi;
}

double deltaP(double r, double a, double En, double xi, double deltaEn, double deltaxi)
{
    return deltaEn * r * r - xi / r - a * deltaxi;
}

double deltaRt(double r, double am1, double a0, double a1, double a2)
{
    return am1 / r + a0 + r * (a1 + r * a2);
}

void KerrEqSpinFrequenciesCorrection(double *deltaOmegaR_, double *deltaOmegaPhi_,
                                     double a, double p, double e, double x)
{
    // printf("a, p, e, x, sep = %f %f %f %f\n", a, p, e, x);
    double M = 1.0;
    double En = KerrGeoEnergy(a, p, e, x);
    double xi = KerrGeoAngularMomentum(a, p, e, x, En) - a * En;

    // get radial roots
    double r1, r2, r3, r4;
    KerrGeoRadialRoots(&r1, &r2, &r3, &r4, a, p, e, x, En, 0.);

    double deltaEn, deltaxi;

    deltaEn = (xi * (-(a * pow(En, 2) * pow(r1, 2) * pow(r2, 2)) - En * pow(r1, 2) * pow(r2, 2) * xi + pow(a, 2) * En * (pow(r1, 2) + r1 * r2 + pow(r2, 2)) * xi +
                     a * (pow(r1, 2) + r1 * (-2 + r2) + (-2 + r2) * r2) * pow(xi, 2))) /
              (pow(r1, 2) * pow(r2, 2) * (a * pow(En, 2) * r1 * r2 * (r1 + r2) + En * (pow(r1, 2) * (-2 + r2) + r1 * (-2 + r2) * r2 - 2 * pow(r2, 2)) * xi + 2 * a * pow(xi, 2)));

    deltaxi = ((pow(r1, 2) + r1 * r2 + pow(r2, 2)) * xi * (En * pow(r2, 2) - a * xi) * (-(En * pow(r1, 2)) + a * xi)) /
              (pow(r1, 2) * pow(r2, 2) * (a * pow(En, 2) * r1 * r2 * (r1 + r2) + En * (pow(r1, 2) * (-2 + r2) + r1 * (-2 + r2) * r2 - 2 * pow(r2, 2)) * xi + 2 * a * pow(xi, 2)));

    double am1, a0, a1, a2;
    am1 = (-2 * a * pow(xi, 2)) / (r1 * r2);
    a0 = -2 * En * (-(a * deltaxi) + deltaEn * pow(r1, 2) + deltaEn * r1 * r2 + deltaEn * pow(r2, 2)) + 2 * (a * deltaEn + deltaxi) * xi;
    a1 = -2 * deltaEn * En * (r1 + r2);
    a2 = -2 * deltaEn * En;

    double kr = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4); // convention without the sqrt
    double hr = (r1 - r2) / (r1 - r3);

    double rp = M + sqrt(pow(M, 2) - pow(a, 2));
    double rm = M - sqrt(pow(M, 2) - pow(a, 2));

    double hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm));

    double Kkr = EllipticK(kr);         //(* Elliptic integral of the first kind *)
    double Ekr = EllipticE(kr);         //(* Elliptic integral of the second kind *)
    double Pihrkr = EllipticPi(hr, kr); //(* Elliptic integral of the third kind *)
    double Pihmkr = EllipticPi(hm, kr);
    double Pihpkr = EllipticPi(hp, kr);

    double Vtr3 = a * xi + ((pow(a, 2) + pow(r3, 2)) * P(r3, a, En, xi)) / CapitalDelta(r3, a);
    double deltaVtr3 = a * deltaxi + (r3 * r3 + a * a) / CapitalDelta(r3, a) * deltaP(r3, a, En, xi, deltaEn, deltaxi);

    double deltaIt1 = (2 * ((deltaEn * Pihrkr * (r2 - r3) * (4 + r1 + r2 + r3)) / 2. + (Ekr * (r1 - r3) * (deltaEn * r1 * r2 * r3 + 2 * xi)) / (2. * r1 * r3) +
                            ((r2 - r3) * ((Pihmkr * (pow(a, 2) + pow(rm, 2)) * deltaP(rm, a, En, xi, deltaEn, deltaxi)) / ((r2 - rm) * (r3 - rm)) -
                                          (Pihpkr * (pow(a, 2) + pow(rp, 2)) * deltaP(rp, a, En, xi, deltaEn, deltaxi)) / ((r2 - rp) * (r3 - rp)))) /
                                (-rm + rp) +
                            Kkr * (-0.5 * (deltaEn * (r1 - r3) * (r2 - r3)) + deltaVtr3))) /
                      sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double cK = Kkr * (-0.5 * (a2 * En * (r1 - r3) * (r2 - r3)) + (pow(a, 4) * En * r3 * (-am1 + pow(r3, 2) * (a1 + 2 * a2 * r3)) +
                                                                   2 * pow(a, 2) * En * pow(r3, 2) * (-(am1 * (-2 + r3)) + a0 * r3 + pow(r3, 3) * (a1 - a2 + 2 * a2 * r3)) +
                                                                   En * pow(r3, 5) * (-2 * a0 - am1 + r3 * (a1 * (-4 + r3) + 2 * a2 * (-3 + r3) * r3)) + 2 * pow(a, 3) * (2 * am1 + a0 * r3 - a2 * pow(r3, 3)) * xi +
                                                                   2 * a * r3 * (am1 * (-6 + 4 * r3) + r3 * (2 * a1 * (-1 + r3) * r3 + a2 * pow(r3, 3) + a0 * (-4 + 3 * r3))) * xi) /
                                                                      (pow(r3, 2) * pow(r3 - rm, 2) * pow(r3 - rp, 2)));
    double cEPi = (En * (a2 * Ekr * r2 * (r1 - r3) + Pihrkr * (r2 - r3) * (2 * a1 + a2 * (4 + r1 + r2 + 3 * r3)))) / 2.;
    double cPi = ((-r2 + r3) * ((Pihmkr * (pow(a, 2) + pow(rm, 2)) * P(rm, a, En, xi) * deltaRt(rm, am1, a0, a1, a2)) / ((r2 - rm) * pow(r3 - rm, 2) * rm) -
                                (Pihpkr * (pow(a, 2) + pow(rp, 2)) * P(rp, a, En, xi) * deltaRt(rp, am1, a0, a1, a2)) / ((r2 - rp) * pow(r3 - rp, 2) * rp))) /
                 (-rm + rp);

    double cE = (Ekr * ((2 * am1 * (-r1 + r3) * xi) / (a * r1) + (r2 * Vtr3 * deltaRt(r3, am1, a0, a1, a2)) / (r2 - r3))) / pow(r3, 2);

    double deltaIt2 = -((cE + cEPi + cK + cPi) / (pow(1 - pow(En, 2), 1.5) * sqrt((r1 - r3) * (r2 - r4))));
    double deltaIt = deltaIt1 + deltaIt2;

    double It = (2 * ((En * (Ekr * r2 * (r1 - r3) + Pihrkr * (r2 - r3) * (4 + r1 + r2 + r3))) / 2. +
                      ((r2 - r3) * ((Pihmkr * (pow(a, 2) + pow(rm, 2)) * P(rm, a, En, xi)) / ((r2 - rm) * (r3 - rm)) - (Pihpkr * (pow(a, 2) + pow(rp, 2)) * P(rp, a, En, xi)) / ((r2 - rp) * (r3 - rp)))) /
                          (-rm + rp) +
                      Kkr * (-0.5 * (En * (r1 - r3) * (r2 - r3)) + Vtr3))) /
                sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double VPhir3 = xi + a / CapitalDelta(r3, a) * P(r3, a, En, xi);
    double deltaVPhir3 = deltaxi + a / CapitalDelta(r3, a) * deltaP(r3, a, En, xi, deltaEn, deltaxi);

    double deltaIPhi1 = (2 * ((Ekr * (r1 - r3) * xi) / (a * r1 * r3) + (a * (r2 - r3) * ((Pihmkr * deltaP(rm, a, En, xi, deltaEn, deltaxi)) / ((r2 - rm) * (r3 - rm)) - (Pihpkr * deltaP(rp, a, En, xi, deltaEn, deltaxi)) / ((r2 - rp) * (r3 - rp)))) / (-rm + rp) + Kkr * deltaVPhir3)) /
                        sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double dK = (Kkr * (-(a * En * pow(r3, 2) * (2 * a0 * (-1 + r3) * r3 + (a1 + 2 * a2) * pow(r3, 3) + am1 * (-4 + 3 * r3))) - pow(a, 3) * En * r3 * (am1 - pow(r3, 2) * (a1 + 2 * a2 * r3)) -
                        pow(a, 2) * (am1 * (-4 + r3) - 2 * a0 * r3 - (a1 + 2 * a2 * (-1 + r3)) * pow(r3, 3)) * xi - pow(-2 + r3, 2) * r3 * (3 * am1 + r3 * (2 * a0 + a1 * r3)) * xi)) /
                (pow(r3, 2) * pow(r3 - rm, 2) * pow(r3 - rp, 2));

    double dPi = -((a * (r2 - r3) * ((Pihmkr * P(rm, a, En, xi) * deltaRt(rm, am1, a0, a1, a2)) / ((r2 - rm) * pow(r3 - rm, 2) * rm) - (Pihpkr * P(rp, a, En, xi) * deltaRt(rp, am1, a0, a1, a2)) / ((r2 - rp) * pow(r3 - rp, 2) * rp))) / (-rm + rp));
    double dE = (Ekr * ((-2 * am1 * (r1 - r3) * xi) / (pow(a, 2) * r1) + (r2 * VPhir3 * deltaRt(r3, am1, a0, a1, a2)) / (r2 - r3))) / pow(r3, 2);

    double deltaIPhi2 = -((dE + dK + dPi) / (pow(1 - pow(En, 2), 1.5) * sqrt((r1 - r3) * (r2 - r4))));
    double deltaIPhi = deltaIPhi1 + deltaIPhi2;

    double IPhi = (2 * ((a * (r2 - r3) * ((Pihmkr * P(rm, a, En, xi)) / ((r2 - rm) * (r3 - rm)) - (Pihpkr * P(rp, a, En, xi)) / ((r2 - rp) * (r3 - rp)))) / (-rm + rp) + Kkr * VPhir3)) /
                  sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double deltaOmegaR = -M_PI / pow(It, 2) * deltaIt;
    double deltaOmegaPhi = deltaIPhi / It - IPhi / pow(It, 2) * deltaIt;

    //                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     cE,                deltaIt2,           It,                deltaIt,            deltaIPhi1, dK, dPi, dE, deltaIPhi2, deltaIPhi, IPhi
    // 0.952696869207406, 2.601147313747245, 11.11111111111111, 9.09090909090909, 1.450341827498306, 0, -0.001534290476164244, -0.1322435748015139, -0.1205695381380546, -0.02411390762761123, 0.0590591407311683, 0.002923427466192832, 0.5641101056459328, 1.435889894354067, 0.03336154445124933, 0.2091139909146586, 0.0217342349165277, 0.0003947869154376093, 1.584149072588183, 1.55761216767624, 1.782193864035892, 1.601724642759611, 1.58446319264442, -112.2728614607676, 1.498105017522236, 111.1816561327114, 1.459858647716873, -7.095639020498573, 133.1766110081966, -7.640013106344259, -0.1508069013343114, -34.72953487758193, 34.00126350567278, 0.7853682931498268, -0.2170281681543273, -0.3678350694886387, 4.044867174992484
    // 0.952697 2.601147                     11.111111          9.090909          1.450342           0.000000 -0.001534        -0.132244            -0.120570            -0.024114             0.059059            0.002923               0.564110           1.435890           0.033362             0.209114            0.021734            0.000395               1.584149           1.557612          1.782194           1.601725           1.584463          -112.272861         1.498105           111.181656         5.642003           -22.992171          -208.799031        -23.536545          -0.150807            -34.729535          34.001264          0.785368            -0.217028            -0.367835            4.044867
    // 0.952697 2.601147 11.111111 9.090909 1.450342 0.000000 -0.001534 -0.132244 -0.120570 -0.024114 0.059059 0.002923 0.564110 1.435890 0.033362 0.209114 0.021734 0.000395 1.584149 1.557612 1.782194 1.601725 1.584463 -112.272861 1.498105 111.181656 1.459859 -7.095639 133.176611 -7.640013 -0.150807 -34.729535 34.001264 0.785368 -0.217028 -0.367835 4.044867
    // 0.952696869207406, 2.601147313747245, 11.11111111111111, 9.09090909090909, 1.450341827498306, 0, -0.001534290476164244, -0.1322435748015139, -0.1205695381380546, -0.02411390762761123, 0.0590591407311683, 0.002923427466192832, 0.5641101056459328, 1.435889894354067, 0.03336154445124933, 0.2091139909146586, 0.0217342349165277, 0.0003947869154376093, 1.584149072588183, 1.55761216767624, 1.782193864035892, 1.601724642759611, 1.58446319264442, -112.2728614607676, 1.498105017522236, 111.1816561327114, 1.459858647716873, -7.095639020498573, 133.1766110081966, -7.640013106344259, -0.1508069013343114, -34.72953487758193, 34.00126350567278, 0.7853682931498268, -0.2170281681543273, -0.3678350694886387, 4.044867174992484
    // printf("En, xi, r1, r2, r3, r4, deltaEn, deltaxi, am1, a0, a1, a2, rm, rp, kr, hr, hm, hp, Kkr, Ekr, Pihrkr, Pihmkr, Pihpkr, cK, cEPi, cPi, cE,deltaIt2, It, deltaIt, deltaPhi1, dK, dPi, dE, deltaPhi2, deltaPhi, IPhi =%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",En, xi, r1, r2, r3, r4, deltaEn, deltaxi, am1, a0, a1, a2, rm, rp, kr, hr, hm, hp, Kkr, Ekr, Pihrkr, Pihmkr, Pihpkr, cK, cEPi, cPi, cE, deltaIt2, It, deltaIt, deltaIPhi1, dK, dPi, dE, deltaIPhi2, deltaIPhi, IPhi);
    *deltaOmegaR_ = deltaOmegaR;
    *deltaOmegaPhi_ = deltaOmegaPhi;
    printf("deltaOmegaR_, deltaOmegaPhi_, = %f %f\n", deltaOmegaR, deltaOmegaPhi);
}

void KerrEqSpinFrequenciesCorrVectorized(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                         double *a, double *p, double *e, double *x, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {

        KerrEqSpinFrequenciesCorrection(&OmegaR_[i], &OmegaPhi_[i],
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
