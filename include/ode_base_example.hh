#include "Interpolant.h"

// Used to pass the interpolants to the ODE solver
struct interp_params
{
    double epsilon;
    Interpolant *Edot;
    Interpolant *Ldot;
    Interpolant *Edot_Kerr;
};

class SchwarzEccFlux
{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;
    double test;

    SchwarzEccFlux(std::string few_dir);

    void deriv_func(double *pdot, double *edot, double *Ydot,
                    double *Omega_phi, double *Omega_theta, double *Omega_r,
                    double epsilon, double a, double p, double e, double Y, double *additional_args);
    ~SchwarzEccFlux();
};

class Relativistic_Kerr_Circ_Flux
{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;
    double test;
    Relativistic_Kerr_Circ_Flux(std::string few_dir);

    double EdotPN(double r, double a);

    void deriv_func(double *pdot, double *edot, double *Ydot,
                    double *Omega_phi, double *Omega_theta, double *Omega_r,
                    double epsilon, double a, double p, double e, double Y, double *additional_args);
    ~Relativistic_Kerr_Circ_Flux();
};
