#include "Interpolant.h"

// Used to pass the interpolants to the ODE solver
struct interp_params
{
    double epsilon;
    Interpolant *Edot;
    Interpolant *Ldot;
};

class SchwarzEccFlux
{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;
    double test;

    SchwarzEccFlux(std::string few_dir);

    void deriv_func(double ydot[], double y[], double epsilon, double a, double *additional_args);
    ~SchwarzEccFlux();
};

class KerrEccentricEquatorial
{
public:
    KerrEccentricEquatorial(std::string few_dir);

    void deriv_func(double ydot[], double y[], double epsilon, double a, double *additional_args);
};
