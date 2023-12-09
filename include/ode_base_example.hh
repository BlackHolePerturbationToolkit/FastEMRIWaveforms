#include "Interpolant.h"

#define KERR 1
#define SCHWARZSCHILD 2

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
    bool equatorial = true;
    int background = SCHWARZSCHILD;
    bool circular = false;
    SchwarzEccFlux(std::string few_dir);

    void deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args);
    ~SchwarzEccFlux();
};

class KerrEccentricEquatorial
{
public:
    TensorInterpolant *pdot_interp;
    TensorInterpolant *edot_interp;
    TensorInterpolant *Edot_interp;
    TensorInterpolant *Ldot_interp;
    KerrEccentricEquatorial(std::string few_dir);
    bool equatorial = true;
    int background = KERR;
    bool circular = false;

    void deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args);
    ~KerrEccentricEquatorial();
};
