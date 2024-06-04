#include "Interpolant.h"
#include "spline.hpp"

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
    bool convert_Y = false;
    bool equatorial = true;
    int background = SCHWARZSCHILD;
    bool circular = false;
    bool integrate_constants_of_motion = false;
    bool integrate_phases = true;

    SchwarzEccFlux(std::string few_dir);

    void deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
    ~SchwarzEccFlux();
};

class SchwarzEccFlux_nofrequencies
{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;
    double test;
    bool convert_Y = false;
    bool equatorial = true;
    int background = SCHWARZSCHILD;
    bool circular = false;
    bool integrate_constants_of_motion = false;
    bool integrate_phases = false;

    SchwarzEccFlux_nofrequencies(std::string few_dir);

    void deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
    ~SchwarzEccFlux_nofrequencies();
};

class KerrEccentricEquatorial
{
public:
    TensorInterpolant *pdot_interp;
    TensorInterpolant *edot_interp;
    TensorInterpolant *Edot_interp;
    TensorInterpolant *Ldot_interp;
    TricubicSpline *tric_p_interp;
    TricubicSpline *tric_e_interp;
    BicubicSpline *bic_psep_interp;
    KerrEccentricEquatorial(std::string few_dir);
    bool convert_Y = false;
    bool equatorial = true;
    int background = KERR;
    bool circular = false;
    bool integrate_constants_of_motion = false;
    bool integrate_phases = true;

    void deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
    ~KerrEccentricEquatorial();
};


class KerrEccentricEquatorial_nofrequencies
{
public:
    TensorInterpolant *pdot_interp;
    TensorInterpolant *edot_interp;
    TensorInterpolant *Edot_interp;
    TensorInterpolant *Ldot_interp;
    KerrEccentricEquatorial_nofrequencies(std::string few_dir);
    bool convert_Y = false;
    bool equatorial = true;
    int background = KERR;
    bool circular = false;
    bool integrate_constants_of_motion = false;
    bool integrate_phases = false;

    void deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
    ~KerrEccentricEquatorial_nofrequencies();
};

class KerrEccentricEquatorial_ELQ
{
public:
    TensorInterpolant *Edot_interp;
    TensorInterpolant *Ldot_interp;
    KerrEccentricEquatorial_ELQ(std::string few_dir);
    bool convert_Y = false;
    bool equatorial = true;
    int background = KERR;
    bool circular = false;
    bool integrate_constants_of_motion = true;
    bool integrate_phases = true;

    void deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
    ~KerrEccentricEquatorial_ELQ();
};


class KerrEccentricEquatorial_ELQ_nofrequencies
{
public:
    TensorInterpolant *Edot_interp;
    TensorInterpolant *Ldot_interp;
    TensorInterpolant2d *Sep_interp;
    KerrEccentricEquatorial_ELQ_nofrequencies(std::string few_dir);
    bool convert_Y = false;
    bool equatorial = true;
    int background = KERR;
    bool circular = false;
    bool integrate_constants_of_motion = true;
    bool integrate_phases = false;

    void deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
    ~KerrEccentricEquatorial_ELQ_nofrequencies();
};