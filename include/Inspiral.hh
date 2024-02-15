#ifndef __INSPIRAL_H__
#define __INSPIRAL_H__

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>
#include "ode.hh"

#include "Interpolant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

// Used to pass the interpolants to the ODE solver
class ParamsHolder
{
public:
    double epsilon;
    double a = 0.0;
    double q;
    string func_name;
    std::vector<ODECarrier> odes;
    double *additional_args;
    int num_add_args;
    int nparams;
    int num_odes = 0;
    int currently_running_ode_index = 0;

    ParamsHolder(int nparams_, int num_add_args_)
    {
        num_add_args = num_add_args_;
        nparams = nparams_;
        additional_args = new double[num_add_args_];
    };
    void add_ode(string func_name, string few_dir)
    {
        ODECarrier carrier(func_name, few_dir);
        odes.push_back(carrier);
        num_odes += 1;
    }
    ~ParamsHolder()
    {
        delete[] additional_args;
    };
};

class InspiralHolder
{
public:
    int length;
    vector<vector<double>> output_arrs;
    vector<double> t_arr;
    int nparams;

    vector<double> y0;
    // vector<double> amp_norm_out_arr;

    double t0, M, mu, a; // , init_flux;

    InspiralHolder(double t0_, vector<double> y0_, int nparams_)
    {
        t0 = t0_;
        y0 = y0_;
        nparams = nparams_;
        length = 0;

        t_arr.push_back(t0);

        for (int i = 0; i < nparams; i += 1)
        {
            vector<double> tmp = {y0[i]};
            output_arrs.push_back(tmp);
        }
        // amp_norm_out_arr.push_back(init_flux);
    };

    void add_point(double t, vector<double> y0)
    {
        t_arr.push_back(t);
        for (int i = 0; i < nparams; i += 1)
        {
            output_arrs[i].push_back(y0[i]);
        }
        length += 1;
    }

    //	~InspiralHolder();
};

class InspiralCarrier
{
public:
    ParamsHolder *params_holder;
    gsl_odeiv2_system sys;
    gsl_odeiv2_step *step;
    gsl_odeiv2_evolve *evolve;
    gsl_odeiv2_control *control;
    int nparams;
    double err = 1e-11;//1e-10;
    bool USE_DENSE_STEPPING = false;
    bool USE_RK8 = true;
    int num_add_args;
    string func_name;

    InspiralCarrier(int nparams_, int num_add_args_);
    void dealloc();
    void add_parameters_to_holder(double M, double mu, double a, double *additional_args);
    void set_integrator_kwargs(double err_set, bool DENSE_STEP_SET, bool RK8_SET);
    void initialize_integrator();
    void destroy_integrator_information();
    void reset_solver();
    int take_step(double *t, double *h, double *y, const double tmax);
    void get_derivatives(double *ydot_, double *y, int nparams_);
    int get_currently_running_ode_index();
    void update_currently_running_ode_index(int currently_running_ode_index);
    int get_number_of_odes();
    void add_ode(string func_name, string few_dir);
    void get_backgrounds(int *backgrounds, int num_odes);
    void get_equatorial(bool *equatorial, int num_odes);
    void get_circular(bool *circular, int num_odes);
    void get_integrate_constants_of_motion(bool *integrate_constants_of_motion, int num_odes);
    void get_integrate_phases(bool *integrate_phases, int num_odes);
    ~InspiralCarrier();
};

#endif //__INSPIRAL_H__
