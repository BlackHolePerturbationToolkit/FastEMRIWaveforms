// Code to compute an eccentric Inspiral driven insipral
// into a Schwarzschild black hole

// Copyright (C) 2020 Niels Warburton, Michael L. Katz, Alvin J.K. Chua, Scott A. Hughes
//
// Based on implementation from Fujita & Shibata 2020
// See specific code documentation for proper citation.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <math.h>
#include <cmath>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

#include <algorithm>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <complex>
#include <cmath>

#include "Interpolant.h"
#include "Inspiral.hh"
#include "Utility.hh"
#include "global.h"
#include "ode.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip> // setprecision
#include <cstring>

#include <stdexcept>

using namespace std;
using namespace chrono;

#define ERROR_INSIDE_SEP 21

#define DIST_TO_SEPARATRIX 0.1
#define INNER_THRESHOLD 1e-8
#define PERCENT_STEP 0.25
#define MAX_ITER 1000

int func_ode_wrap(double t, const double y[], double f[], void *params)
{
    (void)(t); /* avoid unused parameter warning */

    ParamsHolder *params_in = (ParamsHolder *)params;
    // double epsilon = params_in->epsilon;
    // double q = params_in->q;

    params_in->odes[params_in->currently_running_ode_index].get_derivatives(f, y, params_in->epsilon, params_in->a, params_in->additional_args);

    // cout << "e=" << e << "\t" << "edot=" << edot <<  "\t" << "p=" << p <<  endl;
    return GSL_SUCCESS;
}

InspiralCarrier::InspiralCarrier(ODECarrier *carrier_, string func_name, double tmax_, int nparams_, int num_add_args_)
{
    params_holder = new ParamsHolder(carrier_, func_name, nparams_, num_add_args_);
    nparams = nparams_;

    tmax_seconds = tmax_;
}

void InspiralCarrier::inspiral_wrapper(double *t, double *output_arrs, double M, double mu, double a, double *y0_, int *length, int init_len, double *additional_args)
{

    // Compute the adimensionalized time steps and max time
    tmax = tmax_seconds / (M * MTSUN_SI);
    dt = dt_seconds / (M * MTSUN_SI);

    vector<double> y0(nparams); // convert to standard vector
    memcpy(&y0[0], y0_, nparams * sizeof(double));

    double t0 = 0.0;
    memcpy(params_holder->additional_args, additional_args, params_holder->num_add_args * sizeof(double));
    InspiralHolder Inspiral_vals = run_inspiral(t0, M, mu, a, y0);

    // make sure we have allocated enough memory through cython
    if (Inspiral_vals.length > init_len)
    {
        throw invalid_argument("Error: Initial length is too short. Inspiral requires more points. Need to raise max_init_len parameter for inspiral.\n");
        // throw runtime_error("Error: Initial length is too short. Inspiral requires more points. Need to raise max_init_len parameter for inspiral.\n");
    }

    // copy data
    memcpy(t, &Inspiral_vals.t_arr[0], Inspiral_vals.length * sizeof(double));

    for (int i = 0; i < params_holder->nparams; i += 1)

    {
        memcpy(&output_arrs[i * Inspiral_vals.length], &Inspiral_vals.output_arrs[i], Inspiral_vals.length * sizeof(double));
    }

    // indicate how long is the trajectory
    *length = Inspiral_vals.length;
}

void InspiralCarrier::add_parameters_to_holder(double M, double mu, double a)
{
    // Set the mass ratio
    params_holder->epsilon = mu / M;
    params_holder->a = a;
}

void InspiralCarrier::initialize_integrator()
{

    const gsl_odeiv2_step_type *T;
    if (USE_RK8)
        T = gsl_odeiv2_step_rk8pd;
    else
        T = gsl_odeiv2_step_rk4;

    // Initialize the ODE solver
    // gsl_odeiv2_system sys_temp = {func_ode_wrap, NULL, static_cast<size_t>(nparams), params_holder};

    sys = {func_ode_wrap, NULL, static_cast<size_t>(nparams), params_holder};
    ;
    step = gsl_odeiv2_step_alloc(T, params_holder->nparams);
    control = gsl_odeiv2_control_y_new(err, 0);
    evolve = gsl_odeiv2_evolve_alloc(params_holder->nparams);
}

void InspiralCarrier::destroy_integrator_information()
{
    gsl_odeiv2_evolve_free(evolve);
    gsl_odeiv2_control_free(control);
    gsl_odeiv2_step_free(step);
}

void InspiralCarrier::reset_solver(double *t, double t_prev, double *h, double dt, vector<double> y, vector<double> y_prev, int *bad_num)
{
    // reset evolver
    gsl_odeiv2_step_reset(step);
    gsl_odeiv2_evolve_reset(evolve);

    // go back to previous points
    for (int i = 0; i < params_holder->nparams; i += 1)
    {
        y[i] = y_prev[i];
    }
    *t = t_prev;
    *h = (*h) / 2.;

    // check for number of tries to fix this
    *bad_num = (*bad_num) + 1;
    if (*bad_num >= bad_limit)
    {
        throw invalid_argument("error, reached bad limit.\n");
    }
}

double InspiralCarrier::get_p_sep(vector<double> y)
{
    double a = params_holder->a;
    double p = y[0];
    double e = y[1];
    double x = y[2];

    double p_sep, x_temp;

    if ((params_holder->a == 0.0)) // params_holder->enforce_schwarz_sep ||
    {
        p_sep = 6.0 + 2. * e;
    }
    else
    {
        // (params_holder->odes[params_holder->currently_running_ode_index].convert_Y)
        // {
        //     x_temp = Y_to_xI(a, p, e, x);
        //     // if(sanity_check(a, p, e, x_temp)==1){
        //     //     throw invalid_argument( "277 Wrong conversion to x_temp.");
        //     // }
        // }
        // else
        // {
        //     x_temp = x;
        // }

        p_sep = get_separatrix(a, e, x);
    }
    return p_sep;
}

bool InspiralCarrier::stopping_function(vector<double> y, int status)
{

    if (status == 9)
        return true;

    // Stop the inspiral when close to the separatrix
    // convert to proper inclination for separatrix
    double x_temp;
    double p_sep = get_p_sep(y);
    double p = y[0];
    if (p - p_sep < DIST_TO_SEPARATRIX)
        return true;
    else
        return false;
}

vector<double> InspiralCarrier::end_stepper(double *p_sep_out, double *t_temp_out, double *temp_stop_out, double t, vector<double> y, vector<double> ydot, double factor)
{
    // estimate the step to the breaking point and multiply by PERCENT_STEP
    double p_sep = get_p_sep(y);
    double p = y[0];
    double pdot = ydot[0];
    double step_size = PERCENT_STEP / factor * ((p_sep + DIST_TO_SEPARATRIX - p) / pdot);

    // copy current values
    vector<double> temp_y = y;

    // check step
    for (int i = 0; i < y.size(); i += 1)
        temp_y[i] += ydot[i] * step_size;

    double temp_t = t + step_size;
    double temp_p = temp_y[0];
    double temp_stop = temp_p - p_sep;

    *t_temp_out = temp_t;
    *p_sep_out = p_sep;
    *temp_stop_out = temp_stop;

    return temp_y;
}

void InspiralCarrier::finishing_function(double t, vector<double> y)
{
    // Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
    //     // To correct for this we self-integrate from the second-to-last point in the integation to
    //     // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

    // update p_sep (fixes part of issue #17)
    double p_sep = get_p_sep(y);
    double p = y[0];
    vector<double> ydot(nparams);
    vector<double> y_temp(nparams);

    // set initial values
    double factor = 1.0;
    int iter = 0;
    double t_temp, temp_stop;

    while ((p - p_sep > DIST_TO_SEPARATRIX + INNER_THRESHOLD) && (iter < MAX_ITER))
    {
        // Same function in the integrator
        params_holder->odes[params_holder->currently_running_ode_index].get_derivatives(&ydot[0], &y[0], params_holder->epsilon, params_holder->a, params_holder->additional_args);

        y_temp = end_stepper(&p_sep, &t_temp, &temp_stop, t, y, ydot, factor);
        if (temp_stop > DIST_TO_SEPARATRIX)
        {
            // update points
            t = t_temp;
            y = y_temp;
        }
        else
        {
            // all variables stay the same

            // decrease step
            factor *= 0.5;
        }

        iter++;
    }
}

InspiralHolder InspiralCarrier::integrate(double t0, vector<double> y0)
{

    InspiralHolder inspiral_out(t0, y0, nparams);

    double t = t0;
    double h = dt;

    // Compute the inspiral
    int status = 0;

    double t_prev = 0.0;
    vector<double> y_prev = y0;
    vector<double> y = y0;
    double prev_p_sep;
    double p_sep;

    // control it if it keeps returning nans and what not
    int bad_num = 0;

    while (t < tmax)
    {
        status = take_step(&t, &h, y);

        if ((status != GSL_SUCCESS) && (status != 9)) //  && (status != -1))
        {
            char str[80];
            sprintf(str, "error, return value={}\n", status);
            throw invalid_argument(str);
        }
        // if status is 9 meaning inside the separatrix
        // or if any quantity is nan, step back and take a smaller step.
        bool no_nans = true;
        for (int i = 0; i < params_holder->nparams; i += 1)
        {
            if (isnan(y[i]))
                no_nans = false;
        }

        if ((!no_nans))
        {
            reset_solver(&t, t_prev, &h, dt, y, y_prev, &bad_num);
            continue;
        }

        // if it made it here, reset bad num
        bad_num = 0;

        // should not be needed but is safeguard against stepping past maximum allowable time
        // the last point in the trajectory will be at t = tmax
        if (t > tmax)
            break;

        bool stop = stopping_function(y, status);

        if (stop)
        {
            // go back to last values
            y = y_prev;
            t = t_prev;
            break;
        }

        inspiral_out.add_point(t * Msec, y); // adds time in seconds

        t_prev = t;
        y_prev = y; // vector will copy
    }

    finishing_function(t, y);
}

int InspiralCarrier::take_step(double *t, double *h, vector<double> y_)
{
    int status;
    // apply fixed step if dense stepping
    // or do interpolated step

    double *y = &y_[0];

    if (USE_DENSE_STEPPING)
        status = gsl_odeiv2_evolve_apply_fixed_step(evolve, control, step, &sys, t, *h, y);
    else
        status = gsl_odeiv2_evolve_apply(evolve, control, step, &sys, t, tmax, h, y);

    return status;
}

InspiralHolder InspiralCarrier::run_inspiral(double t0, double M, double mu, double a, vector<double> y0)
{
    // years to seconds
    tmax = tmax * YRSID_SI;
    Msec = MTSUN_SI * M;

    add_parameters_to_holder(M, mu, a);
    initialize_integrator();

    InspiralHolder inspiral_out = integrate(t0, y0);

    destroy_integrator_information();
    return inspiral_out;
}

void InspiralCarrier::dealloc()
{
    delete[] params_holder->additional_args;
    delete params_holder;
}

InspiralCarrier::~InspiralCarrier()
{
    return;
}
