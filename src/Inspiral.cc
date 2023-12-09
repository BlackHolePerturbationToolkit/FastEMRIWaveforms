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

InspiralCarrier::InspiralCarrier(ODECarrier *carrier_, string func_name_, int nparams_, int num_add_args_)
{
    params_holder = new ParamsHolder(carrier_, func_name_, nparams_, num_add_args_);
    nparams = nparams_;
    num_add_args = num_add_args_;
    func_name = func_name_;
}

void InspiralCarrier::add_parameters_to_holder(double M, double mu, double a, double *additional_args)
{
    // Set the mass ratio
    params_holder->epsilon = mu / M;
    params_holder->a = a;
    memcpy(&(params_holder->additional_args[0]), additional_args, params_holder->num_add_args * sizeof(double));
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

void InspiralCarrier::reset_solver()
{
    // reset evolver
    gsl_odeiv2_step_reset(step);
    gsl_odeiv2_evolve_reset(evolve);
}

void InspiralCarrier::get_derivatives(double *ydot_, double *y, int nparams_)
{
    if (nparams_ != nparams)
        throw invalid_argument("nparams input for derivatives does not match nparams stored in the c++ class.");

    vector<double> ydot(nparams);
    params_holder->odes[params_holder->currently_running_ode_index].get_derivatives(&ydot[0], y, params_holder->epsilon, params_holder->a, params_holder->additional_args);
    memcpy(ydot_, &ydot[0], nparams * sizeof(double));
}

int InspiralCarrier::take_step(double *t_in, double *h_in, double *y, const double tmax)
{
    int status;
    // apply fixed step if dense stepping
    // or do interpolated step
    double t = *t_in;
    double h = *h_in;

    if (USE_DENSE_STEPPING)
        status = gsl_odeiv2_evolve_apply_fixed_step(evolve, control, step, &sys, &t, h, y);
    else
        status = gsl_odeiv2_evolve_apply(evolve, control, step, &sys, &t, tmax, &h, y);

    if ((status != GSL_SUCCESS) && (status != 9)) //  && (status != -1))
    {
        char str[80];
        sprintf(str, "error, return value={}\n", status);
        throw invalid_argument(str);
    }
    return status;
}

void InspiralCarrier::dealloc()
{
    delete params_holder;
}

InspiralCarrier::~InspiralCarrier()
{
    return;
}
