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
#include <iomanip>      // std::setprecision
#include <cstring>

#include <stdexcept>


using namespace std;
using namespace std::chrono;

#define  ERROR_INSIDE_SEP  21

#define DIST_TO_SEPARATRIX 0.1
#define INNER_THRESHOLD 1e-8
#define PERCENT_STEP 0.25
#define MAX_ITER 1000
// The RHS of the ODEs
int func_ode_wrap (double t, const double y[], double f[], void *params){
	(void)(t); /* avoid unused parameter warning */

    ParamsHolder* params_in = (ParamsHolder*) params;
	//double epsilon = params_in->epsilon;
    //double q = params_in->q;
    double a = params_in->a;
    double epsilon = params_in->epsilon;
	double p = y[0];
	double e = y[1];
    double x = y[2];

    // check for separatrix
    // integrator may naively step over separatrix
    double x_temp;

    // define a sanity check
    if(sanity_check(a, p, e, x)==1){
        return GSL_EBADFUNC;
    }
    double p_sep = 0.0;
    if (params_in->convert_Y)
    {
        // estimate separatrix with Y since it is close to x
        // make sure we are not inside it or root solver will struggle
        p_sep = get_separatrix(a, e, x);
        // make sure we are outside the separatrix
        if (p < p_sep + DIST_TO_SEPARATRIX)
        {
            return GSL_EBADFUNC;
        }
        x_temp = Y_to_xI(a, p, e, x);
    }
    else
    {
        x_temp = x;
    }
    
    if (params_in->enforce_schwarz_sep || (a == 0.0))
    {
        p_sep = 6.0 + 2. * e;
    }
    else
    {
        p_sep = get_separatrix(a, e, x_temp);
    }

    // make sure we are outside the separatrix
    if (p < p_sep + DIST_TO_SEPARATRIX)
    {
        return GSL_EBADFUNC;
    }

    double pdot, edot, xdot;
	double Omega_phi, Omega_theta, Omega_r;

    params_in->func->get_derivatives(&pdot, &edot, &xdot,
                         &Omega_phi, &Omega_theta, &Omega_r,
                         epsilon, a, p, e, x, params_in->additional_args);

    f[0] = pdot;
	f[1] = edot;
    f[2] = xdot;
	f[3] = Omega_phi;
    f[4] = Omega_theta;
	f[5] = Omega_r;

  return GSL_SUCCESS;
}


InspiralCarrier::InspiralCarrier(ODECarrier* testcarrier_, std::string func_name, bool enforce_schwarz_sep_, int num_add_args_, bool convert_Y_, std::string few_dir)
{
    params_holder = new ParamsHolder;
    params_holder->func_name = func_name;
    params_holder->enforce_schwarz_sep = enforce_schwarz_sep_;
    params_holder->num_add_args = num_add_args_;
    params_holder->convert_Y = convert_Y_;
    params_holder->additional_args = new double[num_add_args_];
    params_holder->func = testcarrier_;
}

void InspiralCarrier::inspiral_wrapper(double *t, double *p, double *e, double *x, double *Phi_phi, double *Phi_theta, double *Phi_r, double M, double mu, double a, double p0, double e0, double x0, double Phi_phi0, double Phi_theta0, double Phi_r0, int *length, double tmax, double dt, double err, int DENSE_STEPPING, bool use_rk4, int init_len, double* additional_args)
{
    double t0 = 0.0;
    std::memcpy(params_holder->additional_args, additional_args, params_holder->num_add_args * sizeof(double));
    InspiralHolder Inspiral_vals = run_inspiral(t0, M, mu, a, p0, e0, x0, Phi_phi0, Phi_theta0, Phi_r0, err, tmax, dt, DENSE_STEPPING, use_rk4);
 
    // make sure we have allocated enough memory through cython
    if (Inspiral_vals.length > init_len){
        throw std::invalid_argument("Error: Initial length is too short. Inspiral requires more points. Need to raise max_init_len parameter for inspiral.\n");
        // throw std::runtime_error("Error: Initial length is too short. Inspiral requires more points. Need to raise max_init_len parameter for inspiral.\n");
    }

    // copy data
    memcpy(t, &Inspiral_vals.t_arr[0], Inspiral_vals.length*sizeof(double));
    memcpy(p, &Inspiral_vals.p_arr[0], Inspiral_vals.length*sizeof(double));
    memcpy(e, &Inspiral_vals.e_arr[0], Inspiral_vals.length*sizeof(double));
    memcpy(x, &Inspiral_vals.x_arr[0], Inspiral_vals.length*sizeof(double));
    memcpy(Phi_phi, &Inspiral_vals.Phi_phi_arr[0], Inspiral_vals.length*sizeof(double));
    memcpy(Phi_theta, &Inspiral_vals.Phi_theta_arr[0], Inspiral_vals.length*sizeof(double));
    memcpy(Phi_r, &Inspiral_vals.Phi_r_arr[0], Inspiral_vals.length*sizeof(double));

    // indicate how long is the trajectory
    *length = Inspiral_vals.length;
}

InspiralHolder InspiralCarrier::run_inspiral(double t0, double M, double mu, double a, double p0, double e0, double x0, double Phi_phi0, double Phi_theta0, double Phi_r0, double err, double tmax, double dt, int DENSE_STEPPING, bool use_rk4)
{
    // years to seconds
    tmax = tmax*YRSID_SI;

    InspiralHolder inspiral_out(t0, M, mu, a, p0, e0, x0, Phi_phi0, Phi_theta0, Phi_r0);

	//Set the mass ratio
	params_holder->epsilon = mu/M;
    params_holder->a = a;
    params_holder->enforce_schwarz_sep;

    double Msec = MTSUN_SI*M;

    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Compute the adimensionalized time steps and max time
    dt = dt /(M*MTSUN_SI);
    tmax = tmax/(M*MTSUN_SI);

    // initial point
	double y[6] = { p0, e0, x0, Phi_phi0, Phi_theta0, Phi_r0};


    
    // Initialize the ODE solver
    gsl_odeiv2_system sys = {func_ode_wrap, NULL, 6, params_holder};

    const gsl_odeiv2_step_type *T;
    if (use_rk4) T = gsl_odeiv2_step_rk4;
    else T = gsl_odeiv2_step_rk8pd;

    gsl_odeiv2_step *step 			= gsl_odeiv2_step_alloc (T, 6);
    gsl_odeiv2_control *control 	= gsl_odeiv2_control_y_new (err, 0);
    gsl_odeiv2_evolve *evolve 		= gsl_odeiv2_evolve_alloc (6);

    // Compute the inspiral
	double t = t0;
	double h = dt;
	double t1 = tmax;
    int ind = 1;
    int status = 0;

    double prev_t = 0.0;
    double prev_p_sep = 0.0;
    double y_prev[6] = {p0, e0, x0, 0.0, 0.0, 0.0};

    // control it if it keeps returning nans and what not
    int bad_num = 0;
    int bad_limit = 1000;

	while (t < tmax){
        // apply fixed step if dense stepping
        // or do interpolated step
		if(DENSE_STEPPING) status = gsl_odeiv2_evolve_apply_fixed_step (evolve, control, step, &sys, &t, h, y);
        else status = gsl_odeiv2_evolve_apply (evolve, control, step, &sys, &t, t1, &h, y);

      	if ((status != GSL_SUCCESS) && (status != 9)){
       		printf ("error, return value=%d\n", status);
          	break;
        }
        // if status is 9 meaning inside the separatrix
        // or if any quantity is nan, step back and take a smaller step.
        else if ((std::isnan(y[0]))||(std::isnan(y[1]))||(std::isnan(y[2])) ||(std::isnan(y[3]))||(std::isnan(y[4]))||(std::isnan(y[5])))
        {
            ///printf("checkit error %.18e %.18e %.18e %.18e \n", y[0], y_prev[0], y[1], y_prev[1]);
            // reset evolver
            gsl_odeiv2_step_reset(step);
            gsl_odeiv2_evolve_reset(evolve);

            // go back to previous points
            #pragma unroll
            for (int i = 0; i < 6; i += 1)
            {
                y[i] = y_prev[i];
            }
            t = prev_t;
            h /= 2.;

            // check for number of tries to fix this
            bad_num += 1;
            if (bad_num >= bad_limit)
            {
                printf ("error, reached bad limit.\n");
                break;
            }

            continue;
        }

        // if it made it here, reset bad num
        bad_num = 0;

        double p 		= y[0];
        double e 		= y[1];
        double x        = y[2];

        // check eccentricity
        if (e < 0.0)
        {
            // integrator may have leaked past zero
            if (e > -1e-3)
            {
                e = 1e-6;
            }
            // integrator went way past zero throw error.
            else 
            {
                throw std::invalid_argument("Error: the integrator is stepping the eccentricity too far across zero (e < -1e-3).\n");
            }
        }

        // should not be needed but is safeguard against stepping past maximum allowable time
        // the last point in the trajectory will be at t = tmax
        if (t > tmax) break;

        // count the number of points
        ind++;

        // Stop the inspiral when close to the separatrix
        // convert to proper inclination for separatrix
        double x_temp;
        double p_sep = 0.0;
        if (status != 9)
        {

            if (params_holder->convert_Y)
            {
                x_temp = Y_to_xI(a, p, e, x);
                // if(sanity_check(a, p, e, x_temp)==1){
                //     throw std::invalid_argument( "277 Wrong conversion to x_temp.");
                // }
            }
            else
            {
                x_temp = x;
            }


            if (params_holder->enforce_schwarz_sep || (a == 0.0))
            {
                p_sep = 6.0 + 2. * e;
            }
            else
            {
                p_sep = get_separatrix(a, e, x_temp);
            }

        }

        // status 9 indicates integrator stepped inside separatrix limit
        if((status == 9) || (p - p_sep < DIST_TO_SEPARATRIX))
        {
            // Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
            // To correct for this we self-integrate from the second-to-last point in the integation to
            // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

            // Get old values
            p = y_prev[0];
            e = y_prev[1];
            x = y_prev[2];

            double Phi_phi = y_prev[3];
            double Phi_theta = y_prev[4];
            double Phi_r = y_prev[5];
            t = prev_t;

            // update p_sep (fixes part of issue #17)
            p_sep = prev_p_sep;

            // set initial values
            double factor = 1.0;
            int iter = 0;

            while (p - p_sep > DIST_TO_SEPARATRIX + INNER_THRESHOLD)
            {
                double pdot, edot, xdot, Omega_phi, Omega_theta, Omega_r;

                // Same function in the integrator
                params_holder->func->get_derivatives(&pdot, &edot, &xdot,
                                     &Omega_phi, &Omega_theta, &Omega_r,
                                     params_holder->epsilon, a, p, e, x, params_holder->additional_args);

                // estimate the step to the breaking point and multiply by PERCENT_STEP
                double x_temp;
                if (params_holder->convert_Y)
                {
                    x_temp = Y_to_xI(a, p, e, x);
                    // if(sanity_check(a, p, e, x_temp)==1){
                    // throw std::invalid_argument( "336 Wrong conversion to x_temp");
                    // }
                }
                else
                {
                    x_temp = x;
                }

                if (params_holder->enforce_schwarz_sep || (a == 0.0))
                {
                    p_sep = 6.0 + 2. * e;
                }
                else
                {
                    p_sep = get_separatrix(a, e, x_temp);
                }

                double step_size = PERCENT_STEP / factor * ((p_sep + DIST_TO_SEPARATRIX - p)/pdot);

                // check step
                double temp_t = t + step_size;
                double temp_p = p + pdot * step_size;
                double temp_e = e + edot * step_size;
                double temp_x = x + xdot * step_size;
                double temp_Phi_phi = Phi_phi + Omega_phi * step_size;
                double temp_Phi_theta = Phi_theta + Omega_theta * step_size;
                double temp_Phi_r = Phi_r + Omega_r * step_size;

                double temp_stop = temp_p - p_sep;
                if (temp_stop > DIST_TO_SEPARATRIX)
                {
                    // update points
                    t = temp_t;
                    p = temp_p;
                    e = temp_e;
                    x = temp_x;
                    Phi_phi = temp_Phi_phi;
                    Phi_theta = temp_Phi_theta;
                    Phi_r = temp_Phi_r;
                }
                else
                {
                    // all variables stay the same

                    // decrease step
                    factor *= 0.5;
                }

                iter++;

                // guard against diverging calculations
                if (iter > MAX_ITER)
                {
                    break;
                }

            }

            // add the point and end the integration
            inspiral_out.add_point(t*Msec, p, e, x, Phi_phi, Phi_theta, Phi_r);

            //cout << "# Separatrix reached: exiting inspiral" << endl;
            break;
        }

        inspiral_out.add_point(t*Msec, y[0], y[1], y[2], y[3], y[4], y[5]); // adds time in seconds

        prev_t = t;
        prev_p_sep = p_sep;
        #pragma unroll
        for (int jj = 0; jj < 6; jj += 1) y_prev[jj] = y[jj];

	}

	inspiral_out.length = ind;
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();

	//	duration<double> time_span = duration_cast<duration<double> >(t2 - t1);

    gsl_odeiv2_evolve_free (evolve);
    gsl_odeiv2_control_free (control);
    gsl_odeiv2_step_free (step);
    //cout << "# Computing the inspiral took: " << time_span.count() << " seconds." << endl;
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
