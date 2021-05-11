// Code to compute an eccentric Pn5 driven insipral
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

#include "Inspiral5PN.hh"
#include "dIdt8H_5PNe10.h"
#include "FundamentalFrequencies.hh"
#include "global.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision
#include <cstring>


using namespace std;
using namespace std::chrono;

void get_derivatives(double* pdot, double* edot, double* Ydot,
                     double* Omega_phi, double* Omega_theta, double* Omega_r,
                     double epsilon, double a, double p, double e, double Y)
{
    // evaluate ODEs

	int Nv = 10;
    int ne = 10;
    *pdot = epsilon * dpdt8H_5PNe10 (a, p, e, Y, Nv, ne);

    // needs adjustment for validity
    Nv = 10;
    ne = 8;
	*edot = epsilon * dedt8H_5PNe10 (a, p, e, Y, Nv, ne);

    Nv = 7;
    ne = 10;
    *Ydot = epsilon * dYdt8H_5PNe10 (a, p, e, Y, Nv, ne);

    // convert to proper inclination input to fundamental frequencies
    double xI = Y_to_xI(a, p, e, Y);
    KerrGeoCoordinateFrequencies(Omega_phi, Omega_theta, Omega_r, a, p, e, xI);

}

#define  ERROR_INSIDE_SEP  21
// The RHS of the ODEs
int func (double t, const double y[], double f[], void *params){
	(void)(t); /* avoid unused parameter warning */

    ParamsHolder* params_in = (ParamsHolder*) params;
	//double epsilon = params_in->epsilon;
    //double q = params_in->q;
    double a = params_in->a;
    double epsilon = params_in->epsilon;
	double p = y[0];
	double e = y[1];
    double Y = y[2];

    // check for separatrix
    // integrator may naively step over separatrix
    double xI = Y_to_xI(a, p, e, Y);
    double p_sep = get_separatrix(a, e, xI);

    // make sure we are outside the separatrix
    if (p < p_sep)
    {
        return GSL_EBADFUNC;
    }

    double pdot, edot, Ydot;
	double Phi_phi_dot, Phi_theta_dot, Phi_r_dot;


    get_derivatives(&pdot, &edot, &Ydot,
                         &Phi_phi_dot, &Phi_theta_dot, &Phi_r_dot,
                         epsilon, a, p, e, Y);

    //printf("checkit %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e\n", a, p, e, xI, y[3], y[4], y[5], Phi_phi_dot, Phi_theta_dot, Phi_r_dot);

    f[0] = pdot;
	f[1] = edot;
    f[2] = Ydot;
	f[3] = Phi_phi_dot;
    f[4] = Phi_theta_dot;
	f[5] = Phi_r_dot;

  return GSL_SUCCESS;
}


// Class to carry gsl interpolants for the inspiral data
// also executes inspiral calculations
Pn5Carrier::Pn5Carrier()
{
    params_holder = new ParamsHolder;
}

// When interfacing with cython, it helps to have dealloc function to explicitly call
// rather than the deconstructor
void Pn5Carrier::dealloc()
{
    delete params_holder;
}


#define DIST_TO_SEPARATRIX 0.1
#define INNER_THRESHOLD 1e-8
#define PERCENT_STEP 0.25
#define MAX_ITER 1000

// main function in the Pn5Carrier class
// It takes initial parameters and evolves a trajectory
// tmax and dt affect integration length and time steps (mostly if DENSE_STEPPING == 1)
// use_rk4 allows the use of the rk4 integrator
Pn5Holder Pn5Carrier::run_Pn5(double t0, double M, double mu, double a, double p0, double e0, double Y0, double Phi_phi0, double Phi_theta0, double Phi_r0, double err, double tmax, double dt, int DENSE_STEPPING, bool use_rk4)
{

    // years to seconds
    tmax = tmax*YRSID_SI;

    // get flux at initial values
    // prepare containers for flux information
    Pn5Holder pn5_out(t0, M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0);

	//Set the mass ratio
	params_holder->epsilon = mu/M;
    params_holder->a = a;

    double Msec = MTSUN_SI*M;

    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Compute the adimensionalized time steps and max time
    dt = dt /(M*MTSUN_SI);
    tmax = tmax/(M*MTSUN_SI);

    // initial point
	double y[6] = { p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0};

    // Initialize the ODE solver
    gsl_odeiv2_system sys = {func, NULL, 6, params_holder};

    const gsl_odeiv2_step_type *T;
    if (use_rk4) T = gsl_odeiv2_step_rk4;
    else T = gsl_odeiv2_step_rk8pd;

    gsl_odeiv2_step *step 			= gsl_odeiv2_step_alloc (T, 6);
    gsl_odeiv2_control *control 	= gsl_odeiv2_control_y_new (1e-10, 0);
    gsl_odeiv2_evolve *evolve 		= gsl_odeiv2_evolve_alloc (6);

    // Compute the inspiral
	double t = t0;
	double h = dt;
	double t1 = tmax;
    int ind = 1;
    int status = 0;

    double prev_t = 0.0;
    double y_prev[6] = {p0, e0, Y0, 0.0, 0.0, 0.0};

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
        else if ((status == 9)||(std::isnan(y[0]))||(std::isnan(y[1]))||(std::isnan(y[2])) ||(std::isnan(y[3]))||(std::isnan(y[4]))||(std::isnan(y[5])))
        {
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

        // should not be needed but is safeguard against stepping past maximum allowable time
        // the last point in the trajectory will be at t = tmax
        if (t > tmax) break;

        double p 		= y[0];
        double e 		= y[1];
        double Y        = y[2];

        // count the number of points
        ind++;

        // Stop the inspiral when close to the separatrix
        // convert to proper inclination for separatrix
        double xI = Y_to_xI(a, p, e, Y);
        double p_sep = get_separatrix(a, e, xI);
        if(p - p_sep < DIST_TO_SEPARATRIX)
        {
            // Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
            // To correct for this we self-integrate from the second-to-last point in the integation to
            // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

            // Get old values
            p = y_prev[0];
            e = y_prev[1];
            Y = y_prev[2];
            double Phi_phi = y_prev[3];
            double Phi_theta = y_prev[4];
            double Phi_r = y_prev[5];
            t = prev_t;

            double factor = 1.0;
            int iter = 0;

            while (p - p_sep > DIST_TO_SEPARATRIX + INNER_THRESHOLD)
            {
                double pdot, edot, Ydot, Omega_phi, Omega_theta, Omega_r;

                // Same function in the integrator
                get_derivatives(&pdot, &edot, &Ydot,
                                     &Omega_phi, &Omega_theta, &Omega_r,
                                     params_holder->epsilon, a, p, e, Y);

                // estimate the step to the breaking point and multiply by PERCENT_STEP
                xI = Y_to_xI(a, p, e, Y);
                p_sep = get_separatrix(a, e, xI);

                double step_size = PERCENT_STEP / factor * ((p_sep + DIST_TO_SEPARATRIX - p)/pdot);

                // check step
                double temp_t = t + step_size;
                double temp_p = p + pdot * step_size;
                double temp_e = e + edot * step_size;
                double temp_Y = Y + Ydot * step_size;
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
                    Y = temp_Y;
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
            pn5_out.add_point(t*Msec, p, e, Y, Phi_phi, Phi_theta, Phi_r);

            //cout << "# Separatrix reached: exiting inspiral" << endl;
            break;
        }

        pn5_out.add_point(t*Msec, y[0], y[1], y[2], y[3], y[4], y[5]); // adds time in seconds

        prev_t = t;

        #pragma unroll
        for (int jj = 0; jj < 6; jj += 1) y_prev[jj] = y[jj];

	}

	pn5_out.length = ind;
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();

	//	duration<double> time_span = duration_cast<duration<double> >(t2 - t1);

        gsl_odeiv2_evolve_free (evolve);
        gsl_odeiv2_control_free (control);
        gsl_odeiv2_step_free (step);
		//cout << "# Computing the inspiral took: " << time_span.count() << " seconds." << endl;
		return pn5_out;

}

// wrapper for calling the Pn5 inspiral from cython/python
void Pn5Carrier::Pn5Wrapper(double *t, double *p, double *e, double *Y, double *Phi_phi, double *Phi_theta, double *Phi_r, double M, double mu, double a, double p0, double e0, double Y0, double Phi_phi0, double Phi_theta0, double Phi_r0, int *length, double tmax, double dt, double err, int DENSE_STEPPING, bool use_rk4, int init_len){

	double t0 = 0.0;
		Pn5Holder Pn5_vals = run_Pn5(t0, M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, err, tmax, dt, DENSE_STEPPING, use_rk4);

        // make sure we have allocated enough memory through cython
        if (Pn5_vals.length > init_len){
            throw std::runtime_error("Error: Initial length is too short. Inspiral requires more points. Need to raise max_init_len parameter for inspiral.\n");
        }

        // copy data
		memcpy(t, &Pn5_vals.t_arr[0], Pn5_vals.length*sizeof(double));
		memcpy(p, &Pn5_vals.p_arr[0], Pn5_vals.length*sizeof(double));
		memcpy(e, &Pn5_vals.e_arr[0], Pn5_vals.length*sizeof(double));
        memcpy(Y, &Pn5_vals.Y_arr[0], Pn5_vals.length*sizeof(double));
		memcpy(Phi_phi, &Pn5_vals.Phi_phi_arr[0], Pn5_vals.length*sizeof(double));
		memcpy(Phi_theta, &Pn5_vals.Phi_theta_arr[0], Pn5_vals.length*sizeof(double));
        memcpy(Phi_r, &Pn5_vals.Phi_r_arr[0], Pn5_vals.length*sizeof(double));

        // indicate how long is the trajectory
		*length = Pn5_vals.length;

}
