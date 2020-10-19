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


// The RHS of the ODEs
int func (double t, const double y[], double f[], void *params){
	(void)(t); /* avoid unused parameter warning */

    ParamsHolder* params_in = (ParamsHolder*) params;
	//double epsilon = params_in->epsilon;
    //double q = params_in->q;
    double a = params_in->a;
    double epilson = params_in->epsilon;
	double p = y[0];
	double e = y[1];
    double Y = y[2];

    // evaluate ODEs

	int Nv = 10;
    int ne = 10;
    double pdot = epilson * dpdt8H_5PNe10 (a, p, e, Y, Nv, ne);

    // needs adjustment for validity
    Nv = 10;
    ne = 8;
	double edot = epilson * dedt8H_5PNe10 (a, p, e, Y, Nv, ne);

    Nv = 7;
    ne = 10;
    double Ydot = epilson * dYdt8H_5PNe10 (a, p, e, Y, Nv, ne);

	double Phi_phi_dot, Phi_theta_dot, Phi_r_dot;

    KerrGeoCoordinateFrequencies(&Phi_phi_dot, &Phi_theta_dot, &Phi_r_dot, a, p, e, Y);

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


// main function in the Pn5Carrier class
// It takes initial parameters and evolves a trajectory
// tmax and dt affect integration length and time steps (mostly if DENSE_STEPPING == 1)
// use_rk4 allows the use of the rk4 integrator
Pn5Holder Pn5Carrier::run_Pn5(double t0, double M, double mu, double a, double p0, double e0, double Y0, double Phi_phi0, double Phi_theta0, double Phi_r0, double err, double tmax, double dt, int DENSE_STEPPING, bool use_rk4){

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

	while (t < tmax){

        // apply fixed step if dense stepping
        // or do interpolated step
		if(DENSE_STEPPING) status = gsl_odeiv2_evolve_apply_fixed_step (evolve, control, step, &sys, &t, h, y);
        else int status = gsl_odeiv2_evolve_apply (evolve, control, step, &sys, &t, t1, &h, y);

        // should not be needed but is safeguard against stepping past maximum allowable time
        // the last point in the trajectory will be at t = tmax
        if (t > tmax) break;

      	if (status != GSL_SUCCESS){
       		printf ("error, return value=%d\n", status);
          	break;
        }

        double p 		= y[0];
        double e 		= y[1];
        double Y        = y[2];

        pn5_out.add_point(t*Msec, y[0], y[1], y[2], y[3], y[4], y[5]); // adds time in seconds

        // count the number of points
        ind++;

        // Stop the inspiral when close to the separatrix

        if (p < 12.0)
        {
            double p_sep = get_separatrix(a, e, Y);
            if(p - p_sep < 0.2)
            {
                //cout << "# Separatrix reached: exiting inspiral" << endl;
                break;
            }
        }
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
