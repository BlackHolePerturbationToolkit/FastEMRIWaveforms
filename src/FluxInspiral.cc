// Code to compute an eccentric flux driven insipral
// into a Schwarzschild black hole

// Copyright (C) 2020 Niels Warburton, Michael L. Katz, Alvin J.K. Chua, Scott A. Hughes
//
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
#include <algorithm>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <complex>
#include <cmath>

#include "Interpolant.h"
#include "FluxInspiral.hh"
#include "global.h"
#include "FundamentalFrequencies.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision
#include <cstring>


using namespace std;
using namespace std::chrono;


// This code assumes the data is formated in the following way
const int Ne = 33;
const int Ny = 50;


// The RHS of the ODEs
int func (double t, const double y[], double f[], void *params){
	(void)(t); /* avoid unused parameter warning */
	struct interp_params *interps = (struct interp_params *)params;

	double epsilon = interps->epsilon;
	double p = y[0];
	double e = y[1];

	double y1 = log((p -2.*e - 2.1));

    // evaluate ODEs, starting with PN contribution, then interpolating over remaining flux contribution
	double Omega_phi = 0.0;
    double Omega_r = 0.0;

    SchwarzschildGeoCoordinateFrequencies(&Omega_phi, &Omega_r, p, e);

	double yPN = pow(Omega_phi,2./3.);

	double EdotPN = (96 + 292*Power(e,2) + 37*Power(e,4))/(15.*Power(1 - Power(e,2),3.5)) * pow(yPN, 5);
	double LdotPN = (4*(8 + 7*Power(e,2)))/(5.*Power(-1 + Power(e,2),2)) * pow(yPN, 7./2.);

	double Edot = -epsilon*(interps->Edot->eval(y1, e)*pow(yPN,6.) + EdotPN);
	double Ldot = -epsilon*(interps->Ldot->eval(y1, e)*pow(yPN,9./2.) + LdotPN);

	double pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2));

	double edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
	  (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
	 (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
	(e*(4*Power(e,2) - Power(-6 + p,2))*p));

	double Phi_phi_dot = Omega_phi;
    double Phi_r_dot = Omega_r;

	f[0] = pdot;
	f[1] = edot;
	f[2] = Phi_phi_dot;
	f[3] = Phi_r_dot;

  return GSL_SUCCESS;
}

// initialize amplitude vector norm calculation
void load_and_interpolate_amp_vec_norm_data(Interpolant **amp_vec_norm_interp, const std::string& few_dir){

	// Load and interpolate the flux data
    std::string fp = "few/files/AmplitudeVectorNorm.dat";
    fp = few_dir + fp;
	ifstream Flux_file(fp);

    if (Flux_file.fail())
    {
        throw std::runtime_error("The file AmplitudeVectorNorm.dat did not open sucessfully. Make sure it is located in the proper directory (Path/to/Installation/few/files/).");
    }

	// Load the flux data into arrays
	string Flux_string;
	vector<double> ys, es, vec_norms;
	double y, e, vec_norm;
	while(getline(Flux_file, Flux_string)){

		stringstream Flux_ss(Flux_string);

		Flux_ss >> y >> e >> vec_norm;

		ys.push_back(y);
		es.push_back(e);
		vec_norms.push_back(vec_norm);
	}

	// Remove duplicate elements (only works if ys are perfectly repeating with no round off errors)
	sort( ys.begin(), ys.end() );
	ys.erase( unique( ys.begin(), ys.end() ), ys.end() );

	sort( es.begin(), es.end() );
	es.erase( unique( es.begin(), es.end() ), es.end() );

	*amp_vec_norm_interp = new Interpolant(ys, es, vec_norms);
}

// Initialize flux data for inspiral calculations
void load_and_interpolate_flux_data(struct interp_params *interps, const std::string& few_dir){

	// Load and interpolate the flux data
    std::string fp = "few/files/FluxNewMinusPNScaled_fixed_y_order.dat";
    fp = few_dir + fp;
	ifstream Flux_file(fp);

    if (Flux_file.fail())
    {
        throw std::runtime_error("The file FluxNewMinusPNScaled_fixed_y_order.dat did not open sucessfully. Make sure it is located in the proper directory (Path/to/Installation/few/files/).");
    }

	// Load the flux data into arrays
	string Flux_string;
	vector<double> ys, es, Edots, Ldots;
	double y, e, Edot, Ldot;
	while(getline(Flux_file, Flux_string)){

		stringstream Flux_ss(Flux_string);

		Flux_ss >> y >> e >> Edot >> Ldot;

		ys.push_back(y);
		es.push_back(e);
		Edots.push_back(Edot);
		Ldots.push_back(Ldot);
	}

	// Remove duplicate elements (only works if ys are perfectly repeating with no round off errors)
	sort( ys.begin(), ys.end() );
	ys.erase( unique( ys.begin(), ys.end() ), ys.end() );

	sort( es.begin(), es.end() );
	es.erase( unique( es.begin(), es.end() ), es.end() );

	Interpolant *Edot_interp = new Interpolant(ys, es, Edots);
	Interpolant *Ldot_interp = new Interpolant(ys, es, Ldots);

	interps->Edot = Edot_interp;
	interps->Ldot = Ldot_interp;

}

// Class to carry gsl interpolants for the inspiral data
// also executes inspiral calculations
FluxCarrier::FluxCarrier(std::string few_dir)
{
    interps = new interp_params;

    // prepare the data
    // python will download the data if
    // the user does not have it in the correct place
    load_and_interpolate_flux_data(interps, few_dir);
	load_and_interpolate_amp_vec_norm_data(&amp_vec_norm_interp, few_dir);

}

// When interfacing with cython, it helps to have  dealloc function to explicitly call
// rather than the deconstructor
void FluxCarrier::dealloc()
{

    delete interps->Edot;
    delete interps->Ldot;
    delete interps;

    delete amp_vec_norm_interp;

}

// get the amplitude vector norm at p and e with bicubic spline
double get_step_flux(double p, double e, Interpolant *amp_vec_norm_interp)
{

    double y0 = log((p -2.*e - 2.1));
    double step_flux = amp_vec_norm_interp->eval(y0, e);
    return step_flux;

}

// main function in the FluxCarrier class
// It takes initial parameters and evolves a trajectory
// tmax and dt affect integration length and time steps (mostly if DENSE_STEPPING == 1)
// use_rk4 allows the use of the rk4 integrator
FLUXHolder FluxCarrier::run_FLUX(double t0, double M, double mu, double p0, double e0, double Phi_phi0, double Phi_r0, double err, double tmax, double dt, int DENSE_STEPPING, bool use_rk4){

    // years to seconds
    tmax = tmax*YRSID_SI;

    // get flux at initial values
    double init_flux = get_step_flux(p0, e0, amp_vec_norm_interp);

    // prepare containers for flux information
    FLUXHolder flux_out(t0, M, mu, p0, e0, Phi_phi0, Phi_r0, init_flux);

	//Set the mass ratio
	interps->epsilon = mu/M;

    double Msec = MTSUN_SI*M;

    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Compute the adimensionalized time steps and max time
    dt = dt /(M*MTSUN_SI);
    tmax = tmax/(M*MTSUN_SI);

    // initial point
	double y[4] = { p0, e0, Phi_phi0, Phi_r0 };

    // Initialize the ODE solver
    gsl_odeiv2_system sys = {func, NULL, 4, interps};

    const gsl_odeiv2_step_type *T;
    if (use_rk4) T = gsl_odeiv2_step_rk4;
    else T = gsl_odeiv2_step_rk8pd;

    gsl_odeiv2_step *step 			= gsl_odeiv2_step_alloc (T, 4);
    gsl_odeiv2_control *control 	= gsl_odeiv2_control_y_new (1e-10, 0);
    gsl_odeiv2_evolve *evolve 		= gsl_odeiv2_evolve_alloc (4);

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

        double step_flux = get_step_flux(p, e, amp_vec_norm_interp);

        flux_out.add_point(t*Msec, y[0], y[1], y[2], y[3], step_flux); // adds time in seconds

        // count the number of points
        ind++;

        // Stop the inspiral when close to the separatrix
        if(p - 6 -2*e < 0.1){
            //cout << "# Separatrix reached: exiting inspiral" << endl;
            break;
        }
	}

	flux_out.length = ind;
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();

	//	duration<double> time_span = duration_cast<duration<double> >(t2 - t1);

        gsl_odeiv2_evolve_free (evolve);
        gsl_odeiv2_control_free (control);
        gsl_odeiv2_step_free (step);
		//cout << "# Computing the inspiral took: " << time_span.count() << " seconds." << endl;
		return flux_out;

}

// wrapper for calling the flux inspiral from cython/python
void FluxCarrier::FLUXWrapper(double *t, double *p, double *e, double *Phi_phi, double *Phi_r, double *amp_norm, double M, double mu, double p0, double e0, double Phi_phi0, double Phi_r0, int *length, double tmax, double dt, double err, int DENSE_STEPPING, bool use_rk4, int init_len){

	double t0 = 0.0;
		FLUXHolder flux_vals = run_FLUX(t0, M, mu, p0, e0, Phi_phi0, Phi_r0, err, tmax, dt, DENSE_STEPPING, use_rk4);

        // make sure we have allocated enough memory through cython
        if (flux_vals.length > init_len){
            throw std::runtime_error("Error: Initial length is too short. Inspiral requires more points. Need to raise max_init_len parameter for inspiral.\n");
        }

        // copy data
		memcpy(t, &flux_vals.t_arr[0], flux_vals.length*sizeof(double));
		memcpy(p, &flux_vals.p_arr[0], flux_vals.length*sizeof(double));
		memcpy(e, &flux_vals.e_arr[0], flux_vals.length*sizeof(double));
		memcpy(Phi_phi, &flux_vals.Phi_phi_arr[0], flux_vals.length*sizeof(double));
		memcpy(Phi_r, &flux_vals.Phi_r_arr[0], flux_vals.length*sizeof(double));
        memcpy(amp_norm, &flux_vals.amp_norm_out_arr[0], flux_vals.length*sizeof(double));

        // indicate how long is the trajectory
		*length = flux_vals.length;

}
