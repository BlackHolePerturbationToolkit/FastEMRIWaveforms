// Code to compute an eccentric flux driven insipral
// into a Schwarzschild black hole
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

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision
#include <cstring>

using namespace std;
using namespace std::chrono;

// Definitions needed for Mathematicas CForm output
#define Power(x, y)     (pow((double)(x), (double)(y)))
#define Sqrt(x)         (sqrt((double)(x)))
#define Pi              M_PI


// This code assumes the data is formated in the following way
const int Ne = 33;
const int Ny = 50;

const double YearInSeconds 		= 60*60*25*365.25;

//const int DENSE_STEPPING = 0;

// Define elliptic integrals that use Mathematica's conventions
double EllipticK(double k){
        return gsl_sf_ellint_Kcomp(sqrt(k), GSL_PREC_DOUBLE);
}

double EllipticF(double phi, double k){
        return gsl_sf_ellint_F(phi, sqrt(k), GSL_PREC_DOUBLE) ;
}

double EllipticE(double k){
        return gsl_sf_ellint_Ecomp(sqrt(k), GSL_PREC_DOUBLE);
}

double EllipticEIncomp(double phi, double k){
        return gsl_sf_ellint_E(phi, sqrt(k), GSL_PREC_DOUBLE) ;
}

double EllipticPi(double n, double k){
        return gsl_sf_ellint_Pcomp(sqrt(k), -n, GSL_PREC_DOUBLE);
}

double EllipticPiIncomp(double n, double phi, double k){
        return gsl_sf_ellint_P(phi, sqrt(k), -n, GSL_PREC_DOUBLE);
}

// The RHS of the ODEs
int func (double t, const double y[], double f[], void *params){
	(void)(t); /* avoid unused parameter warning */
	struct interp_params *interps = (struct interp_params *)params;

	double epsilon = interps->epsilon;
	double p = y[0];
	double e = y[1];

	double y1 = log((p -2.*e - 2.1));

	// Need to evaluate 4 different elliptic integrals here. Cache them first to avoid repeated calls.
	double EllipE 	= EllipticE(4*e/(p-6.0+2*e));
	double EllipK 	= EllipticK(4*e/(p-6.0+2*e));;
	double EllipPi1 = EllipticPi(16*e/(12.0 + 8*e - 4*e*e - 8*p + p*p), 4*e/(p-6.0+2*e));
	double EllipPi2 = EllipticPi(2*e*(p-4)/((1.0+e)*(p-6.0+2*e)), 4*e/(p-6.0+2*e));

	double Omega_phi = (2*Power(p,1.5))/(Sqrt(-4*Power(e,2) + Power(-2 + p,2))*(8 + ((-2*EllipPi2*(6 + 2*e - p)*(3 + Power(e,2) - p)*Power(p,2))/((-1 + e)*Power(1 + e,2)) - (EllipE*(-4 + p)*Power(p,2)*(-6 + 2*e + p))/(-1 + Power(e,2)) +
          (EllipK*Power(p,2)*(28 + 4*Power(e,2) - 12*p + Power(p,2)))/(-1 + Power(e,2)) + (4*(-4 + p)*p*(2*(1 + e)*EllipK + EllipPi2*(-6 - 2*e + p)))/(1 + e) + 2*Power(-4 + p,2)*(EllipK*(-4 + p) + (EllipPi1*p*(-6 - 2*e + p))/(2 + 2*e - p)))/
        (EllipK*Power(-4 + p,2))));

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



	double Phi_phi_dot 	= Omega_phi;

	double Phi_r_dot 	= (p*Sqrt((-6 + 2*e + p)/(-4*Power(e,2) + Power(-2 + p,2)))*Pi)/
   (8*EllipK + ((-2*EllipPi2*(6 + 2*e - p)*(3 + Power(e,2) - p)*Power(p,2))/((-1 + e)*Power(1 + e,2)) - (EllipE*(-4 + p)*Power(p,2)*(-6 + 2*e + p))/(-1 + Power(e,2)) +
        (EllipK*Power(p,2)*(28 + 4*Power(e,2) - 12*p + Power(p,2)))/(-1 + Power(e,2)) + (4*(-4 + p)*p*(2*(1 + e)*EllipK + EllipPi2*(-6 - 2*e + p)))/(1 + e) + 2*Power(-4 + p,2)*(EllipK*(-4 + p) + (EllipPi1*p*(-6 - 2*e + p))/(2 + 2*e - p)))/
      Power(-4 + p,2));

	f[0] = pdot;
	f[1] = edot;
	f[2] = Phi_phi_dot;
	f[3] = Phi_r_dot;

  return GSL_SUCCESS;
}

void load_and_interpolate_amp_vec_norm_data(Interpolant **amp_vec_norm_interp){

	// Load and interpolate the flux data
	ifstream Flux_file("inspiral/data/AmplitudeVectorNorm.dat");

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


void load_and_interpolate_flux_data(struct interp_params *interps){

	// Load and interpolate the flux data
	ifstream Flux_file("inspiral/data/FluxNewMinusPNScaled_fixed_y_order.dat");

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



FluxCarrier::FluxCarrier()
{
    interps = new interp_params;

    load_and_interpolate_flux_data(interps);
	load_and_interpolate_amp_vec_norm_data(&amp_vec_norm_interp);

}

void FluxCarrier::dealloc()
{

    delete interps->Edot;
    delete interps->Ldot;
    delete interps;

    delete amp_vec_norm_interp;

}

double get_step_flux(double p, double e, Interpolant *amp_vec_norm_interp)
{

    double y0 = log((p -2.*e - 2.1));
    double step_flux = amp_vec_norm_interp->eval(y0, e);
    return step_flux;

}


FLUXHolder run_FLUX(double t0, double M, double mu, double p0, double e0, double err, FluxCarrier *flux_carrier, int DENSE_STEPPING){

    double init_flux = get_step_flux(p0, e0, flux_carrier->amp_vec_norm_interp);

    FLUXHolder flux_out(t0, M, mu, p0, e0, init_flux);

    interp_params interps = *(flux_carrier->interps);
	//Set the mass ratio
	interps.epsilon = mu/M;

    double Msec = MTSUN_SI*M;

    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Set the samplerate in Hertz
    double samplerate = 0.1;

    // Signal length (in seconds)
    double max_signal_length = 1*YearInSeconds;

    // Compute the adimensionalized time steps and max time
    double dt = 1/samplerate /(M*MTSUN_SI);
    double tmax = max_signal_length/(M*MTSUN_SI);

    // Initial values
	// TODO do we want to set initial phases here?


	double y[4] = { p0, e0, 0.0, 0.0 };

    // Initialize the ODE solver
    gsl_odeiv2_system sys = {func, NULL, 4, &interps};
    const gsl_odeiv2_step_type *T = gsl_odeiv2_step_rk8pd;

    gsl_odeiv2_step *step 			= gsl_odeiv2_step_alloc (T, 4);
    gsl_odeiv2_control *control 	= gsl_odeiv2_control_y_new (1e-10, 0);
    gsl_odeiv2_evolve *evolve 		= gsl_odeiv2_evolve_alloc (4);

    // Compute the inspiral
	double t = t0;
	double h = dt;
	double t1 = tmax;
    int ind = 0;
	if(DENSE_STEPPING) t1 = dt;
	while (t < tmax){

        int status = gsl_odeiv2_evolve_apply (evolve, control, step, &sys, &t, t1, &h, y);
		if(DENSE_STEPPING) t1 += dt;
      	if (status != GSL_SUCCESS){
       		printf ("error, return value=%d\n", status);
          	break;
        }

        double p 		= y[0];
        double e 		= y[1];

        double step_flux = get_step_flux(p, e, flux_carrier->amp_vec_norm_interp);

        flux_out.add_point(t*Msec, y[0], y[1], y[2], y[3], step_flux); // adds time in seconds

        ind++;
        // Stop the inspiral when close to the separatrix
        if(p - 6 -2*e < 0.1){
            //cout << "# Separatrix reached: exiting inspiral" << endl;
            break;
        }

		// Output format: t, p, e, Phi_phi, Phi_r
				//printf ("%.5e %.5e %.5e %.5e %.5e\n", flux_out.t_arr.push_back(t), flux_out.p_arr.push_back(y[0]), flux_out.e_arr.push_back(y[1]), y[2], y[3]);

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

void FLUXWrapper(double *t, double *p, double *e, double *Phi_phi, double *Phi_r, double *amp_norm, double M, double mu, double p0, double e0, int *length, FluxCarrier *flux_carrier, double err, int DENSE_STEPPING){

	double t0 = 0.0;
		FLUXHolder flux_vals = run_FLUX(t0, M, mu, p0, e0, err, flux_carrier, DENSE_STEPPING);

		memcpy(t, &flux_vals.t_arr[0], flux_vals.length*sizeof(double));
		memcpy(p, &flux_vals.p_arr[0], flux_vals.length*sizeof(double));
		memcpy(e, &flux_vals.e_arr[0], flux_vals.length*sizeof(double));
		memcpy(Phi_phi, &flux_vals.Phi_phi_arr[0], flux_vals.length*sizeof(double));
		memcpy(Phi_r, &flux_vals.Phi_r_arr[0], flux_vals.length*sizeof(double));
        memcpy(amp_norm, &flux_vals.amp_norm_out_arr[0], flux_vals.length*sizeof(double));

		*length = flux_vals.length;

}

/*
int main (void) {

	// Set the initial values
	double p0 = 12.5;
	double e0 = 0.5;
	double t0 = 0;

	FLUXHolder check = run_FLUX(t0, p0, e0);

	for (int i=0; i<check.length; i++){
			printf("%e %e %e\n", check.t_arr[i], check.p_arr[i], check.e_arr[i]);
	}
	printf("length is %d\n", check.length);


}
*/
