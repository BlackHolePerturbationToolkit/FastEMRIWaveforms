// Code to compute an eccentric flux driven insipral
// into a Schwarzschild black hole
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>
#include <memory.h>

#include "Interpolant.h"
#include "FluxInspiral.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;


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

	double Edot = -epsilon*interps->Edot->eval(p-2*e, e);
	double Ldot = -epsilon*interps->Ldot->eval(p-2*e, e);

	double pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2));

	double edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
	  (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
	 (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
	(e*(4*Power(e,2) - Power(-6 + p,2))*p));

	// Need to evaluate 4 different elliptic integrals here. Cache them first to avoid repeated calls.
	double EllipE 	= EllipticE(4*e/(p-6.0+2*e));
	double EllipK 	= EllipticK(4*e/(p-6.0+2*e));;
	double EllipPi1 = EllipticPi(16*e/(12.0 + 8*e - 4*e*e - 8*p + p*p), 4*e/(p-6.0+2*e));
	double EllipPi2 = EllipticPi(2*e*(p-4)/((1.0+e)*(p-6.0+2*e)), 4*e/(p-6.0+2*e));

	double Phi_phi_dot 	= (2*Power(p,1.5))/(Sqrt(-4*Power(e,2) + Power(-2 + p,2))*(8 + ((-2*EllipPi2*(6 + 2*e - p)*(3 + Power(e,2) - p)*Power(p,2))/((-1 + e)*Power(1 + e,2)) - (EllipE*(-4 + p)*Power(p,2)*(-6 + 2*e + p))/(-1 + Power(e,2)) +
          (EllipK*Power(p,2)*(28 + 4*Power(e,2) - 12*p + Power(p,2)))/(-1 + Power(e,2)) + (4*(-4 + p)*p*(2*(1 + e)*EllipK + EllipPi2*(-6 - 2*e + p)))/(1 + e) + 2*Power(-4 + p,2)*(EllipK*(-4 + p) + (EllipPi1*p*(-6 - 2*e + p))/(2 + 2*e - p)))/
        (EllipK*Power(-4 + p,2))));

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


void load_and_interpolate_flux_data(struct interp_params *interps){


}


FluxCarrier::FluxCarrier()
{
    interps = new interp_params;

    // Load and interpolate the flux data
	ifstream Flux_file("inspiral/data/Flux.dat");

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

	Interpolant *Edot_interp = new Interpolant(ys, es, Edots);
	Interpolant *Ldot_interp = new Interpolant(ys, es, Ldots);

	interps->Edot = Edot_interp;
	interps->Ldot = Ldot_interp;

}

void FluxCarrier::dealloc()
{

    delete interps->Edot;
    delete interps->Ldot;
    delete interps;

}


FLUXHolder run_FLUX(double t0, double M, double mu, double p0, double e0, double err, FluxCarrier *flux_carrier){
	FLUXHolder flux_out(t0, M, mu, p0, e0);

	interp_params interps = *(flux_carrier->interps);
	//Set the mass ratio
	interps.epsilon = mu/M;

    double Msec = MTSUN_SI*M;

    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

	double tmax = 1e7;


	// Initial values
	// TODO do we want to set initial phases here?
	double y[4] = { p0, e0, 0.0, 0.0 };

	// Initialize the ODE solver
		gsl_odeiv2_system sys = {func, NULL, 4, &interps};
		const gsl_odeiv2_step_type *T = gsl_odeiv2_step_rk8pd;

		gsl_odeiv2_step *s 		= gsl_odeiv2_step_alloc (T, 4);
		gsl_odeiv2_control *c 	= gsl_odeiv2_control_y_new (err, err);
		gsl_odeiv2_evolve *e 	= gsl_odeiv2_evolve_alloc (4);

	double t = t0;
	double h = 1.0;
	int ind=0;
	while (t < tmax){
				int status = gsl_odeiv2_evolve_apply (e, c, s, &sys, &t, tmax, &h, y);

				if (status != GSL_SUCCESS){
					printf ("error, return value=%d\n", status);
						break;
				}
		// Stop the inspiral when close to the separatrix
		if(y[0] - 6 -2*y[1] < 0.1){
			//cout << "# Separatrix reached: exiting inspiral" << endl;
			break;
		}

		flux_out.add_point(t*Msec, y[0], y[1], y[2], y[3]); // adds time in seconds

		// Output format: t, p, e, Phi_phi, Phi_r
				//printf ("%.5e %.5e %.5e %.5e %.5e\n", flux_out.t_arr.push_back(t), flux_out.p_arr.push_back(y[0]), flux_out.e_arr.push_back(y[1]), y[2], y[3]);

				ind++;
		}
	flux_out.length = ind;
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();

	//	duration<double> time_span = duration_cast<duration<double> >(t2 - t1);

		gsl_odeiv2_evolve_free (e);
		gsl_odeiv2_control_free (c);
		gsl_odeiv2_step_free (s);
		//cout << "# Computing the inspiral took: " << time_span.count() << " seconds." << endl;
		return flux_out;

}

void FLUXWrapper(double *t, double *p, double *e, double *Phi_phi, double *Phi_r, double M, double mu, double p0, double e0, int *length, FluxCarrier *flux_carrier, double err){

	double t0 = 0.0;
		FLUXHolder flux_vals = run_FLUX(t0, M, mu, p0, e0, err, flux_carrier);

		memcpy(t, &flux_vals.t_arr[0], flux_vals.length*sizeof(double));
		memcpy(p, &flux_vals.p_arr[0], flux_vals.length*sizeof(double));
		memcpy(e, &flux_vals.e_arr[0], flux_vals.length*sizeof(double));
		memcpy(Phi_phi, &flux_vals.Phi_phi_arr[0], flux_vals.length*sizeof(double));
		memcpy(Phi_r, &flux_vals.Phi_r_arr[0], flux_vals.length*sizeof(double));

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
