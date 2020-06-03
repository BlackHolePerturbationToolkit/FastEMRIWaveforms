// Code to compute an eccentric flux driven insipral
// into a Schwarzschild black hole
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <complex>
#include <cmath>

#include "Interpolant.h"
#include "SpinWeightedSphericalHarmonics.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision


using namespace std;
using namespace std::chrono;
using namespace std::complex_literals;

// Definitions needed for Mathematicas CForm output
#define Power(x, y)     (pow((double)(x), (double)(y)))
#define Sqrt(x)         (sqrt((double)(x)))
#define Pi              M_PI

// Used to pass the interpolants to the ODE solver
struct interp_params{
	double epsilon;
	Interpolant *Edot;
	Interpolant *Ldot;
};

// The 11 below means the lmax = 10
struct waveform_amps{
	Interpolant ***re[11];
	Interpolant ***im[11];
};

// This code assumes the data is formated in the following way
const int Ne = 33;
const int Ny = 50;

const double SolarMassInSeconds = 4.925e-6;
const double YearInSeconds 		= 60*60*25*365.25;

const int DENSE_STEPPING = 1;

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


void load_and_interpolate_flux_data(struct interp_params *interps){

	// Load and interpolate the flux data
	ifstream Flux_file("data/FluxNewMinusPNScaled.dat");
	
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
	
	// Remove duplicate elements (only works if ys are perfectly repeatined with no round off errors)
	sort( ys.begin(), ys.end() );
	ys.erase( unique( ys.begin(), ys.end() ), ys.end() );
	
	sort( es.begin(), es.end() );
	es.erase( unique( es.begin(), es.end() ), es.end() );
	
	Interpolant *Edot_interp = new Interpolant(ys, es, Edots);
	Interpolant *Ldot_interp = new Interpolant(ys, es, Ldots);
	
	interps->Edot = Edot_interp;
	interps->Ldot = Ldot_interp;
	
}

void create_amplitude_interpolant(hid_t file_id, int l, int m, int n, int Ne, int Ny, vector<double>& ys, vector<double>& es, Interpolant **re, Interpolant **im){
	
	// amplitude data has a real and imaginary part
	double *modeData = new double[2*Ne*Ny];

	char dataset_name[50];
	
	sprintf( dataset_name, "/l%dm%d/n%dk0", l,m,n );

	/* read dataset */
	H5LTread_dataset_double(file_id, dataset_name, modeData);

	vector<double> modeData_re(Ne*Ny);
	vector<double> modeData_im(Ne*Ny);

	for(int i = 0; i < Ne; i++){
		for(int j = 0; j < Ny; j++){
			modeData_re[j + Ny*i] = modeData[2*(Ny - 1 -j + Ny*i)];
			modeData_im[j + Ny*i] = modeData[2*(Ny - 1 -j + Ny*i) + 1];
		}
	}

	*re = new Interpolant(ys, es, modeData_re);
	*im = new Interpolant(ys, es, modeData_im);
}

void load_and_interpolate_amplitude_data(int lmax, int nmax, struct waveform_amps *amps){
	
	hid_t 	file_id;
	hsize_t	dims[2];
	
	file_id = H5Fopen ("/Volumes/GoogleDrive/My Drive/FastEMRIWaveforms/Teuk_amps_a0.0_lmax_10_nmax_30_new.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

	/* get the dimensions of the dataset */
	H5LTget_dataset_info(file_id, "/grid", dims, NULL, NULL);

	/* create an appropriately sized array for the data */
	double *gridRaw = new double[dims[0]*dims[1]];

	/* read dataset */
	H5LTread_dataset_double(file_id, "/grid", gridRaw);

	vector<double> es(Ne);
	vector<double> ys(Ny);

	for(int i = 0; i < Ny; i++){
		double p = gridRaw[1 + 4*i];
		double e = 0;
		
		ys[Ny - 1 - i] = log(0.1 * (10.*p -20*e -21.) );
	}
	
	for(int i = 0; i < Ne; i++){
		es[i] = gridRaw[2 + 4*Ny*i];
	}
	
	for(int l = 2; l <= lmax; l++){
			amps->re[l] = new Interpolant**[l+1];
			amps->im[l] = new Interpolant**[l+1];
			for(int m = 0; m <= l; m++){
				amps->re[l][m] = new Interpolant*[2*nmax +1];
				amps->im[l][m] = new Interpolant*[2*nmax +1];
			}
	}
	
	
	// Load the amplitude data
	for(int l = 2; l <= lmax; l++){
		for(int m = 0; m <= l; m++){
			for(int n = -nmax; n <= nmax; n++){
 				create_amplitude_interpolant(file_id, l, m, n, Ne, Ny, ys, es, &amps->re[l][m][n+nmax], &amps->im[l][m][n+nmax]);		
			}
		}
	}
	
}

int main (int argc, char* argv[]) {
	
	int lmax;
	
	if ( argc != 2 ){
		cout << "Usage: ./FluxInspiral lmax" << endl;
		exit(0);
	}else{
		lmax = atoi(argv[1]); 
	}
	int nmax = 30;
	
	struct waveform_amps amps;
	
	cout << "# Loading and interpolating the amplitude data (this will take a few seconds)" << endl;
	load_and_interpolate_amplitude_data(lmax, nmax, &amps);
	
	
	// double p1 = 12;
	// double e1 = 0.5;
	// double y1 = log(0.1 * (10.*p1 -20*e1 -21.));
	//
	// cout << amps.re[2][2][-5+nmax]->eval(y1, e1) << " " << amps.im[2][2][-5+nmax]->eval(y1, e1) << endl;
		
	// Set the mass of the primary in units of solar masses
	double M = 1e6;
	
	// Set the samplerate in Hertz
	double samplerate = 0.1;
	
	// Signal length (in seconds)
	double max_signal_length = 1*YearInSeconds/356.*60;
	
	// Compute the adimensionalized time steps and max time
	double dt = 1/samplerate /(M*SolarMassInSeconds);
	double tmax = max_signal_length/(M*SolarMassInSeconds);
	
	printf("# tmax =  %lf\n", tmax);
	printf("# time step = %lf\n", dt);
	
	// Sky position
	double theta_d  = M_PI/2.0;
	double phi_d 	= 0.0;
	printf("# sky position: theta = %.12lf, phi = %.12lf\n", theta_d, phi_d);
	
	struct interp_params interps;
	//Set the mass ratio
	interps.epsilon = 1e-5;
	
	cout << "# Loading and interpolating the flux data" << endl;
	load_and_interpolate_flux_data(&interps);
	
	
	// Set the initial values
	double p0 = 12.5;
	double e0 = 0.6;
	double Phi_phi0 = 0;
	double Phi_r0 = 0;
	double t0 = 0;
	
	// Initial values - must start at periastron (otherwise extra phasing factors come in)
	double y[4] = { p0, e0, Phi_phi0, Phi_r0 };
	
	// Precompute the spherical harmonics
	complex<double> *Ylm[lmax+1];
	for(int l = 2; l <= lmax; l++){
		Ylm[l] = new complex<double>[l + 1];
	}
		
	for(int l = 2; l <= lmax; l++){
		for(int m = 0; m <= l; m++){
			 Ylm[l][m] = SpinWeightedSphericalHarmonic(l, m, theta_d, phi_d);
			 //cout << l << " " << m << " " << Ylm[l][m] << endl;
		 }
	 }

	// print out the data at the initial timestep
	double y0 = log((p0 -2.*e0 - 2.1));
	
	complex<double> hwave0 = 0;
	for(int l = 2; l <= lmax; l++){
		for(int m = 0; m <= l; m++){
			complex<double> hwavelm0 = 0;
			for(int n = -nmax; n <= nmax; n++){
				double Phi0 = m*Phi_phi0 + n*Phi_r0;
				hwavelm0 += (amps.re[l][m][n+nmax]->eval(y0,e0) + 1i*amps.im[l][m][n+nmax]->eval(y0,e0))*exp(-1i*Phi0);
			}
			hwavelm0 *= Ylm[l][m];
			hwave0 += hwavelm0;
		}
	}
	
	
	// Output format: t, p, e, Phi_phi, Phi_r
	printf ("# Output format: t p e Phi_phi Phi_r h+ h×\n");
  	printf ("%.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", t0, p0, e0, Phi_phi0, Phi_r0, hwave0.real(), hwave0.imag() );
	
	// Initialize the ODE solver
  	gsl_odeiv2_system sys = {func, NULL, 4, &interps};
    const gsl_odeiv2_step_type *T = gsl_odeiv2_step_rk8pd;

    gsl_odeiv2_step *step 			= gsl_odeiv2_step_alloc (T, 4);
    gsl_odeiv2_control *control 	= gsl_odeiv2_control_y_new (1e-10, 0);
    gsl_odeiv2_evolve *evolve 		= gsl_odeiv2_evolve_alloc (4);

	high_resolution_clock::time_point wallclock1 = high_resolution_clock::now();
	
	// Compute the inspiral
	double t = t0;
	double h = dt;
	double t1 = tmax;
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
		double Phi_phi 	= y[2];
		double Phi_r 	= y[3];
		
		// Stop the inspiral when close to the separatrix
		if(p - 6 -2*e < 0.1){
			cout << "# Separatrix reached: exiting inspiral" << endl;
			break;
		}
		
		double y1 = log((p -2.*e - 2.1));
			
		complex<double> hwave = 0;
		for(int l = 2; l <= lmax; l++){
			for(int m = 0; m <= l; m++){
				complex<double> hwavelm = 0;
				for(int n = -nmax; n <= nmax; n++){
					double Phi = m*Phi_phi + n*Phi_r;
					hwavelm += (amps.re[l][m][n+nmax]->eval(y1,e) + 1i*amps.im[l][m][n+nmax]->eval(y1,e))*exp(-1i*Phi);
				}
				hwavelm *= Ylm[l][m];
				hwave += hwavelm;
			}
		}
		
		// Output format: t, p, e, Phi_phi, Phi_r
      	printf ("%.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", t, p, e, Phi_phi, Phi_r, hwave.real(), hwave.imag() );
    }
	high_resolution_clock::time_point wallclock2 = high_resolution_clock::now();
	
    duration<double> time_span = duration_cast<duration<double> >(wallclock2 - wallclock1);
    cout << "# Computing the inspiral took: " << time_span.count() << " seconds." << endl;

  	gsl_odeiv2_evolve_free (evolve);
  	gsl_odeiv2_control_free (control);
  	gsl_odeiv2_step_free (step);
  	return 0;
}