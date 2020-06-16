#ifndef __FLUX_H__
#define __FLUX_H__

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>

#include "Interpolant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Definitions needed for Mathematicas CForm output
#define Power(x, y)     (pow((double)(x), (double)(y)))
#define Sqrt(x)         (sqrt((double)(x)))
#define Pi              M_PI
#define MTSUN_SI 4.925491025543575903411922162094833998e-6

// Used to pass the interpolants to the ODE solver
struct interp_params{
	double epsilon;
	Interpolant *Edot;
	Interpolant *Ldot;
};

class FLUXHolder{
public:
		int length;
		std::vector<double> t_arr;
		std::vector<double> p_arr;
		std::vector<double> e_arr;
		std::vector<double> Phi_phi_arr;
		std::vector<double> Phi_r_arr;
        std::vector<double> amp_norm_out_arr;

		double t0, M, mu, p0, e0, init_flux;

		FLUXHolder(double t0_, double M_, double mu_, double p0_, double e0_, double init_flux_){
				t0 = t0_;
                M = M_;
                mu = mu_;
				p0 = p0_;
				e0 = e0_;
                init_flux = init_flux_;

				t_arr.push_back(t0);
				p_arr.push_back(p0);
				e_arr.push_back(e0);
				Phi_phi_arr.push_back(0.0);
				Phi_r_arr.push_back(0.0);
                amp_norm_out_arr.push_back(init_flux);
		};

		void add_point(double t, double p, double e, double Phi_phi, double Phi_r, double step_flux){
			t_arr.push_back(t);
			p_arr.push_back(p);
			e_arr.push_back(e);
			Phi_phi_arr.push_back(Phi_phi);
			Phi_r_arr.push_back(Phi_r);
            amp_norm_out_arr.push_back(step_flux);
		}

	//	~FLUXHolder();

};

class FluxCarrier{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;

    FluxCarrier();
    void dealloc();
};

FLUXHolder run_FLUX(double t0, double M, double mu, double p0, double e0, FluxCarrier *flux_carrier, double err);

void FLUXWrapper(double *t, double *p, double *e, double *Phi_phi, double *Phi_r, double *amp_norm, double M, double mu, double p0, double e0, int *length, FluxCarrier *flux_carrier, double err);

#endif //__FLUX_H__
