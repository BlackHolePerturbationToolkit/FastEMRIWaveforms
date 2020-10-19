#ifndef __PN5_H__
#define __PN5_H__

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


// Used to pass the interpolants to the ODE solver
typedef struct tag_ParamsHolder{
	double epsilon;
	double a;
    double q;
} ParamsHolder;

class Pn5Holder{
public:
		int length;
		std::vector<double> t_arr;
		std::vector<double> p_arr;
		std::vector<double> e_arr;
        std::vector<double> Y_arr;
		std::vector<double> Phi_phi_arr;
		std::vector<double> Phi_r_arr;
        std::vector<double> Phi_theta_arr;
        //std::vector<double> amp_norm_out_arr;

		double t0, M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0; // , init_flux;

		Pn5Holder(double t0_, double M_, double mu_, double a_, double p0_, double e0_, double Y0_, double Phi_phi0_, double Phi_theta0_, double Phi_r0_){
				t0 = t0_;
                M = M_;
                mu = mu_;
                a = a_;
				p0 = p0_;
				e0 = e0_;
                Y0 = Y0_;
                Phi_phi0 = Phi_phi0_;
                Phi_theta0 = Phi_theta0_;
                Phi_r0 = Phi_r0_;

				t_arr.push_back(t0);
				p_arr.push_back(p0);
				e_arr.push_back(e0);
                Y_arr.push_back(Y0);
				Phi_phi_arr.push_back(Phi_phi0);
                Phi_theta_arr.push_back(Phi_theta0);
                Phi_r_arr.push_back(Phi_r0);
                //amp_norm_out_arr.push_back(init_flux);
		};

		void add_point(double t, double p, double e, double Y, double Phi_phi, double Phi_theta, double Phi_r){
			t_arr.push_back(t);
			p_arr.push_back(p);
			e_arr.push_back(e);
            Y_arr.push_back(Y);
			Phi_phi_arr.push_back(Phi_phi);
			Phi_theta_arr.push_back(Phi_theta);
			Phi_r_arr.push_back(Phi_r);
		}

	//	~Pn5Holder();

};


class Pn5Carrier{
public:
    ParamsHolder *params_holder;

    Pn5Carrier();

    Pn5Holder run_Pn5(double t0, double M, double mu, double a, double p0, double e0, double Y0, double Phi_phi0, double Phi_theta0, double Phi_r0,
        double err, double tmax, double dt, int DENSE_STEPPING, bool use_rk4);

    void Pn5Wrapper(double *t, double *p, double *e, double *Y, double *Phi_phi, double *Phi_theta, double *Phi_r, double M, double mu, double a, double p0, double e0, double Y0, double Phi_phi0, double Phi_theta0, double Phi_r0, int *length, double tmax, double dt, double err, int DENSE_STEPPING, bool use_rk4, int init_len);

    void dealloc();
};


#endif //__PN5_H__
