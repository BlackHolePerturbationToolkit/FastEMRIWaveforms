#ifndef __FLUX_H__
#define __FLUX_H__

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>

#include "Interpolant.h"
#include "FluxInspiral.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;


class NITHolder{
public:
		int length;
		std::vector<double> t_arr;
		std::vector<double> p_arr;
		std::vector<double> e_arr;
		std::vector<double> Phi_phi_arr;
		std::vector<double> Phi_r_arr;
		double t0, M, mu, p0, e0;

		NITHolder(double t0_, double M_, double mu_, double p0_, double e0_){
				t0 = t0_;
                M = M_;
                mu = mu_;
				p0 = p0_;
				e0 = e0_;

				t_arr.push_back(t0);
				p_arr.push_back(p0);
				e_arr.push_back(e0);
				Phi_phi_arr.push_back(0.0);
				Phi_r_arr.push_back(0.0);
		};

		void add_point(double t, double p, double e, double Phi_phi, double Phi_r){
			t_arr.push_back(t);
			p_arr.push_back(p);
			e_arr.push_back(e);
			Phi_phi_arr.push_back(Phi_phi);
			Phi_r_arr.push_back(Phi_r);
		}

	//	~NITHolder();

};

NITHolder run_NIT(double t0, double M, double mu, double p0, double e0, double err);

void NITWrapper(double *t, double *p, double *e, double *Phi_phi, double *Phi_r, double M, double mu, double p0, double e0, int *length, double err);

#endif //__FLUX_H__
