// Code to compute eccentric flux amplitudes with bicubic splines

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
#include "Amplitude.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision

#include <stdio.h>

// if not using omp remove it
#ifdef __USE_OMP__
#include <omp.h>
#endif

using namespace std;
using namespace std::chrono;

// This code assumes the data is formated in the following way
const int Ne = 33;
const int Ny = 50;


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

// initialize amplitude interpolants for each mode
void create_amplitude_interpolant(hid_t file_id, int l, int m, int n, int Ne, int Ny, vector<double>& ys, vector<double>& es, Interpolant **re, Interpolant **im){

	// amplitude data has a real and imaginary part
	double *modeData = new double[2*Ne*Ny];

	char dataset_name[50];

	sprintf( dataset_name, "/l%dm%d/n%dk0", l,m,n );

	/* read dataset */
	H5LTread_dataset_double(file_id, dataset_name, modeData);

	vector<double> modeData_re(Ne*Ny);
	vector<double> modeData_im(Ne*Ny);

	for(int i = 0; i < Ne; i++)
    {
		for(int j = 0; j < Ny; j++)
        {
			modeData_re[j + Ny*i] = modeData[2*(Ny - 1 -j + Ny*i)];
			modeData_im[j + Ny*i] = modeData[2*(Ny - 1 -j + Ny*i) + 1];
		}
	}

    // initialize interpolants
	*re = new Interpolant(ys, es, modeData_re);
	*im = new Interpolant(ys, es, modeData_im);

    delete[] modeData;
}

// collect data and initialize amplitude interpolants
void load_and_interpolate_amplitude_data(int lmax, int nmax, struct waveform_amps *amps, const std::string& few_dir){

	hid_t 	file_id;
	hsize_t	dims[2];

    std::string fp = "few/files/Teuk_amps_a0.0_lmax_10_nmax_30_new.h5";
    fp = few_dir + fp;
	file_id = H5Fopen (fp.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

	/* get the dimensions of the dataset */
	H5LTget_dataset_info(file_id, "/grid", dims, NULL, NULL);

	/* create an appropriately sized array for the data */
	double *gridRaw = new double[dims[0]*dims[1]];

	/* read dataset */
	H5LTread_dataset_double(file_id, "/grid", gridRaw);

	vector<double> es(Ne);
	vector<double> ys(Ny);

    // convert p -> y
	for(int i = 0; i < Ny; i++)
    {
		double p = gridRaw[1 + 4*i];
		double e = 0;

		ys[Ny - 1 - i] = log(0.1 * (10.*p -20*e -21.) );
	}

	for(int i = 0; i < Ne; i++)
    {
		es[i] = gridRaw[2 + 4*Ny*i];
	}

	for(int l = 2; l <= lmax; l++)
    {
			amps->re[l] = new Interpolant**[l+1];
			amps->im[l] = new Interpolant**[l+1];
			for(int m = 0; m <= l; m++)
            {
				amps->re[l][m] = new Interpolant*[2*nmax +1];
				amps->im[l][m] = new Interpolant*[2*nmax +1];
			}
	}


	// Load the amplitude data
	for(int l = 2; l <= lmax; l++)
    {
		for(int m = 0; m <= l; m++)
        {
			for(int n = -nmax; n <= nmax; n++)
            {
                create_amplitude_interpolant(file_id, l, m, n, Ne, Ny, ys, es, &amps->re[l][m][n+nmax], &amps->im[l][m][n+nmax]);
			}
		}
	}

    delete[] gridRaw;
}

// Amplitude Carrier is class for interaction with python carrying gsl interpolant information
AmplitudeCarrier::AmplitudeCarrier(int lmax_, int nmax_, std::string few_dir)
{
    lmax = lmax_;
    nmax = nmax_;

    amps = new struct waveform_amps;

    load_and_interpolate_amplitude_data(lmax, nmax, amps, few_dir);

}

// need to have dealloc method for cython interface
void AmplitudeCarrier::dealloc()
{

    // clear memory
    for(int l = 2; l <= lmax; l++)
    {
        for(int m = 0; m <= l; m++)
        {
            for(int n = -nmax; n <= nmax; n++)
            {
                delete amps->re[l][m][n+nmax];
                delete amps->im[l][m][n+nmax];
            }
            delete amps->re[l][m];
            delete amps->im[l][m];
        }
        delete amps->re[l];
        delete amps->im[l];
    }
    delete amps;
}


// main function for computing amplitudes
void AmplitudeCarrier::Interp2DAmplitude(std::complex<double> *amplitude_out, double *p_arr, double *e_arr, int *l_arr, int *m_arr, int *n_arr, int num, int num_modes)
{

    complex<double> I(0.0, 1.0);

    for (int i=0; i<num; i++)
    {
    	for(int mode_i = 0; mode_i < num_modes; mode_i++)
        {
            double p = p_arr[i];
            double e = e_arr[i];

            double y = log((p -2.*e - 2.1));

            // calculate amplitudes for this mode
            int l = l_arr[mode_i]; int m = m_arr[mode_i]; int n = n_arr[mode_i];
			amplitude_out[i*num_modes + mode_i]= amps->re[l][m][n+nmax]->eval(y,e) + I*amps->im[l][m][n+nmax]->eval(y,e);

        }
    }
}
