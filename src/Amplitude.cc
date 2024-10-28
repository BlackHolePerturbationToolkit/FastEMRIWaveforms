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
#include "Utility.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip> // std::setprecision

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

// initialize amplitude interpolants for each mode
void create_amplitude_interpolant(hid_t file_id, int l, int m, int n, int Ne, int Ny, vector<double> &ys, vector<double> &es, Interpolant **re, Interpolant **im)
{

    // amplitude data has a real and imaginary part
    double *modeData = new double[2 * Ne * Ny];

    char dataset_name[50];

    sprintf(dataset_name, "/l%dm%d/n%dk0", l, m, n);

    /* read dataset */
    H5LTread_dataset_double(file_id, dataset_name, modeData);

    vector<double> modeData_re(Ne * Ny);
    vector<double> modeData_im(Ne * Ny);

    for (int i = 0; i < Ne; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            modeData_re[j + Ny * i] = modeData[2 * (Ny - 1 - j + Ny * i)];
            modeData_im[j + Ny * i] = modeData[2 * (Ny - 1 - j + Ny * i) + 1];
        }
    }

    // initialize interpolants
    *re = new Interpolant(ys, es, modeData_re);
    *im = new Interpolant(ys, es, modeData_im);

    delete[] modeData;
}

// collect data and initialize amplitude interpolants
void load_and_interpolate_amplitude_data(int lmax, int nmax, struct waveform_amps *amps, const std::string &few_dir)
{

    hid_t file_id;
    hsize_t dims[2];

    std::string fp = "few/files/Teuk_amps_a0.0_lmax_10_nmax_30_new.h5";
    fp = few_dir + fp;
    file_id = H5Fopen(fp.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    /* get the dimensions of the dataset */
    H5LTget_dataset_info(file_id, "/grid", dims, NULL, NULL);

    /* create an appropriately sized array for the data */
    double *gridRaw = new double[dims[0] * dims[1]];

    /* read dataset */
    H5LTread_dataset_double(file_id, "/grid", gridRaw);

    vector<double> es(Ne);
    vector<double> ys(Ny);

    // convert p -> y
    for (int i = 0; i < Ny; i++)
    {
        double p = gridRaw[1 + 4 * i];
        double e = 0;

        ys[Ny - 1 - i] = log(0.1 * (10. * p - 20 * e - 21.));
    }

    for (int i = 0; i < Ne; i++)
    {
        es[i] = gridRaw[2 + 4 * Ny * i];
    }

    for (int l = 2; l <= lmax; l++)
    {
        amps->re[l] = new Interpolant **[l + 1];
        amps->im[l] = new Interpolant **[l + 1];
        for (int m = 0; m <= l; m++)
        {
            amps->re[l][m] = new Interpolant *[2 * nmax + 1];
            amps->im[l][m] = new Interpolant *[2 * nmax + 1];
        }
    }

    // Load the amplitude data
    for (int l = 2; l <= lmax; l++)
    {
        for (int m = 0; m <= l; m++)
        {
            for (int n = -nmax; n <= nmax; n++)
            {
                create_amplitude_interpolant(file_id, l, m, n, Ne, Ny, ys, es, &amps->re[l][m][n + nmax], &amps->im[l][m][n + nmax]);
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
    for (int l = 2; l <= lmax; l++)
    {
        for (int m = 0; m <= l; m++)
        {
            for (int n = -nmax; n <= nmax; n++)
            {
                delete amps->re[l][m][n + nmax];
                delete amps->im[l][m][n + nmax];
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

    for (int i = 0; i < num; i++)
    {
        for (int mode_i = 0; mode_i < num_modes; mode_i++)
        {
            double p = p_arr[i];
            double e = e_arr[i];

            double y = log((p - 2. * e - 2.1));

            // calculate amplitudes for this mode
            int l = l_arr[mode_i];
            int m = m_arr[mode_i];
            int n = n_arr[mode_i];
            amplitude_out[i * num_modes + mode_i] = amps->re[l][m][n + nmax]->eval(y, e) + I * amps->im[l][m][n + nmax]->eval(y, e);
        }
    }
}

//---------------------------------------------- *****KERR DATA
// initialize amplitude interpolants for each mode

const int Nu = 99;
const int Na = 100;
const int Ne2 = 1;

void create_amplitude_interpolant_Kerr(hid_t file_id, int l, int m, int n, int Na, int Nu, int Ne2, vector<double> &as, vector<double> &us, vector<double> &es, Interpolant **re, Interpolant **im)
{

    // amplitude data has a real and imaginary part
    double *modeData = new double[4 * Na * Nu];

    char dataset_name[50];

    sprintf(dataset_name, "/Clmkn/l%dm%dn%d", l, m, n);
    // printf("dataset %s \n", dataset_name);

    /* read dataset */
    H5LTread_dataset_double(file_id, dataset_name, modeData);

    vector<double> modeData_re(Na * Nu);
    vector<double> modeData_im(Na * Nu);

    for (int i = 0; i < Na; i++)
    {
        for (int j = 0; j < Nu; j++)
        {
            modeData_re[j + Nu * i] = modeData[2 * (j + Nu * i) + 0]; // the 1st (0) col is spin the 2nd (1) is u and then it is real and imag parts of Clms
            modeData_im[j + Nu * i] = modeData[2 * (j + Nu * i) + 1];
            // printf("debug output line 402 :%d %d \t %1.6e %1.6e \n",i,j, modeData_re[j + Nu*i],modeData_im[j + Nu*i] );
        }
    }

    // initialize interpolants
    *re = new Interpolant(us, as, modeData_re);
    *im = new Interpolant(us, as, modeData_im); // the bug was here

    delete[] modeData;
}

// *****KERR DATA
// collect data and initialize amplitude interpolants
void load_and_interpolate_amplitude_data_Kerr(int lmax, int nmax, struct waveform_amps_Kerr *amps, const std::string &few_dir)
{

    hid_t file_id;
    hsize_t dims[2];

    std::string fp = "few/files/Clm00_e0.0_lmax_30.h5";
    fp = few_dir + fp;
    file_id = H5Fopen(fp.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    /* get the dimensions of the dataset */
    H5LTget_dataset_info(file_id, "/grid", dims, NULL, NULL);

    /* create an appropriately sized array for the data */
    double *gridRaw = new double[dims[0] * dims[1]];

    /* read dataset */
    H5LTread_dataset_double(file_id, "/grid", gridRaw);

    vector<double> as(Na);
    vector<double> us(Nu);
    vector<double> es(Ne2);

    // convert p -> y
    for (int i = 0; i < Nu; i++)
    {
        us[i] = gridRaw[3 + 4 * i];
        // printf("us: %d %1.5e \n",i, us[i]);
    }

    for (int i = 0; i < Na; i++)
    {
        as[i] = gridRaw[0 + 4 * Nu * i];
        // printf("spins: %d %1.5e \n",i, as[i]);
    }

    for (int i = 0; i < Ne2; i++)
    {
        es[i] = gridRaw[1 + 4 * Nu * i];
    }

    for (int l = 2; l <= lmax; l++)
    {
        amps->re[l] = new Interpolant **[l + 1]; // FOr each l, we have 2l+1 "m" mode, but since we do not need negative "m" then it would be only l+1 "m" modes
        amps->im[l] = new Interpolant **[l + 1]; // we allocate the same amount of memory to imaginary part as well, assuming that the real and imag part are two different variables
        for (int m = 0; m <= l; m++)
        {
            amps->re[l][m] = new Interpolant *[2 * nmax + 1];
            amps->im[l][m] = new Interpolant *[2 * nmax + 1];
        }
    }

    // Load the amplitude data
    for (int l = 2; l <= lmax; l++)
    {
        for (int m = 0; m <= l; m++)
        {
            for (int n = -nmax; n <= nmax; n++) // the bug was here instead of n++ I had m++ :(
            {

                create_amplitude_interpolant_Kerr(file_id, l, m, n, Na, Nu, Ne2, as, us, es, &amps->re[l][m][n + nmax], &amps->im[l][m][n + nmax]);
            }
        }
    }

    delete[] gridRaw;
}

// Amplitude Carrier is class for interaction with python carrying gsl interpolant information  for KERR********
AmplitudeCarrier_Kerr::AmplitudeCarrier_Kerr(int lmax_, int nmax_, std::string few_dir)
{
    lmax = lmax_;
    nmax = nmax_;

    amps = new struct waveform_amps_Kerr;

    load_and_interpolate_amplitude_data_Kerr(lmax, nmax, amps, few_dir);
    // printf("after load func\n");
}

// need to have dealloc method for cython interface  for KERR********
void AmplitudeCarrier_Kerr::dealloc()
{

    // clear memory
    for (int l = 2; l <= lmax; l++)
    {
        for (int m = 0; m <= l; m++)
        {
            for (int n = -nmax; n <= nmax; n++)
            {

                delete amps->re[l][m][n + nmax];
                delete amps->im[l][m][n + nmax];
            }

            delete amps->re[l][m];
            delete amps->im[l][m];
        }
        delete amps->re[l];
        delete amps->im[l];
    }
    delete amps;
}

// main function for computing amplitudes for KERR********
void AmplitudeCarrier_Kerr::Interp2DAmplitude_Kerr(std::complex<double> *amplitude_out, double *a_arr, double *p_arr, double *e_arr, int *l_arr, int *m_arr, int *n_arr, int num, int num_modes)
{

    // printf("in the interp3d func\n");
    complex<double> I(0.0, 1.0);

#ifdef __USE_OMP__
#pragma omp parallel for collapse(2)
#endif // __USE_OMP__
    for (int i = 0; i < num; i++)
    {
        for (int mode_i = 0; mode_i < num_modes; mode_i++)
        {
            double p = p_arr[i];
            double a = a_arr[i];
            double e = e_arr[i];
            // double e = 0.0;
            double x = 1.0;
            double ps = get_separatrix(a, e, x);
            double u = log((p - ps + 3.9));

            // calculate amplitudes for this mode
            int l = l_arr[mode_i];
            int m = m_arr[mode_i];
            int n = n_arr[mode_i];
            amplitude_out[i * num_modes + mode_i] = amps->re[l][m][n + nmax]->eval(u, a) + I * amps->im[l][m][n + nmax]->eval(u, a);
        }
    }
}
