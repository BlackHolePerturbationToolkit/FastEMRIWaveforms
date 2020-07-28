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
#include "Amplitude.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision

#ifdef __USE_OMP__
#include <omp.h>
#endif
#include <stdio.h>


using namespace std;
using namespace std::chrono;
// using namespace std::complex_literals;

// Definitions needed for Mathematicas CForm output
#define Power(x, y)     (pow((double)(x), (double)(y)))
#define Sqrt(x)         (sqrt((double)(x)))
#define Pi              M_PI


// This code assumes the data is formated in the following way
const int Ne = 33;
const int Ny = 50;

const double SolarMassInSeconds = 4.925491025543575903411922162094833998e-6;
const double YearInSeconds 		= 60*60*25*365.25;

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

// TODO: make it selectable by mode ?
// TODO: free memory from inside interpolants
AmplitudeCarrier::AmplitudeCarrier(int lmax_, int nmax_, std::string few_dir)
{
    printf("OMP THREADS: %d\n", omp_get_num_threads());
    lmax = lmax_;
    nmax = nmax_;

    amps = new struct waveform_amps;

    cout << "# Loading and interpolating the amplitude data (this will take a few seconds)" << endl;
    load_and_interpolate_amplitude_data(lmax, nmax, amps, few_dir);

}

void AmplitudeCarrier::dealloc()
{

    delete amps;

}


void Interp2DAmplitude(std::complex<double> *amplitude_out, double *p_arr, double *e_arr, int *l_arr, int *m_arr, int *n_arr, int num, int num_modes, AmplitudeCarrier *amps_carrier)
{

    struct waveform_amps *amps = amps_carrier->amps;
    int lmax = amps_carrier->lmax;
    int nmax = amps_carrier->nmax;

    complex<double> I(0.0, 1.0);

    //reduction (+:hwave)
    #ifdef __USE_OMP__
    #pragma omp parallel for collapse(2)
    #endif
    for (int i=0; i<num; i++){
    	for(int mode_i = 0; mode_i < num_modes; mode_i++){
            double p = p_arr[i];
            double e = e_arr[i];

            double y = log((p -2.*e - 2.1));

            int l = l_arr[mode_i]; int m = m_arr[mode_i]; int n = n_arr[mode_i];
			amplitude_out[i*num_modes + mode_i]= amps->re[l][m][n+nmax]->eval(y,e) + I*amps->im[l][m][n+nmax]->eval(y,e);

            //if ((l == 2) && (m ==0)) printf("%d %d %d %e %e\n", l, m, n, amps->re[l][m][n+nmax]->eval(y,e), amps->im[l][m][n+nmax]->eval(y,e));
        }
    }
}
