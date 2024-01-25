// NIT_inspiral - code to rapidly compute extreme mass-ratio inspirals using self-force results
// Copyright (C) 2017  Niels Warburton
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

#ifndef __INTERPOLANT_H__
#define __INTERPOLANT_H__

#include <vector>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <stdbool.h>
#import <stdlib.h>

#define TPI_FAIL -1
#define TPI_SUCCESS 0

#define CHECK_RANGES
using namespace std;

typedef vector<double> Vector;

typedef struct array_t {
    double *vec;
    int n;
} array_yo;

class Interpolant{
	public:
		// 1D interpolation
		Interpolant(Vector x, Vector f);
		double eval(double x);

		// 2D interpolation
		Interpolant(Vector x, Vector y, Vector f);
		double eval(double x, double y);

		// Destructor
		~Interpolant();


	private:
		int interp_type;	// Set to 1 for 1D interpolation and 2 for 2D interpolation

		gsl_spline *spline;
		gsl_spline2d *spline2d;
		gsl_interp_accel *xacc;
		gsl_interp_accel *yacc;
};

class TensorInterpolant{
    public:
    double *coeff;
    int coeff_N;
    gsl_bspline_workspace **bw_out;   // Output: pointer to array of pointers to

    TensorInterpolant(Vector x, Vector y, Vector z, Vector flatten_coeff);
    double eval(double x, double y, double z);

    ~TensorInterpolant();
};

class TensorInterpolant2d{
    public:
    double *coeff;
    int coeff_N;
    gsl_bspline_workspace **bw_out;   // Output: pointer to array of pointers to

    TensorInterpolant2d(Vector x, Vector y, Vector flatten_coeff);
    double eval(double x, double y);

    ~TensorInterpolant2d();
};

/******************************* 1D functions *********************************/

int Interpolation_Setup_1D(
    double *xvec,                       // Input: knots: FIXME: knots are calculate internally, so shouldn't need to do that here
    int nx,                             // Input length of knots array xvec
    gsl_bspline_workspace **bw          // Output: Initialized B-spline workspace
);

int Bspline_basis_1D(
    double *B_array,                    // Output: the evaluated cubic B-splines 
                                        // B_i(x) for the knots defined in bw
    int n,                              // Input: length of Bx4_array
    gsl_bspline_workspace *bw,          // Input: Initialized B-spline workspace
    double x                            // Input: evaluation point
);

int Bspline_basis_3rd_derivative_1D(
    double *D3_B_array,                 // Output: the evaluated 3rd derivative of cubic
                                        // B-splines B_i(x) for the knots defined in bw
    int n,                              // Input: length of Bx4_array
    gsl_bspline_workspace *bw,          // Input: Initialized B-spline workspace
    double x                            // Input: evaluation point
);

int AssembleSplineMatrix_C(
    gsl_vector *xi,                     // Input: nodes xi
    gsl_matrix **phi,                   // Output: the matrix of spline coefficients
    gsl_vector **knots,                 // Output: the vector of knots including the endpoints
                                        // with multiplicity three
    gsl_bspline_workspace **bw          // Output: Bspline workspace
);

int SetupSpline1D(
    double *x,                          // Input: nodes
    double *y,                          // Input: data
    int n,                              // Input: number of data points
    double **c,                         // Output: spline coefficients
    gsl_bspline_workspace **bw          // Output: Bspline workspace
);

double EvaluateSpline1D(
    double *c,                          // Input: spline coefficients output by SetupSpline1D()
    gsl_bspline_workspace *bw,          // Input: Bspline workspace
    double xx                           // Input: evaluation point for spline
);

void TP_Interpolation_Setup_ND(
    array_yo *nodes,                       // Input: array of arrys containing the nodes
                                        // for each parameter space dimension
    int n,                              // Input: Dimensionality of parameter space
    gsl_bspline_workspace ***bw_out     // Output: pointer to array of pointers to
                                        // B-spline workspaces
);

int TP_Interpolation_ND(
    double *v,                          // Input: flattened TP spline coefficient array
    int n,                              // Input: length of TP spline coefficient array v
    double* X,                          // Input: parameter space evaluation point of length m
    int m,                              // Input: dimensionality of parameter space
    gsl_bspline_workspace **bw,         // Input: array of pointers to B-spline workspaces
    double *y                           // Output: TP spline evaluated at X
);
#endif // __INTERPOLANT_H__
