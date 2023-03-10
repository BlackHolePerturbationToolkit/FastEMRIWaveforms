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
