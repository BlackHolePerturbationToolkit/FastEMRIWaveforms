/*
 * Copyright (C) 2017, 2022 Michael Pürrer, Jonathan Blackman.
 *
 *  This file is part of TPI.
 *
 *  TPI is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  TPI is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with TPI.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <stdbool.h>
#import <stdlib.h>

#define TPI_FAIL -1
#define TPI_SUCCESS 0

#define CHECK_RANGES

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

/******************************* Generic code *********************************/

typedef struct array_t {
    double *vec;
    int n;
} array;

void TP_Interpolation_Setup_ND(
    array *nodes,                       // Input: array of arrys containing the nodes
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