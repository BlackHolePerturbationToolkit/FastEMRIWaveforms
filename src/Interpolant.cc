// Code to compute an eccentric flux driven insipral
// into a Schwarzschild black hole

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

#include <Interpolant.h>
#include <algorithm>

#include <iostream>


#ifdef DEBUG
#include <time.h>
#include <stdio.h>
#endif


/******************************* Generic code *********************************/

void TP_Interpolation_Setup_ND(
    array_yo *nodes,                     // Input: array of arrys containing the nodes
                                      // for each parameter space dimension
    int n,                            // Input: Dimensionality of parameter space
    gsl_bspline_workspace ***bw_out   // Output: pointer to array of pointers to
                                      // B-spline workspaces: The array of pointers
                                      // must be allocated
) {
    gsl_bspline_workspace **bw = *bw_out;

    for (int j=0; j<n; j++) {
        int nc = nodes[j].n + 2;
        // Setup cubic B-spline workspaces
        const size_t nbreak = nc-2;  // must have nbreak = n-2 for cubic splines
        bw[j] = gsl_bspline_alloc(4, nbreak);
        gsl_vector *breakpts = gsl_vector_alloc(nbreak);

        for (size_t i=0; i<nbreak; i++)
            gsl_vector_set(breakpts, i, nodes[j].vec[i]);

        gsl_bspline_knots(breakpts, bw[j]);
        gsl_vector_free(breakpts);
    }
}

int TP_Interpolation_ND(
    double *v,                    // Input: flattened TP spline coefficient array
    int n,                        // Input: length of TP spline coefficient array v
    double* X,                    // Input: parameter space evaluation point of length m
    int m,                        // Input: dimensionality of parameter space
    gsl_bspline_workspace **bw,   // Input: array of pointers to B-spline workspaces
    double *y                     // Output: TP spline evaluated at X
) {
// TODO: The B-spline basis functions could be cached for repeated evaluations on the same TP grid, but using different coefficients (belonging to different physical quantities)
// the final computation could be moved to Python:
// can we replicate the organization of the computation in Python here in C?

#ifdef CHECK_RANGES
    for (int j=0; j<m; j++) {
        gsl_vector* knots = bw[j]->knots;
        double x_min = gsl_vector_get(knots, 0);
        double x_max = gsl_vector_get(knots, knots->size - 1);
        if (X[j] < x_min || X[j] > x_max) {
            //fprintf(stderr, "Error in TP_Interpolation_ND: X[%d] = %g is outside of knots vector [%g, %g]!\n", j, X[j], x_min, x_max);
            return TPI_FAIL;
        }
    }
#endif

#ifdef DEBUG
    clock_t time1;
    clock_t time2;
    clock_t time3;
    time1 = clock();
#endif

    int nc[m];
    gsl_vector *B[m];
    size_t is[m]; // first non-zero spline
    size_t ie[m]; // last non-zero spline
    for (int j=0; j<m; j++) {
        // Dimensionality of coefficients for each dimension
        nc[j] = bw[j]->n;

        // Store nonzero cubic (order k=4) B-spline basis functions
        B[j] = gsl_vector_alloc(4);

        // Evaluate all potentially nonzero cubic B-spline basis functions at X
        // and store them in the array of vectors Bx[].
        // Since the B-splines are of compact support we only need to store a small
        // number of basis functions to avoid computing terms that would be zero anyway.
        gsl_bspline_eval_nonzero(X[j], B[j], &is[j], &ie[j], bw[j]);
    }

#ifdef DEBUG
    time2 = clock();
    printf("%0.6f seconds for bspline evals\n", ((float)time2 - (float)time1)/CLOCKS_PER_SEC);
#endif

    // This will hold the value of the TP spline interpolant
    // To compute it we need to calculate an m-dimensional sum over
    // spline coefficients and non-zero B-spline bases.
    double sum = 0;

    // Start logic of dynamic nested loop of depth m
    int max = 4; // upper bound of each nested loop
    int *slots = (int *) malloc(sizeof(int) * m); // m indices in range(0, 4)

    // Store the products of the first k bsplines, and the kth partial sums of the indices.
    // Prepend the identity so we can always index with [i-1].
    double *b_prod_hierarchy = (double *) malloc(sizeof(double) * (m+1));
    int *i_sum_hierarchy = (int *) malloc(sizeof(int) * (m+1));

    // Initialize the indices and current bspline products
    int idx_sum = 0;
    double product = 1;
    b_prod_hierarchy[0] = 1;
    i_sum_hierarchy[0] = 0;
    for (int i = 0; i < m; i++) {
        slots[i] = 0;
        product *= gsl_vector_get(B[i], 0);
        b_prod_hierarchy[i+1] = product;
        idx_sum = idx_sum * nc[i] + is[i];
        i_sum_hierarchy[i+1] = idx_sum;
    }

    // Loop over last index first, loop over first index last.
    int index;

    while (true) {
        // Add the current coefficient times the product of all current bsplines
        sum += v[ i_sum_hierarchy[m] ] * b_prod_hierarchy[m];

        // Update the slots to the next valid configuration
        slots[m-1]++;
        index = m-1;
        while (slots[index] == max) {
            // Overflow, we're done
            if (index == 0)
                goto cleanup;

            slots[index--] = 0;
            slots[index]++;
        }

        // Now update the index sums and bspline products for anything that was altered
        while (index < m) {
            b_prod_hierarchy[index+1] = b_prod_hierarchy[index] * gsl_vector_get(B[index], slots[index]);
            i_sum_hierarchy[index+1] = i_sum_hierarchy[index] * nc[index] + is[index] + slots[index];
            index++;
        }
    }

    cleanup:
#ifdef DEBUG
    time3 = clock();
    printf("%0.6f seconds for coef eval and sum\n", ((float)time3 - (float)time2)/CLOCKS_PER_SEC);
#endif

    for (int j=0; j<m; j++)
        gsl_vector_free(B[j]);
    free(slots);
    free(b_prod_hierarchy);
    free(i_sum_hierarchy);

    *y = sum;
    return TPI_SUCCESS;
}

int TP_Interpolation_N_slowD(
    double *v,                    // Input: flattened TP spline coefficient array
    int n,                        // Input: length of TP spline coefficient array v
    double* X,                    // Input: parameter space evaluation point of length m
    int m,                        // Input: dimensionality of parameter space
    gsl_bspline_workspace **bw,   // Input: array of pointers to B-spline workspaces
    double *y                     // Output: TP spline evaluated at X
) {

#ifdef CHECK_RANGES
    for (int j=0; j<m; j++) {
        gsl_vector* knots = bw[j]->knots;
        double x_min = gsl_vector_get(knots, 0);
        double x_max = gsl_vector_get(knots, knots->size - 1);
        if (X[j] < x_min || X[j] > x_max) {
            //fprintf(stderr, "Error in TP_Interpolation_ND: X[%d] = %g is outside of knots vector [%g, %g]!\n", j, X[j], x_min, x_max);
            return TPI_FAIL;
        }
    }
#endif

    int nc[m];
    gsl_vector *B[m];
    size_t is[m]; // first non-zero spline
    size_t ie[m]; // last non-zero spline
    for (int j=0; j<m; j++) {
        // Dimensionality of coefficients for each dimension
        nc[j] = bw[j]->n;

        // Store nonzero cubic (order k=4) B-spline basis functions
        B[j] = gsl_vector_alloc(4);

        // Evaluate all potentially nonzero cubic B-spline basis functions at X
        // and store them in the array of vectors Bx[].
        // Since the B-splines are of compact support we only need to store a small
        // number of basis functions to avoid computing terms that would be zero anyway.
        gsl_bspline_eval_nonzero(X[j], B[j], &is[j], &ie[j], bw[j]);
    }

    // This will hold the value of the TP spline interpolant
    // To compute it we need to calculate an m-dimensional sum over
    // spline coefficients and non-zero B-spline bases.
    double sum = 0;

    // Start logic of dynamic nested loop of depth m
    int max = 4; // upper bound of each nested loop
    int *slots = (int *) malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++)
        slots[i] = 0;
    int index = 0;
    while (true) {
        // TP spline computation for a single point in coefficient space

        // Now compute coefficient at desired parameters X
        // from e.g. C(x,y,z) = c_ijk * Beta_i * Bchi1_j * Bchi2_k
        // while summing over indices where the B-splines are nonzero.
        // Indices for individual dimensions are stored in the variable slots.

        // Compute starting indices for m-dimensional slice
        int ii[m];
        for (int j=0; j<m; j++)
            ii[j] = is[j] + slots[m-j-1];

        // Convert to linear indexing
        // For instance,    (ii*ncy + jj)*ncz + kk
        // is rewritten as: ii[0] * (ii[1] + ii[2]*nc[m-2]) * nc[m-1]
        int idx = 0;
        for (int j=0; j<m-1; j++) {
            idx += ii[j];
            idx *= nc[j+1];
        }
        idx += ii[m-1];

        double term = v[ idx ];
        for (int j=0; j<m; j++)
            term *= gsl_vector_get(B[j], slots[m-j-1]);
        sum += term;
        // Done doing TP spline computations for this point

        // Increment
        slots[0]++;

        // Carry
        while (slots[index] == max) {
            // Overflow, we're done
            if (index == m - 1)
                goto cleanup;

            slots[index++] = 0;
            slots[index]++;
        }

        index = 0;
    }

    cleanup:
    for (int j=0; j<m; j++)
        gsl_vector_free(B[j]);
    free(slots);

    *y = sum;
    return TPI_SUCCESS;
}


//Construct a 1D interpolant of f(x)
Interpolant::Interpolant(Vector x, Vector f){

	interp_type = 1;

	xacc = gsl_interp_accel_alloc();
    spline = gsl_spline_alloc (gsl_interp_cspline, x.size());

    gsl_spline_init (spline, &x[0], &f[0], x.size());

}

// Function that is called to evaluate the 1D interpolant
double Interpolant::eval(double x){
	return gsl_spline_eval(spline, x, xacc);
}

// Construct a 2D interpolant of f(x,y)
Interpolant::Interpolant(Vector x, Vector y, Vector f){

	interp_type = 2;

	// Create the interpolant
    const gsl_interp2d_type *T = gsl_interp2d_bicubic;

    const size_t nx = x.size(); /* number of x grid points */
    const size_t ny = y.size(); /* number of y grid points */

    double *za = (double *)malloc(nx * ny * sizeof(double));
    spline2d = gsl_spline2d_alloc(T, nx, ny);
    xacc = gsl_interp_accel_alloc();
    yacc = gsl_interp_accel_alloc();

	for(unsigned int i = 0; i < nx; i++){
		for(unsigned int j = 0; j < ny; j++){
	    	gsl_spline2d_set(spline2d, za, i, j, f[j*nx + i]);
		}
	}

	/* initialize interpolation */
	gsl_spline2d_init(spline2d, &x[0], &y[0], za, nx, ny);

}

// Function that is called to evaluate the 2D interpolant
double Interpolant::eval(double x, double y){
	return gsl_spline2d_eval(spline2d, x, y, xacc, yacc);
}

// Destructor
Interpolant::~Interpolant(){
	delete(xacc);
	if(interp_type == 1){
		delete(spline);
	}else{
		delete(spline2d);
		delete(yacc);
	}
}
