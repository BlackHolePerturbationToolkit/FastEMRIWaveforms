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

#include "TensorProductInterpolation.h"

#ifdef DEBUG
#include <time.h>
#include <stdio.h>
#endif


/******************************* Generic code *********************************/

void TP_Interpolation_Setup_ND(
    array *nodes,                     // Input: array of arrys containing the nodes
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

/******************************* 1D functions *********************************/

// Initialize B-spline workspaces and knots
int Interpolation_Setup_1D(
    double *xvec,                       // Input: knots: FIXME: knots are calculate internally, so shouldn't need to do that here
    int nx,                             // Input length of knots array xvec
    gsl_bspline_workspace **bw          // Output: Initialized B-spline workspace
) {
    int ncx = nx + 2;

    // Setup cubic B-spline workspace
    const size_t nbreak_x = ncx-2;  // must have nbreak = n-2 for cubic splines

    if (*bw) {
        fprintf(stderr, "Error: Interpolation_Setup_1D(): B-spline workspace pointer should be NULL.\n");
        return TPI_FAIL;
    }
    *bw = gsl_bspline_alloc(4, nbreak_x);

    gsl_vector *breakpts_x = gsl_vector_alloc(nbreak_x);

    for (size_t i=0; i<nbreak_x; i++)
      gsl_vector_set(breakpts_x, i, xvec[i]);

    // gsl computes the knots from the given breakpoints (Python notation):
    //
    //   knots[:4] = breakpts_x[0]
    //   knots[4:-3] = breakpts_x[1:-1]
    //   knots[-3:] = breakpts_x[-1]
    //
    // where k is the spline order, k=4 for cubic.
    // Thus, len(knots) = len(breakpts_x) + 2*3.
    gsl_bspline_knots(breakpts_x, *bw);

    gsl_vector_free(breakpts_x);

    return (*bw)->n; // dimensions of the B-spline basis
}

// Evaluate the B-spline basis functions B_i(x) for all i at x.
// Here we specialize to cubic B-spline bases.
int Bspline_basis_1D(
    double *B_array,           // Output: the evaluated cubic B-splines
                               // B_i(x) for the knots defined in bw
    int n,                     // Input: length of Bx4_array
    gsl_bspline_workspace *bw, // Input: Initialized B-spline workspace
    double x                   // Input: evaluation point
) {
    double a = gsl_vector_get(bw->knots, 0);
    double b = gsl_vector_get(bw->knots, bw->knots->size - 1);
    if (x < a || x > b) {
        //fprintf(stderr, "Error: Bspline_basis_1D(): x: %g is outside of knots vector with bounds [%g, %g]!\n", x, a, b);
        return TPI_FAIL;
    }

    gsl_vector *B = gsl_vector_alloc(bw->n);
    gsl_bspline_eval(x, B, bw);

    for (size_t i=0; i<bw->n; i++)
        B_array[i] = gsl_vector_get(B, i);

    // FIXME: add gsl_vector_free(B);
    return TPI_SUCCESS;
}

// Evaluate the 3rd derivative of the B-spline basis functions B_i(x) for all i at x.
// Here we specialize to cubic B-spline bases.
int Bspline_basis_3rd_derivative_1D(
    double *D3_B_array,               // Output: the evaluated 3rd derivative of cubic
                                      // B-splines B_i(x) for the knots defined in bw
    int n,                            // Input: length of Bx4_array
    gsl_bspline_workspace *bw,        // Input: Initialized B-spline workspace
    double x                          // Input: evaluation point
) {
    double a = gsl_vector_get(bw->knots, 0);
    double b = gsl_vector_get(bw->knots, bw->knots->size - 1);
    if (x < a || x > b) {
        //fprintf(stderr, "Error: Bspline_basis_3rd_derivative_1D(): x: %g is outside of knots vector with bounds [%g, %g]!\n", x, a, b);
        return TPI_FAIL;
    }

    size_t n_deriv = 3;
    gsl_matrix *D3_B = gsl_matrix_alloc(bw->n, n_deriv+1);
    gsl_bspline_deriv_eval(x, n_deriv, D3_B, bw);

    for (size_t i=0; i<bw->n; i++)
        D3_B_array[i] = gsl_matrix_get(D3_B, i, 3); // just copy the 3rd derivative

    return TPI_SUCCESS;
}

// Functions below for 1D cubic spline interpolation with "not-a-knot" boundary conditions.

int AssembleSplineMatrix_C(gsl_vector *xi, gsl_matrix **phi, gsl_vector **knots, gsl_bspline_workspace **bw) {
    // Assemble spline matrix for cubic spline with not-a-knot boundary conditions
    int ret = 0;

    if (xi == NULL)
        return TPI_FAIL;
    if (*phi != NULL)
        return TPI_FAIL;
    if (*knots != NULL)
        return TPI_FAIL;

    // Set up B-spline workspace and knots
    int n = xi->size;
    double *x = gsl_vector_ptr(xi, 0);
    *knots = gsl_vector_alloc(n + 2*3);
    for (int i=0; i < 3; i++)
        gsl_vector_set(*knots, i, x[0]);
    for (int i=0; i < n; i++)
        gsl_vector_set(*knots, i + 3, x[i]);
    for (int i=n + 3; i < n+ 2*3; i++)
        gsl_vector_set(*knots, i, x[n-1]);

    // for (int i=0; i < n + 2*3; i++)
    //     fprintf(stderr, "%d: %g\n", i, gsl_vector_get(*knots, i));

    // Calculate the matrix of all B-spline basis functions at all gridpoints xi
    int N = n + 2;
    // fprintf(stderr, "%d %d\n", n, N);
    *phi = gsl_matrix_alloc(N, N);
    gsl_matrix_set_zero(*phi);
    double *B_array = malloc(N*sizeof(double)); // temporary storage

    // Initialize B-splines
    if (*bw != NULL)
        return TPI_FAIL;
    Interpolation_Setup_1D(gsl_vector_ptr(xi, 0), xi->size, bw);
    for (int i=1; i < N-1; i++) {
        // compare to Mma / Python codes
        // Compute B-spline basis function at point xi[i]
        // except for the first and last row which we will fill later
        ret = Bspline_basis_1D(B_array, N, *bw, x[i-1]);
        if (ret != TPI_SUCCESS)
            return ret;
        for (int j=0; j < N; j++)
            gsl_matrix_set(*phi, i, j, B_array[j]);
    }

    // Prepare not-a-knot conditions from continuity of the 3rd derivative 
    // at the 2nd and the penultimate gridpoint

    // Impose these conditions in-between the first and last two points:
    double xi12mean  = (x[0]  + x[1]) / 2.;
    double xi23mean  = (x[1]  + x[2]) / 2.;
    double xim32mean = (x[n-3] + x[n-2]) / 2.;
    double xim21mean = (x[n-2] + x[n-1]) / 2.;

    // Coefficients for first and last rows
    double *r1 = malloc(N*sizeof(double));
    double *rm1 = malloc(N*sizeof(double));

    ret = Bspline_basis_3rd_derivative_1D(r1, N, *bw, xi12mean);
    ret |= Bspline_basis_3rd_derivative_1D(B_array, N, *bw, xi23mean);
    if (ret != TPI_SUCCESS)
        return ret;
    for (int j=0; j<N; j++)
        gsl_matrix_set(*phi, 0, j, r1[j] - B_array[j]);

    ret = Bspline_basis_3rd_derivative_1D(rm1, N, *bw, xim32mean);
    ret |= Bspline_basis_3rd_derivative_1D(B_array, N, *bw, xim21mean);
    if (ret != TPI_SUCCESS)
        return ret;
    for (int j=0; j<N; j++)
        gsl_matrix_set(*phi, N-1, j, rm1[j] - B_array[j]);

    // for (int i=0; i < N; i++)
    //   for (int j=0; j < N; j++)
    //     fprintf(stderr, "m(%d,%d) = %g\n", i, j, gsl_matrix_get(*phi, i, j));

    free(B_array);
    free(r1);
    free(rm1);
    return TPI_SUCCESS;
}

int SetupSpline1D(double *x, double *y, int n, double **c, gsl_bspline_workspace **bw) {
    int N = n + 2;

    // Compute spline matrix
    gsl_vector *xi = gsl_vector_alloc(n);
    for (int i=0; i<n; i++)
        gsl_vector_set(xi, i, x[i]);

    gsl_matrix *phi = NULL;
    gsl_vector *knots = NULL;
    int ret = AssembleSplineMatrix_C(xi, &phi, &knots, bw);
    if (ret != TPI_SUCCESS)
        return ret;

    // Set up RHS
    gsl_vector *F0 = gsl_vector_alloc(N); // to contain zero-padded ordinates (y-values)
    for (int i=0; i<n; i++)
        gsl_vector_set(F0, i+1, y[i]);

    // printf ("F0 = \n");
    // gsl_vector_fprintf(stdout, F0, "%g");

    // Solve spline system
    gsl_permutation *p = gsl_permutation_alloc(N);
    gsl_vector *d = gsl_vector_alloc(N);
    int signum;
    ret = gsl_linalg_LU_decomp(phi, p, &signum); // LU decomposition of phi is stored in the same matrix
    ret = gsl_linalg_LU_solve(phi, p, F0, d);

    // printf ("d = \n");
    // gsl_vector_fprintf(stdout, d, "%g");

    gsl_permutation_free(p);
    gsl_vector_free(F0);
    gsl_matrix_free(phi);
    gsl_vector_free(xi);
    gsl_vector_free(knots);
    
    // Store coefficients in output array
    if (*c != NULL)
        return TPI_FAIL;
    *c = malloc(N*sizeof(double));
    for (int i=0; i<N; i++)
        (*c)[i] = gsl_vector_get(d, i);
    gsl_vector_free(d);

    return TPI_SUCCESS;
}

double EvaluateSpline1D(double *c, gsl_bspline_workspace *bw, double xx) {
    // Store nonzero cubic (order k=4) B-spline basis functions
    gsl_vector *Bx = gsl_vector_alloc(4);
    double sum = 0;
    size_t is; // first non-zero spline
    size_t ie; // last non-zero spline
    gsl_bspline_eval_nonzero(xx, Bx, &is, &ie, bw);

    // Now compute coefficient at desired parameters from C(x) = c_i * B_i
    // summing over indices i where the B-splines are nonzero.
    for (int i=0; i<4; i++)
        sum += c[is + i] * gsl_vector_get(Bx, i);
    return sum;
}
