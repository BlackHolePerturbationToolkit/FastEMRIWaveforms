//x86_64-apple-darwin13.4.0-clang++ -undefined dynamic_lookup -Wl,-rpath,/Users/lorenzosperi/anaconda3/envs/few_env/lib -L/Users/lorenzosperi/anaconda3/envs/few_env/lib -Wl,-rpath,/Users/lorenzosperi/anaconda3/envs/few_env/lib -L/Users/lorenzosperi/anaconda3/envs/few_env/lib -Wl,-pie -Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -Wl,-rpath,/Users/lorenzosperi/anaconda3/envs/few_env/lib -L/Users/lorenzosperi/anaconda3/envs/few_env/lib -march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -isystem /Users/lorenzosperi/anaconda3/envs/few_env/include -D_FORTIFY_SOURCE=2 -isystem /Users/lorenzosperi/anaconda3/envs/few_env/include  -lgsl -lgslcblas -llapack -llapacke -lgomp -lhdf5 -lhdf5_hl -o main EllK_interp.cpp spline.cpp
#include <vector>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <iostream>
#include <chrono>
#include "spline.hpp"

// Assuming EllipticK is a function that takes a double and returns a double
// Define elliptic integrals that use Mathematica's conventions
double EllipticK(double k)
{
    gsl_sf_result result;
    // cout << "CHECK1" << endl;
    int status = gsl_sf_ellint_Kcomp_e(sqrt(k), 2e-16, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticK failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}


int main() {
    // Generate data points
    Vector x_values;
    Vector y_values;
    for (double x = 0; x <= 1; x += 0.0001) {
        x_values.push_back(x);
        
        
        y_values.push_back(EllipticK(x*x));
    }

    // Create a CubicSpline object
    // construct a cubic spline object
    CubicSpline *cb;
    cb = new CubicSpline(x_values, y_values);

    // check interpolation error on randomly chosen points
    std::vector<double> errors;
    for (int i = 0; i < 1000; i++)
    {
        double x = (double)rand() / RAND_MAX;
        
        auto start = std::chrono::high_resolution_clock::now();
        double y = EllipticK(x*x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken to evaluate EllipticK for x = " << x << ": " << elapsed.count() << " seconds" << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        double y_interp = cb->evaluate(x);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Time taken to evaluate interpolant for x = " << x << ": " << elapsed.count() << " seconds" << std::endl;
        
        double error = y - y_interp;
        errors.push_back(error);
        std::cout << "Interpolation error at x = " << x << ": " << error << std::endl;
    }

    // Calculate average error
    double sum = 0.0;
    for (double error : errors)
    {
        sum += error;
    }
    double average_error = sum / errors.size();
    std::cout << "Average interpolation error: " << average_error << std::endl;

    // Calculate maximum error
    double max_error = *std::max_element(errors.begin(), errors.end());
    std::cout << "Maximum interpolation error: " << max_error << std::endl;

    // Clean up
    delete cb;

    return 0;
}