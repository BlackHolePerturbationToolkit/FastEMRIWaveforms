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

#include <Interpolant.h>
#include <algorithm>


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
	
	// Create Vectors that contain the ordered, unique entries in x and y
	Vector xs = x;
	Vector ys = y;
	
	sort( xs.begin(), xs.end() );
	xs.erase( unique( xs.begin(), xs.end() ), xs.end() );
	
	sort( ys.begin(), ys.end() );
	ys.erase( unique( ys.begin(), ys.end() ), ys.end() );
	
	// Create the interpolant
    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
	
    const size_t nx = xs.size(); /* x grid points */
    const size_t ny = ys.size(); /* y grid points */
	
    double *za = (double *)malloc(nx * ny * sizeof(double));
    spline2d = gsl_spline2d_alloc(T, nx, ny);
    xacc = gsl_interp_accel_alloc();
    yacc = gsl_interp_accel_alloc();
	
	
    vector<double>::iterator it_x;
    vector<double>::iterator it_y;
	for(unsigned int i = 0; i < x.size(); i++){
		
	    it_x = find(xs.begin(), xs.end(), x[i]);
	    it_y = find(ys.begin(), ys.end(), y[i]);
		
		auto pos_x = distance(xs.begin(), it_x);
		auto pos_y = distance(ys.begin(), it_y);		

	    gsl_spline2d_set(spline2d, za, pos_x, pos_y, f[i]);
	}
	
	/* initialize interpolation */
	gsl_spline2d_init(spline2d, &xs[0], &ys[0], za, nx, ny);
	
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