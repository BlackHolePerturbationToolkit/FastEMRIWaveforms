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

#include <vector>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

using namespace std;

typedef vector<double> Vector;

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