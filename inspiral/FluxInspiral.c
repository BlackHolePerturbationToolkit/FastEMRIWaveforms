// Code to compute an eccentric flux driven insipral
// into a Schwarzschild black hole

// Definitions needed for Mathematicas CForm output
#define Power(x, y)     (pow((double)(x), (double)(y)))
#define Sqrt(x)         (sqrt((double)(x)))

#include <math.h>


void RHS(double *f, double p, double e, double Edot, double Ldot){
	
	double pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2));
	
	double edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) + 
       (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
     (e*(4*Power(e,2) - Power(-6 + p,2))*p));

	// Need to evaluate 4 different elliptic integrals here. Cache them first to avoid repeated calls.
	double EllipE 	= 0;
	double EllipK 	= 0;
	double EllipPi1 = 0;
	double EllipPi2 = 0;
	
	double Phi_phi_dot 	= 0;
	double Phi_r_dot 	= 0;
}

int main(){
	
	
	return 0;
}