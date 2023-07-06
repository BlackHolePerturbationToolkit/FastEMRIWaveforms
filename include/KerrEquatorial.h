#ifndef DEDT_CHEB
#define DEDT_CHEB

/*
  Header of the Interpolated fluxes
*/

//! \file KerrEquatorial.h

// Define macro 

// Define type

// Declare prototype 
double edot_Cheby_full(const double a, const double e, const double r);
double pdot_Cheby_full(const double a, const double e, const double r);
void Jac(const double a, const double p, const double ecc, const double xi,
		  const double E, const double Lz, const double Q,
		  const double Edot, const double Lzdot, const double Qdot,
		  double & pdot, double & eccdot, double & xidot);

#endif // DEDT_CHEB
