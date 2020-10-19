#ifndef DIDT8H_5PNE10_H_
#define DIDT8H_5PNE10_H_

/*
  Header of the PostNewtonian fluxes (5PN, e^10; arbitrary inclinations)

  11th Feb. 2020; Sis

*/

//! \file dIdt8H_5PNe10.h

// Define macro 

// Define type

// Declare prototype 
double dEdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);
double dLdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);
double dCdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);

double dpdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);
double dedt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);
double dYdt8H_5PNe10 (const double q, const double p, const double e, const double Y, const int Nv, const int ne);

#endif // DIDT8H_5PNE10_H_
