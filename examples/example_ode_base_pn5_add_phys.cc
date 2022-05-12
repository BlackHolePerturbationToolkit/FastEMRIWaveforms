#include <math.h>
#include <stdio.h>
#include <algorithm>

#include "global.h"
#include "Utility.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>      // std::setprecision
#include <cstring>

#include "dIdt8H_5PNe10.h"
#include "ode.hh"

#define pn5_add_phys_Y
#define pn5_add_phys_citation1 Pn5_citation
#define pn5_add_phys_num_add_args 2
__deriv__
void pn5_add_phys(double* pdot, double* edot, double* Ydot,
                  double* Omega_phi, double* Omega_theta, double* Omega_r,
                  double epsilon, double a, double p, double e, double Y, double* additional_args)
{
    // evaluate ODEs
    double phys_factor1 = additional_args[0];
    double phys_factor2 = additional_args[1];

    // the frequency variables are pointers!
    double x = Y_to_xI(a, p, e, Y);
    KerrGeoCoordinateFrequencies(Omega_phi, Omega_theta, Omega_r, a, p, e, x);

    *Omega_phi = (*Omega_phi) * (1.0 + phys_factor2);

	int Nv = 10;
    int ne = 10;
    *pdot = epsilon * dpdt8H_5PNe10 (a, p, e, Y, Nv, ne) * (1.0 + phys_factor1);

    // needs adjustment for validity
    Nv = 10;
    ne = 8;
	*edot = epsilon * dedt8H_5PNe10 (a, p, e, Y, Nv, ne) * (1.0 + phys_factor1);

    Nv = 7;
    ne = 10;
    *Ydot = epsilon * dYdt8H_5PNe10 (a, p, e, Y, Nv, ne);

}