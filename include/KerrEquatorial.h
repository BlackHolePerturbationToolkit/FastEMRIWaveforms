#ifndef DEDT_CHEB
#define DEDT_CHEB

/*
  Header of the Interpolated fluxes
*/

//! \file KerrEquatorial.h

// Define macro 

// Define type

// Declare prototype 
double dEdt_Cheby(const double a, const double p, const double e, const double rISCO, const double ps);
double dLdt_Cheby(const double a, const double p, const double e, const double rISCO, const double ps);
double pdot_Cheby(const double a, const double p, const double e, const double rISCO, const double ps);
double edot_Cheby(const double a, const double p, const double e, const double rISCO, const double ps);
double edot_dspin_Cheby(const double a, const double p, const double e, const double rISCO, const double ps);
double pdot_dspin_Cheby(const double a, const double p, const double e, const double rISCO, const double ps);
double dOmegaPhi_dspin(const double a, const double p, const double e, const double rISCO, const double ps);
double dOmegaR_dspin(const double a, const double p, const double e, const double rISCO, const double ps);
double edot_Cheby_full(const double a, const double e, const double r);
double pdot_Cheby_full(const double a, const double e, const double r);
double dE_de_Equatorial(const double a, const double p, const double e);
double dL_de_Equatorial(const double a, const double p, const double e);
double dL_dp_Equatorial(const double a, const double p, const double e);
double dE_dp_Equatorial(const double a, const double p, const double e);
double pdot_from_fluxes(const double a, const double p, const double e, const double Edot, const double Ldot);
double edot_from_fluxes(const double a, const double p, const double e, const double Edot, const double Ldot);


class GenericKerrRadiation {
public:
  GenericKerrRadiation(const double semilatus, const double ecc, const double energy,
      const double angmom, const double carter, const double spin);
  double p_LSO(double csi);

  double Edot, Lzdot, Qdot;
  double radot, rpdot;
  double pdot, edot, idot, cosidot;
  void pei_FluxEvolution(double Edot, double Lzdot, double Qdot);

private:

  double Denom(const double r);
  double Numer1(const double r);
  double Numer2(const double r);
  double Numer3(const double r);

  
  double p, e, E, Lz, Q, kerr_a;
  // added dt time step and time variable
  double cosiota, siniotasqr, sini, ra, rp, Lovercosi, a_sm, ome, ope, ome2;


};

#endif // DEDT_CHEB
