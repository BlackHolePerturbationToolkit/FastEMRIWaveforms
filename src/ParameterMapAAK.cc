#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_multifit.h>

#include "ParameterMapAAK.hh"
#include "global.h"

#ifdef __USE_OMP__
#include "omp.h"
#endif


struct sol_par{

  double Omega_r;
  double Omega_theta;
  double Omega_phi;
  double M;
  double e;
  double iota;

};

CUDA_CALLABLE_MEMBER
double drdm(double v,double e,double Y,double q){
  double v2=v*v;
  double v3=v2*v;
  double e2=e*e;
  double e4=e2*e2;
  double Y2=Y*Y;
  double q2=q*q;
  double eq=(16. + 8.*(-3. + e2)*v2 - 16.*(-3. + e2)*q*v3*Y +
     8.*(33. + 4.*e2 - 3.*e4)*q*v3*v2*Y +
     v3*v3*(-351. + 132.*q2 + e2*(-135. + 21.*e2 + 5.*e4 + 2.*(7. + e2)*q2) +
        2.*(-204. + 13.*e2*(-3. + e2))*q2*Y2) +
     2.*v2*v2*(-45. + 3.*e4 + 4.*q2*(1. - 4.*Y2) + 2.*e2*q2*(1. + Y2)))/(16.*v);
  return eq;
}

CUDA_CALLABLE_MEMBER
double dtdm(double v,double e,double Y,double q){
  double v2=v*v;
  double v3=v2*v;
  double v4=v2*v2;
  double e2=e*e;
  double e4=e2*e2;
  double Y2=Y*Y;
  double q2=q*q;
  double eq=(2.*e4*(240. + v2*(-120. + v*(42.*(-27. + 4.*q2)*v + (-6567. + 1996.*q2)*v3 +
              48.*q*(8. + 77.*v2)*Y - 4.*q2*v*(90. + 1577.*v2)*Y2))) +
     e4*e2*(560. + v2*(-360. + 960.*q*v*Y + 8816.*q*v3*Y +
           v4*(-15565. + 24.*q2*(200. - 629.*Y2)) +
           v2*(-2742. + 80.*q2*(5. - 11.*Y2)))) -
     8.*e2*(-48. + v2*(8. - 64.*q*v*Y - 688.*q*v3*Y +
           2.*v2*(99. + 16.*q2*(-1. + 2.*Y2)) + v4*(1233. + 8.*q2*(-47. + 150.*Y2))))
       + 16.*(16. + v2*(24. + v2*(27.*(2. + 5.*v2) - 48.*q*v*Y +
              4.*q2*(2. - v2 + (-2. + 3.*v2)*Y2)))))/(256.*v4);
  return eq;
}

CUDA_CALLABLE_MEMBER
double dthetadm(double v,double e,double Y,double q){
  double v2=v*v;
  double v3=v2*v;
  double e2=e*e;
  double e4=e2*e2;
  double Y2=Y*Y;
  double q2=q*q;
  double eq=(16. + 8.*(3. + e2)*v2 - 16.*(3. + e2)*q*v3*Y -
     8.*(3. + e2)*(5. + 3.*e2)*q*v3*v2*Y +
     v3*v3*(135. - 54.*q2 + e2*(5.*(27. + 9.*e2 + e4) + 2.*(-38. + e2)*q2) +
        2.*(57. + 90.*e2 + 13.*e4)*q2*Y2) +
     2.*v2*v2*(27. + 3.*e4 + 2.*q2*(-1. + 7.*Y2) + 2.*e2*(9. + q2*(1. + Y2))))/(16.*v);
  return eq;
}

CUDA_CALLABLE_MEMBER
double dphidm(double v,double e,double Y,double q){
  double v2=v*v;
  double v3=v2*v;
  double e2=e*e;
  double e4=e2*e2;
  double Y2=Y*Y;
  double q2=q*q;
  double eq=(16. + 8.*(3. + e2)*v2 - 16.*q*v3*(-2. + (3. + e2)*Y) -
     8.*q*v3*v2*(-6. + 15.*Y + 3.*e4*Y + 2.*e2*(-4. + 7.*Y)) +
     2.*v2*v2*(27. + 3.*e4 + 2.*q2*(-1. + Y)*(1. + 7.*Y) + 2.*e2*(9. + q2*(1. + Y2))) +
     v3*v3*(5.*e4*e2 + e4*(45. + q2*(2. + 26.*Y2)) +
        e2*(135. + 4.*q2*(-19. + 5.*Y*(-7. + 9.*Y))) + 3.*(45. + 2.*q2*(-9. + Y*(-6. + 19.*Y)))))/(16.*v);
  return eq;
}

// ----- magnitude of azimuthal angular frequency for prograde/retrograde orbits -----
CUDA_CALLABLE_MEMBER
double OmegaPhi(double v, double e, double cosiota, double s, double M){

  double omegaphi;
  if(cosiota>0) omegaphi=dphidm(v,e,cosiota,s)/dtdm(v,e,cosiota,s)/M;
  else omegaphi=dphidm(v,e,-cosiota,-s)/dtdm(v,e,-cosiota,-s)/M;

  return omegaphi;

}

int sol_fun(const gsl_vector *x,void *p,gsl_vector *f){

  double Omega_r=((struct sol_par*)p)->Omega_r;
  double Omega_theta=((struct sol_par*)p)->Omega_theta;
  double Omega_phi=((struct sol_par*)p)->Omega_phi;
  double M=((struct sol_par*)p)->M;
  double e=((struct sol_par*)p)->e;
  double iota=((struct sol_par*)p)->iota;

  const double v_map=gsl_vector_get(x,0);
  const double M_map=gsl_vector_get(x,1);
  const double s_map=gsl_vector_get(x,2);

  double Omega_r_map,Omega_theta_map,Omega_phi_map;
  if(cos(iota)>0){
    Omega_r_map=drdm(v_map,e,cos(iota),s_map)/dtdm(v_map,e,cos(iota),s_map)/2./M_PI;
    Omega_theta_map=dthetadm(v_map,e,cos(iota),s_map)/dtdm(v_map,e,cos(iota),s_map)/2./M_PI;
    Omega_phi_map=dphidm(v_map,e,cos(iota),s_map)/dtdm(v_map,e,cos(iota),s_map)/2./M_PI;
  }
  else{
    Omega_r_map=drdm(v_map,e,-cos(iota),-s_map)/dtdm(v_map,e,-cos(iota),-s_map)/2./M_PI;
    Omega_theta_map=dthetadm(v_map,e,-cos(iota),-s_map)/dtdm(v_map,e,-cos(iota),-s_map)/2./M_PI;
    Omega_phi_map=-dphidm(v_map,e,-cos(iota),-s_map)/dtdm(v_map,e,-cos(iota),-s_map)/2./M_PI;
  }

  const double f0=Omega_r*M_map-Omega_r_map*M;
  const double f1=Omega_theta*M_map-Omega_theta_map*M;
  const double f2=Omega_phi*M_map-Omega_phi_map*M;

  gsl_vector_set(f,0,f0);
  gsl_vector_set(f,1,f1);
  gsl_vector_set(f,2,f2);

  return GSL_SUCCESS;

}


void ParMap(double map[],double Omega[], double p, double M, double s, double e, double iota){

  const gsl_multiroot_fsolver_type *sol_typ;
  gsl_multiroot_fsolver *sol;

  int status;
  size_t i=0;

  const size_t n=3;
  struct sol_par par={Omega[0],Omega[1],Omega[2],M,e,iota};
  gsl_multiroot_function f={&sol_fun,n,&par};

  double x0[n]={1./sqrt(p),M,s};
  gsl_vector *x=gsl_vector_alloc(n);

  gsl_vector_set(x,0,x0[0]);
  gsl_vector_set(x,1,x0[1]);
  gsl_vector_set(x,2,x0[2]);

  sol_typ=gsl_multiroot_fsolver_hybrids;
  sol=gsl_multiroot_fsolver_alloc(sol_typ,n);
  gsl_multiroot_fsolver_set(sol,&f,x);

  //print_state(i,sol);

  do{
    i++;
    status=gsl_multiroot_fsolver_iterate(sol);
    //print_state(i,sol);
    if(status) break;
    status=gsl_multiroot_test_residual(sol->f,1.e-6);
  }
  while(status==GSL_CONTINUE&&i<1000);

  //printf("status = %s\n",gsl_strerror(status));

  map[0]=gsl_vector_get(sol->x,0);
  map[1]=gsl_vector_get(sol->x,1);
  map[2]=gsl_vector_get(sol->x,2);

  gsl_multiroot_fsolver_free(sol);
  gsl_vector_free(x);

}

void ParMapVector(double* v_map, double* M_map, double* S_map, double* OmegaPhi, double* OmegaTheta, double* OmegaR,
                  double* p, double* e, double* iota, double M, double s, int length)
{

    for (int i = 0; i < length; i++)
    {
        double map[3];
        double Omega[3];
        Omega[0] = OmegaR[i];
        Omega[1] = OmegaTheta[i];
        Omega[2] = OmegaPhi[i];

        ParMap(map, Omega, p[i], M, s, e[i], iota[i]);

        v_map[i] = map[0];
        M_map[i] = map[1];
        S_map[i] = map[2];
    }

}
