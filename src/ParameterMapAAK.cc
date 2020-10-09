#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_sf_bessel.h>

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

    #ifdef __USE_OMP__
    #pragma omp parallel for
    #endif
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

CUDA_CALLABLE_MEMBER
double d_dot_product(const double *u,const double *v){
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

CUDA_CALLABLE_MEMBER
void d_cross(const double *u,const double *v,double *w){
  w[0] = u[1]*v[2]-u[2]*v[1];
  w[1] = u[2]*v[0]-u[0]*v[2];
  w[2] = u[0]*v[1]-u[1]*v[0];
}

CUDA_CALLABLE_MEMBER
double d_vec_norm(const double *u){
    return sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
}

CUDA_CALLABLE_MEMBER
void d_RotCoeff(double* rot, double* n, double* L, double* S, double* nxL, double* nxS,
                double iota,double theta_S,double phi_S,double theta_K,double phi_K,double alpha){

  n[0] = sin(theta_S)*cos(phi_S);
  n[1] = sin(theta_S)*sin(phi_S);
  n[2] = cos(theta_S);
  S[0] = sin(theta_K)*cos(phi_K);
  S[1] = sin(theta_K)*sin(phi_K);
  S[2] = cos(theta_K);
  L[0] = cos(iota)*sin(theta_K)*cos(phi_K)+sin(iota)*(sin(alpha)*sin(phi_K)-cos(alpha)*cos(theta_K)*cos(phi_K));
  L[1] = cos(iota)*sin(theta_K)*sin(phi_K)-sin(iota)*(sin(alpha)*cos(phi_K)+cos(alpha)*cos(theta_K)*sin(phi_K));
  L[2] = cos(iota)*cos(theta_K)+sin(iota)*cos(alpha)*sin(theta_K);
  d_cross(n,L,nxL);
  d_cross(n,S,nxS);

  double norm=d_vec_norm(nxL)*d_vec_norm(nxS);
  double dot,cosrot,sinrot;
  //gsl_blas_ddot(nxL,nxS,&dot);
  dot = d_dot_product(nxL,nxS);
  cosrot=dot/norm;
  //gsl_blas_ddot(L,nxS,&dot);
  dot = d_dot_product(L,nxS);
  sinrot=dot;
  //gsl_blas_ddot(S,nxL,&dot);
  dot = d_dot_product(S,nxL);
  sinrot-=dot;
  sinrot/=norm;

  printf("%e %e %e %e %e %e\n", n[0], n[1], n[2], S[0], S[1], S[2]);
  rot[0]=2.*cosrot*cosrot-1.;
  rot[1]=cosrot*sinrot;
  rot[2]=-rot[1];
  rot[3]=rot[0];
}

void waveform(double* hI, double* hII,
              double* tvec, double* evec, double* vvec,
              double* gimvec, double* Phivec, double* alpvec, double* nuvec, double* gimdotvec, double* OmegaPhi_spin_mapped_vec,
              double M_phys, double mu, double lam, double qS, double phiS, double qK, double phiK, double dist,
              int length, int nmodes, bool mich)
{

      #ifdef __CUDACC__
      __shared__ double rot_all[4 * NUM_THREADS];
      __shared__ double n_all[3 * NUM_THREADS];
      __shared__ double L_all[3 * NUM_THREADS];
      __shared__ double S_all[3 * NUM_THREADS];
      __shared__ double nxL_all[3 * NUM_THREADS];
      __shared__ double nxS_all[3 * NUM_THREADS];

      double* rot = rot_all[threadIdx.x * 4];
      double* n_rot = n_all[threadIdx.x * 3];
      double* L_rot = L_all[threadIdx.x * 3];
      double* S_rot = S_all[threadIdx.x * 3];
      double* nxL_rot = nxL_all[threadIdx.x * 3];
      double* nxS_rot = nxS_all[threadIdx.x * 3];

      #endif

      double coslam=cos(lam);
      double sinlam=sin(lam);
      double cosqS=cos(qS);
      double sinqS=sin(qS);
      double cosqK=cos(qK);
      double sinqK=sin(qK);
      double cosphiK=cos(phiK);
      double sinphiK=sin(phiK);
      double halfsqrt3=sqrt(3.)/2.;
      double mu_sec = mu * MTSUN_SI;
      double zeta=mu_sec/dist/Gpc; // M/D

      double up_ldc = (cosqS*sinqK*cos(phiS-phiK) - cosqK*sinqS);
        double dw_ldc = (sinqK*sin(phiS-phiK));
        double psi_ldc;
        if (dw_ldc != 0.0) {
          psi_ldc = -atan2(up_ldc, dw_ldc);
        }
        else {
      psi_ldc = 0.5*M_PI;
        }
        double c2psi_ldc=cos(2.*psi_ldc);
        double s2psi_ldc=sin(2.*psi_ldc);

      double FplusI=c2psi_ldc;
      double FcrosI=-s2psi_ldc;
      double FplusII=s2psi_ldc;
      double FcrosII=c2psi_ldc;

      #ifdef __USE_OMP__
      #pragma omp parallel for
      #endif
      for (int i = 0; i < length; i++)
      {

          #ifdef __CUDACC__
          #else

          double rot_temp[4];
          double n_temp[3];
          double L_temp[3];
          double S_temp[3];
          double nxL_temp[3];
          double nxS_temp[3];

          double* rot = &rot_temp[0];
          double* n_rot = &n_temp[0];
          double* L_rot = &L_temp[0];
          double* S_rot = &S_temp[0];
          double* nxL_rot = &nxL_temp[0];
          double* nxS_rot = &nxS_temp[0];

          #endif

          hI[i]=0.;
          hII[i]=0.;

          double t=tvec[i];
          double e=evec[i];
          double v=vvec[i];
          double gim=gimvec[i];
          double Phi=Phivec[i];
          double alp=alpvec[i];
          double nu=nuvec[i];
          double gimdot=gimdotvec[i];
          double OmegaPhi_spin_mapped = OmegaPhi_spin_mapped_vec[i];

          double cosalp=cos(alp);
          double sinalp=sin(alp);
          double cosqL=cosqK*coslam+sinqK*sinlam*cosalp;
          double sinqL=sqrt(1.-cosqL*cosqL);
          double phiLup=sinqK*sinphiK*coslam-cosphiK*sinlam*sinalp-cosqK*sinphiK*sinlam*cosalp;
          double phiLdown=sinqK*cosphiK*coslam+sinphiK*sinlam*sinalp-cosqK*cosphiK*sinlam*cosalp;
          double phiL=atan2(phiLup,phiLdown);
          double Ldotn=cosqL*cosqS+sinqL*sinqS*cos(phiL-phiS);
          double Ldotn2=Ldotn*Ldotn;
          double Sdotn=cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS);
          double betaup=-Sdotn+coslam*Ldotn;
          double betadown=sinqS*sin(phiK-phiS)*sinlam*cosalp+(cosqK*Sdotn-cosqS)/sinqK*sinlam*sinalp;
          double beta=atan2(betaup,betadown);
          double gam=2.*(gim+beta);
          double cos2gam=cos(gam);
          double sin2gam=sin(gam);

          double orbphs,cosorbphs,sinorbphs,FplusI,FcrosI,FplusII,FcrosII;
        if(mich){
          orbphs=2.*M_PI*t/YRSID_SI;
          cosorbphs=cos(orbphs-phiS);
          sinorbphs=sin(orbphs-phiS);
          double cosq=.5*cosqS-halfsqrt3*sinqS*cosorbphs;
          double phiw=orbphs+atan2(halfsqrt3*cosqS+.5*sinqS*cosorbphs,sinqS*sinorbphs);
          double psiup=.5*cosqK-halfsqrt3*sinqK*cos(orbphs-phiK)-cosq*(cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS));
          double psidown=.5*sinqK*sinqS*sin(phiK-phiS)-halfsqrt3*cos(orbphs)*(cosqK*sinqS*sin(phiS)-cosqS*sinqK*sin(phiK))-halfsqrt3*sin(orbphs)*(cosqS*sinqK*cos(phiK)-cosqK*sinqS*cos(phiS));
          double psi=atan2(psiup,psidown);
          double cosq1=.5*(1.+cosq*cosq);
          double cos2phi=cos(2.*phiw);
          double sin2phi=sin(2.*phiw);
          double cos2psi=cos(2.*psi);
          double sin2psi=sin(2.*psi);
          FplusI=cosq1*cos2phi*cos2psi-cosq*sin2phi*sin2psi;
          FcrosI=cosq1*cos2phi*sin2psi+cosq*sin2phi*cos2psi;
          FplusII=cosq1*sin2phi*cos2psi+cosq*cos2phi*sin2psi;
          FcrosII=cosq1*sin2phi*sin2psi-cosq*cos2phi*cos2psi;
        }

          double Amp=pow(OmegaPhi_spin_mapped*M_phys*MTSUN_SI,2./3.)*zeta;

          d_RotCoeff(rot, n_rot, L_rot, S_rot, nxL_rot, nxS_rot,
                   lam,qS,phiS,qK,phiK,alp);

          // TODO: CHECK this in terms of further parallelization
          for(int n=1;n<=nmodes;n++)
          {

              double fn,Doppler,nPhi;
              if(mich){
                fn=n*nu+gimdot/M_PI;
                Doppler=2.*M_PI*fn*AUsec*sinqS*cosorbphs;
                nPhi=n*Phi+Doppler;
              }
              else nPhi=n*Phi;

               double ne=n*e;
               double J0, J1, J2, J3, J4;
              #ifdef __CUDACC__

              if(n==1){ J0=-1.0*j1(ne); }
              else { J0 = jn(n-2, ne); }

              J1=jn(n-1, ne);
              J2=jn(n, ne);
              J3=jn(n+1,ne);
              J4=jn(n+2,ne);

              #else

              if(n==1){ J0=-1.0*gsl_sf_bessel_J1(ne); }
              else { J0 = gsl_sf_bessel_Jn(n-2, ne); }

              J1=gsl_sf_bessel_Jn(n-1, ne);
              J2=gsl_sf_bessel_Jn(n, ne);
              J3=gsl_sf_bessel_Jn(n+1,ne);
              J4=gsl_sf_bessel_Jn(n+2,ne);

              #endif

      double a=-n*Amp*(J0-2.*e*J1+2./n*J2+2.*e*J3-J4)*cos(nPhi);
      double b=-n*Amp*sqrt(1-e*e)*(J0-2.*J2+J4)*sin(nPhi);
      double c=2.*Amp*J2*cos(nPhi);
      double Aplus=-(1.+Ldotn2)*(a*cos2gam-b*sin2gam)+c*(1-Ldotn2);
      double Acros=2.*Ldotn*(b*cos2gam+a*sin2gam);


      // ----- rotate to NK wave frame -----
      double Aplusold=Aplus;
      double Acrosold=Acros;

      Aplus=Aplusold*rot[0]+Acrosold*rot[1];
      Acros=Aplusold*rot[2]+Acrosold*rot[3];
      // ----------

      double hnI, hnII;
      if(mich){
      	hnI=halfsqrt3*(FplusI*Aplus+FcrosI*Acros);
        hnII=halfsqrt3*(FplusII*Aplus+FcrosII*Acros);
      }
      else{
      	hnI=FplusI*Aplus+FcrosI*Acros;
        hnII=FplusII*Aplus+FcrosII*Acros;
      }

      hI[i]+=hnI;
      hII[i]+=hnII;
  }
    }
}
