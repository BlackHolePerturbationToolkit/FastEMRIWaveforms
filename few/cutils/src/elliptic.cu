// Code to compute elliptic integrals in CUDA

// Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
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

// WARNING: THIS FILE IS NOT FINALIZED, VERIFIED, OR CURRENTLY USED. 

#include "global.h"
#include "stdio.h"
#include <cfloat>
#include <assert.h>
#include "elliptic.hh"

#define NUM_THREADS 256

/*
__device__
static _Tp P[] = {
  1.53552577301013293365E-4,
  2.50888492163602060990E-3,
  8.68786816565889628429E-3,
  1.07350949056076193403E-2,
  7.77395492516787092951E-3,
  7.58395289413514708519E-3,
  1.15688436810574127319E-2,
  2.18317996015557253103E-2,
  5.68051945617860553470E-2,
  4.43147180560990850618E-1,
  1.00000000000000000299E0
};
__device__
static _Tp Q[] = {
  3.27954898576485872656E-5,
  1.00962792679356715133E-3,
  6.50609489976927491433E-3,
  1.68862163993311317300E-2,
  2.61769742454493659583E-2,
  3.34833904888224918614E-2,
  4.27180926518931511717E-2,
  5.85936634471101055642E-2,
  9.37499997197644278445E-2,
  2.49999999999888314361E-1
};

__device__ __host__ _Tp polevl(_Tp x, _Tp *coeffs, int N)
{
    _Tp ans;
    int i;
    const _Tp *p;

    p = coeffs;
    ans = *p++;
    i = N;

    do
	ans = ans * x + *p++;
    while (--i);

    return (ans);
}

__device__ __host__ _Tp ellpe(_Tp x)
{
    x = 1.0 - x;
    if (x <= 0.0) {
	if (x == 0.0)
	    return (1.0);
	assert(0);
    }
    if (x > 1.0) {
        return ellpe(1.0 - 1/x) * sqrt(x);
    }
    return (polevl(x, P, 10) - log(x) * (x * polevl(x, Q, 9)));
}

__device__ static _Tp PK[] =
{
 1.37982864606273237150E-4,
 2.28025724005875567385E-3,
 7.97404013220415179367E-3,
 9.85821379021226008714E-3,
 6.87489687449949877925E-3,
 6.18901033637687613229E-3,
 8.79078273952743772254E-3,
 1.49380448916805252718E-2,
 3.08851465246711995998E-2,
 9.65735902811690126535E-2,
 1.38629436111989062502E0
};

__device__ static _Tp QK[] =
{
 2.94078955048598507511E-5,
 9.14184723865917226571E-4,
 5.94058303753167793257E-3,
 1.54850516649762399335E-2,
 2.39089602715924892727E-2,
 3.01204715227604046988E-2,
 3.73774314173823228969E-2,
 4.88280347570998239232E-2,
 7.03124996963957469739E-2,
 1.24999999999870820058E-1,
 4.99999999999999999821E-1
};
*/

// TODO: change _Tp back to template
template<typename _Tp>
__device__   _Tp

__ellint_rd(_Tp __x, _Tp __y, _Tp __z)
{
  const _Tp __eps = FLT_EPSILON;
  const _Tp __errtol = pow(__eps / _Tp(8), _Tp(1) / _Tp(6));
      const _Tp __c0 = _Tp(1) / _Tp(4);
      const _Tp __c1 = _Tp(3) / _Tp(14);
      const _Tp __c2 = _Tp(1) / _Tp(6);
      const _Tp __c3 = _Tp(9) / _Tp(22);
      const _Tp __c4 = _Tp(3) / _Tp(26);
      _Tp __xn = __x;
      _Tp __yn = __y;
      _Tp __zn = __z;
      _Tp __sigma = _Tp(0);
      _Tp __power4 = _Tp(1);
      _Tp __mu;
      _Tp __xndev, __yndev, __zndev;
      const unsigned int __max_iter = 100;
      for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
        {
          __mu = (__xn + __yn + _Tp(3) * __zn) / _Tp(5);
          __xndev = (__mu - __xn) / __mu;
          __yndev = (__mu - __yn) / __mu;
          __zndev = (__mu - __zn) / __mu;
          _Tp __epsilon = max(abs(__xndev), abs(__yndev));
          __epsilon = max(__epsilon, abs(__zndev));
          if (__epsilon < __errtol)
            break;
          _Tp __xnroot = sqrt(__xn);
          _Tp __ynroot = sqrt(__yn);
          _Tp __znroot = sqrt(__zn);
          _Tp __lambda = __xnroot * (__ynroot + __znroot)
                       + __ynroot * __znroot;
          __sigma += __power4 / (__znroot * (__zn + __lambda));
          __power4 *= __c0;
          __xn = __c0 * (__xn + __lambda);
          __yn = __c0 * (__yn + __lambda);
          __zn = __c0 * (__zn + __lambda);
        }
      // Note: __ea is an SPU badname.
      _Tp __eaa = __xndev * __yndev;
      _Tp __eb = __zndev * __zndev;
      _Tp __ec = __eaa - __eb;
      _Tp __ed = __eaa - _Tp(6) * __eb;
      _Tp __ef = __ed + __ec + __ec;
      _Tp __s1 = __ed * (-__c1 + __c3 * __ed
                               / _Tp(3) - _Tp(3) * __c4 * __zndev * __ef
                               / _Tp(2));
      _Tp __s2 = __zndev
               * (__c2 * __ef
                + __zndev * (-__c3 * __ec - __zndev * __c4 - __eaa));
      return _Tp(3) * __sigma + __power4 * (_Tp(1) + __s1 + __s2)
                                    / (__mu * sqrt(__mu));
    }

template<typename _Tp>
__device__   _Tp
__ellint_rc(_Tp __x, _Tp __y)
{
      const _Tp __c0 = _Tp(1) / _Tp(4);
      const _Tp __c1 = _Tp(1) / _Tp(7);
      const _Tp __c2 = _Tp(9) / _Tp(22);
      const _Tp __c3 = _Tp(3) / _Tp(10);
      const _Tp __c4 = _Tp(3) / _Tp(8);
      _Tp __xn = __x;
      _Tp __yn = __y;
      const _Tp __eps = FLT_EPSILON;
      const _Tp __errtol = pow(__eps / _Tp(30), _Tp(1) / _Tp(6));
      _Tp __mu;
      _Tp __sn;
      const unsigned int __max_iter = 100;
      for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
        {
          __mu = (__xn + _Tp(2) * __yn) / _Tp(3);
          __sn = (__yn + __mu) / __mu - _Tp(2);
          if (abs(__sn) < __errtol)
            break;
          const _Tp __lambda = _Tp(2) * sqrt(__xn) * sqrt(__yn)
                         + __yn;
          __xn = __c0 * (__xn + __lambda);
          __yn = __c0 * (__yn + __lambda);
        }
      _Tp __s = __sn * __sn
              * (__c3 + __sn*(__c1 + __sn * (__c4 + __sn * __c2)));
      return (_Tp(1) + __s) / sqrt(__mu);
    }

template<typename _Tp>
__device__   _Tp
__ellint_rf(_Tp __x, _Tp __y, _Tp __z)
{
  const _Tp __c0 = _Tp(1) / _Tp(4);
  const _Tp __c1 = _Tp(1) / _Tp(24);
  const _Tp __c2 = _Tp(1) / _Tp(10);
  const _Tp __c3 = _Tp(3) / _Tp(44);
  const _Tp __c4 = _Tp(1) / _Tp(14);
  _Tp __xn = __x;
  _Tp __yn = __y;
  _Tp __zn = __z;
  const _Tp __eps = FLT_EPSILON;
  const _Tp __errtol = pow(__eps, _Tp(1) / _Tp(6));
  _Tp __mu;
  _Tp __xndev, __yndev, __zndev;
  const unsigned int __max_iter = 100;
  for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
    {
      __mu = (__xn + __yn + __zn) / _Tp(3);
      __xndev = 2 - (__mu + __xn) / __mu;
      __yndev = 2 - (__mu + __yn) / __mu;
      __zndev = 2 - (__mu + __zn) / __mu;
      _Tp __epsilon = max(abs(__xndev), abs(__yndev));
      __epsilon = max(__epsilon, abs(__zndev));
      if (__epsilon < __errtol)
        break;
      const _Tp __xnroot = sqrt(__xn);
      const _Tp __ynroot = sqrt(__yn);
      const _Tp __znroot = sqrt(__zn);
      const _Tp __lambda = __xnroot * (__ynroot + __znroot)
                         + __ynroot * __znroot;
      __xn = __c0 * (__xn + __lambda);
      __yn = __c0 * (__yn + __lambda);
      __zn = __c0 * (__zn + __lambda);
    }
  const _Tp __e2 = __xndev * __yndev - __zndev * __zndev;
  const _Tp __e3 = __xndev * __yndev * __zndev;
  const _Tp __s  = _Tp(1) + (__c1 * __e2 - __c2 - __c3 * __e3) * __e2
           + __c4 * __e3;
  return __s / sqrt(__mu);
}

template<typename _Tp>
__device__   _Tp
__comp_ellint_2(_Tp __k)
{
      const _Tp __kk = __k * __k;
      return __ellint_rf(_Tp(0), _Tp(1) - __kk, _Tp(1))
           - __kk * __ellint_rd(_Tp(0), _Tp(1) - __kk, _Tp(1)) / _Tp(3);
}



template<typename _Tp>
__device__   _Tp
    __ellint_rj(_Tp __x, _Tp __y, _Tp __z, _Tp __p)
    {
          const _Tp __c0 = _Tp(1) / _Tp(4);
          const _Tp __c1 = _Tp(3) / _Tp(14);
          const _Tp __c2 = _Tp(1) / _Tp(3);
          const _Tp __c3 = _Tp(3) / _Tp(22);
          const _Tp __c4 = _Tp(3) / _Tp(26);
          _Tp __xn = __x;
          _Tp __yn = __y;
          _Tp __zn = __z;
          _Tp __pn = __p;
          _Tp __sigma = _Tp(0);
          _Tp __power4 = _Tp(1);
          const _Tp __eps = FLT_EPSILON;
          const _Tp __errtol = pow(__eps / _Tp(8), _Tp(1) / _Tp(6));
          _Tp __lambda, __mu;
          _Tp __xndev, __yndev, __zndev, __pndev;
          const unsigned int __max_iter = 100;
          for (unsigned int __iter = 0; __iter < __max_iter; ++__iter)
            {
              __mu = (__xn + __yn + __zn + _Tp(2) * __pn) / _Tp(5);
              __xndev = (__mu - __xn) / __mu;
              __yndev = (__mu - __yn) / __mu;
              __zndev = (__mu - __zn) / __mu;
              __pndev = (__mu - __pn) / __mu;
              _Tp __epsilon = max(abs(__xndev), abs(__yndev));
              __epsilon = max(__epsilon, abs(__zndev));
              __epsilon = max(__epsilon, abs(__pndev));
              if (__epsilon < __errtol)
                break;
              const _Tp __xnroot = sqrt(__xn);
              const _Tp __ynroot = sqrt(__yn);
              const _Tp __znroot = sqrt(__zn);
              const _Tp __lambda = __xnroot * (__ynroot + __znroot)
                                 + __ynroot * __znroot;
              const _Tp __alpha1 = __pn * (__xnroot + __ynroot + __znroot)
                                + __xnroot * __ynroot * __znroot;
              const _Tp __alpha2 = __alpha1 * __alpha1;
              const _Tp __beta = __pn * (__pn + __lambda)
                                      * (__pn + __lambda);
              __sigma += __power4 * __ellint_rc(__alpha2, __beta);
              __power4 *= __c0;
              __xn = __c0 * (__xn + __lambda);
              __yn = __c0 * (__yn + __lambda);
              __zn = __c0 * (__zn + __lambda);
              __pn = __c0 * (__pn + __lambda);
            }
          // Note: __ea is an SPU badname.
          _Tp __eaa = __xndev * (__yndev + __zndev) + __yndev * __zndev;
          _Tp __eb = __xndev * __yndev * __zndev;
          _Tp __ec = __pndev * __pndev;
          _Tp __e2 = __eaa - _Tp(3) * __ec;
          _Tp __e3 = __eb + _Tp(2) * __pndev * (__eaa - __ec);
          _Tp __s1 = _Tp(1) + __e2 * (-__c1 + _Tp(3) * __c3 * __e2 / _Tp(4)
                            - _Tp(3) * __c4 * __e3 / _Tp(2));
          _Tp __s2 = __eb * (__c2 / _Tp(2)
                   + __pndev * (-__c3 - __c3 + __pndev * __c4));
          _Tp __s3 = __pndev * __eaa * (__c2 - __pndev * __c3)
                   - __c2 * __pndev * __ec;
          return _Tp(3) * __sigma + __power4 * (__s1 + __s2 + __s3)
                                             / (__mu * sqrt(__mu));
        }


//__device__ static fod C1 = 1.3862943611198906188E0;

template<typename _Tp>
__device__   _Tp
__comp_ellint_3(_Tp __k, _Tp __nu)
{
      const _Tp __kk = __k * __k;
      return __ellint_rf(_Tp(0), _Tp(1) - __kk, _Tp(1))
           + __nu
           * __ellint_rj(_Tp(0), _Tp(1) - __kk, _Tp(1), _Tp(1) - __nu)
           / _Tp(3);
}

template<typename _Tp>
__device__   _Tp
    __ellint_3(_Tp __k, _Tp __nu, _Tp __phi)
    {
          //  Reduce phi to -pi/2 < phi < +pi/2.
          const int __n = floorf(__phi / PI
                                   + _Tp(0.5L));
          const _Tp __phi_red = __phi
                              - __n * PI;
          const _Tp __kk = __k * __k;
          const _Tp __s = sin(__phi_red);
          const _Tp __ss = __s * __s;
          const _Tp __sss = __ss * __s;
          const _Tp __c = cos(__phi_red);
          const _Tp __cc = __c * __c;
          const _Tp __Pi = __s
                         * __ellint_rf(__cc, _Tp(1) - __kk * __ss, _Tp(1))
                         + __nu * __sss
                         * __ellint_rj(__cc, _Tp(1) - __kk * __ss, _Tp(1),
                                       _Tp(1) - __nu * __ss) / _Tp(3);
          if (__n == 0)
            return __Pi;
          else
            return __Pi + _Tp(2) * __n * __comp_ellint_3(__k, __nu);
        }

template<typename _Tp>
__device__   _Tp
    __ellint_2(_Tp __k, _Tp __phi)
    {
          //  Reduce phi to -pi/2 < phi < +pi/2.
          const int __n = floor(__phi / PI
                                   + _Tp(0.5L));
          const _Tp __phi_red = __phi
                              - __n * PI;
          const _Tp __kk = __k * __k;
          const _Tp __s = sin(__phi_red);
          const _Tp __ss = __s * __s;
          const _Tp __sss = __ss * __s;
          const _Tp __c = cos(__phi_red);
          const _Tp __cc = __c * __c;
          const _Tp __E = __s
                        * __ellint_rf(__cc, _Tp(1) - __kk * __ss, _Tp(1))
                        - __kk * __sss
                        * __ellint_rd(__cc, _Tp(1) - __kk * __ss, _Tp(1))
                        / _Tp(3);
          if (__n == 0)
            return __E;
          else
            return __E + _Tp(2) * __n * __comp_ellint_2(__k);
        }

/*

__device__ __host__ _Tp ellpk(_Tp x)
{

if( (x < 0.0) || (x > 1.0) )
	{
	//mtherr( "ellpk", DOMAIN );
	return( 0.0 );
	}

if( x > MACHEP )
	{
	return( polevl(x,PK,10) - log(x) * polevl(x,QK,10) );
	}
else
	{
	if( x == 0.0 )
		{
		//mtherr( "ellpk", SING );
		return( MAXNUM );
		}
	else
		{
		return( C1 - 0.5 * log(x) );
		}
	}
}
*/
template<typename _Tp>
__device__   _Tp
     __comp_ellint_1(const _Tp __k)
     {
         return __ellint_rf(_Tp(0), _Tp(1) - __k * __k, _Tp(1));
     }

template<typename _Tp>
__device__ _Tp
     __ellint_1(const _Tp __k, const _Tp __phi)
     {
           //  Reduce phi to -pi/2 < phi < +pi/2.
           const int __n = floor(__phi / PI
                                    + _Tp(0.5L));
           const _Tp __phi_red = __phi
                               - __n * PI;

           const _Tp __s = sin(__phi_red);
           const _Tp __c = cos(__phi_red);

           const _Tp __F = __s
                         * __ellint_rf(__c * __c,
                                 _Tp(1) - __k * __k * __s * __s, _Tp(1));

          if (__n == 0)
             return __F;
          else
            return __F + _Tp(2) * __n * __comp_ellint_1(__k);
}

/*
* Complete Elliptic Integral of the first kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticK(const _Tp m)
{
    const _Tp __k = sqrt(m);
    return __comp_ellint_1(__k);
}

/*
* Incomplete Elliptic Integral of the first kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticF(const _Tp phi, const _Tp m)
{
    const _Tp __k = sqrt(m);
    return __ellint_1(__k, phi);
}

/*
* Incomplete/Complete Elliptic Integral of the second kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticE(const _Tp phi, const _Tp m)
{
    const _Tp __k = sqrt(m);
    if (phi == PI/2)
        return __comp_ellint_2(__k);

    return __ellint_2(__k, phi);
}

/*
* Incomplete/Complete Elliptic Integral of the third kind
*
*/
template<typename _Tp>
__device__ _Tp
EllipticPi(_Tp n, _Tp phi, _Tp m)
{
    const _Tp __k = sqrt(m);
    if (phi == PI/2){

        return __comp_ellint_3(__k, n);
    }

    return __ellint_3(__k, n, phi);
}

template<typename _Tp>
__device__ _Tp
Get_Phi_osc(_Tp p, _Tp e)
{

  return 4*sqrt(p/(-6 + 2*e + p))*EllipticK((4*e)/(-6 + 2*e + p));

}

template<typename _Tp>
__device__ _Tp
Get_T_osc(_Tp p, _Tp e)
{

  return (2*p*sqrt((-4*pow(e,2) + pow(-2 + p,2))*(-6 + 2*e + p))*EllipticE(PI/2, (4*e)/(-6 + 2*e + p)))/((1 - pow(e,2))*(-4 + p)) -
     (2*sqrt(-4*pow(e,2) + pow(-2 + p,2))*p*EllipticK((4*e)/(-6 + 2*e + p)))/((1 - pow(e,2))*sqrt(-6 + 2*e + p)) -
     (4*sqrt(-4*pow(e,2) + pow(-2 + p,2))*(8*(1 - pow(e,2)) + (1 + 3*pow(e,2) - p)*p)*EllipticPi((-2*e)/(1 - e),PI/2.,(4*e)/(-6 + 2*e + p)))/
      ((1 - e)*(1 - pow(e,2))*(-4 + p)*sqrt(-6 + 2*e + p)) + (16*sqrt(-4*pow(e,2) + pow(-2 + p,2))*EllipticPi((4*e)/(-2 + 2*e + p),PI/2.,(4*e)/(-6 + 2*e + p)))/
      (sqrt(-6 + 2*e + p)*(-2 + 2*e + p));
}

template<typename _Tp>
__device__ _Tp
Get_V0(_Tp p, _Tp e, _Tp xi)
{
  return Get_Phi_osc(p, e)/(2.*PI)*(xi - PI) - 2*sqrt(p/(-6 + 2*e + p))*EllipticF(xi/2. - PI/2.,(4*e)/(-6 + 2*e + p));
}

template<typename _Tp>
__device__ _Tp
Get_U0(_Tp p, _Tp e, _Tp xi)
{
  return Get_T_osc(p, e)/(2.*PI)*(xi - PI) - (-((p*sqrt((-4*pow(e,2) + pow(-2 + p,2))*(-6 + 2*e + p))*EllipticE(PI/2. - xi/2.,(4*e)/(-6 + 2*e + p)))/((1 - pow(e,2))*(-4 + p))) +
   (sqrt(-4*pow(e,2) + pow(-2 + p,2))*p*EllipticF(PI/2. - xi/2.,(4*e)/(-6 + 2*e + p)))/((1 - pow(e,2))*sqrt(-6 + 2*e + p)) -
   (2*sqrt(-4*pow(e,2) + pow(-2 + p,2))*(8*(1 - pow(e,2)) + (1 + 3*pow(e,2) - p)*p)*EllipticPi((-2*e)/(1 - e),-PI/2. + xi/2.,(4*e)/(-6 + 2*e + p)))/
    ((1 - e)*(1 - pow(e,2))*(-4 + p)*sqrt(-6 + 2*e + p)) + (8*sqrt(-4*pow(e,2) + pow(-2 + p,2))*
      EllipticPi((4*e)/(-2 + 2*e + p),-PI/2. + xi/2.,(4*e)/(-6 + 2*e + p)))/(sqrt(-6 + 2*e + p)*(-2 + 2*e + p)) -
   (e*p*sqrt((-4*pow(e,2) + pow(-2 + p,2))*(-6 + p - 2*e*cos(xi)))*sin(xi))/((1 - pow(e,2))*(-4 + p)*(1 + e*cos(xi))));
}



__global__
void ellpe_kernel(fod *parr, fod *earr, fod *U_out, fod *V_out, int num){
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num;
       i += blockDim.x * gridDim.x){
         double p = parr[i];
         double e = earr[i];
         //double n=0.6;
         //double m=0.4;
         //double phi = PI/4;
         //double p = 10.0;
         //double e = 0.5;
         double xi = 0.1;
        //out[i] = EllipticE(0.5);
        //out[i] = EllipticF(PI/2, 0.5);
        //out[i] = EllipticPi(n=0.6, phi=3.1415926535897932384626433832795028841971/4., m=0.4);
        //out[i] = EllipticPi(n,phi, m);
        U_out[i] = Get_U0(p, e, xi);
        V_out[i] = Get_V0(p, e, xi);
        //out[i] = Get_T_osc(p, e);
    }
}



void ellpe_test(fod *input_mat, int num){

  fod *p, *e;

  cudaMalloc(&p, num*sizeof(fod));
  cudaMalloc(&e, num*sizeof(fod));

  cudaMemcpy(p, input_mat, num*sizeof(fod), cudaMemcpyHostToDevice);
  cudaMemcpy(e, &input_mat[num], num*sizeof(fod), cudaMemcpyHostToDevice);

  fod *U_out, *V_out;
  cudaMalloc(&U_out, num*sizeof(fod));
  cudaMalloc(&V_out, num*sizeof(fod));

  int num_blocks = ceil((num + NUM_THREADS -1)/NUM_THREADS);
  dim3 gridDim(num_blocks); //, num_teuk_modes);
  ellpe_kernel<<<gridDim, NUM_THREADS>>>(p, e, U_out, V_out, num);
  cudaDeviceSynchronize();
  cudaGetLastError();

  fod *check = new fod[num];
  fod *check2 = new fod[num];
  cudaMemcpy(check, U_out, num*sizeof(fod), cudaMemcpyDeviceToHost);
  cudaMemcpy(check2, V_out, num*sizeof(fod), cudaMemcpyDeviceToHost);

  //printf("%lf, %lf, %lf, %lf\n", input_mat[0], input_mat[num], check[0], check2[0]);

  delete[] check;
  delete[] check2;
  cudaFree(p);
  cudaFree(e);
  cudaFree(U_out);
  cudaFree(V_out);
}
