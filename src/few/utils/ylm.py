# Function for ylm generation for FastEMRIWaveforms Packages

from cmath import exp
from math import cos, pow, sin, sqrt
from math import pi as PI
from typing import Optional, Union

import numpy as np
from numba import njit

# base classes
from ..utils.baseclasses import ParallelModuleBase

I = 1j


# fmt: off
@njit(fastmath=False)
def _ylm_kernel_inner(
    l: int,
    m: int,
    theta: float,
    phi: float
    ) -> complex:
    if (l==2) and (m==-2):
        temp = (sqrt(5./PI)*pow(sin(theta/2.),4))/(2.*exp(2.*I*phi))
    elif (l==2 and m==-1):
        temp = (sqrt(5./PI)*cos(theta/2.)*pow(sin(theta/2.),3))/exp(I*phi)
    elif (l==2 and m==0):
        temp = sqrt(15./(2.*PI))*pow(cos(theta/2.),2)*pow(sin(theta/2.),2)
    elif (l==2 and m==1):
        temp = exp(I*phi)*sqrt(5./PI)*pow(cos(theta/2.),3)*sin(theta/2.)
    elif (l==2 and m==2):
        temp = (exp(2.*I*phi)*sqrt(5./PI)*pow(cos(theta/2.),4))/2.
    elif (l==3 and m==-3):
        temp = (sqrt(21./(2.*PI))*cos(theta/2.)*pow(sin(theta/2.),5))/exp(3.*I*phi)
    elif (l==3 and m==-2):
        temp = (sqrt(7./PI)*(5.*pow(cos(theta/2.),2)*pow(sin(theta/2.),4) - pow(sin(theta/2.),6)))/(2.*exp(2.*I*phi))
    elif (l==3 and m==-1):
        temp = -((sqrt(7./(10.*PI))*(-10.*pow(cos(theta/2.),3)*pow(sin(theta/2.),3) + 5.*cos(theta/2.)*pow(sin(theta/2.),5)))/exp(I*phi))
    elif (l==3 and m==0):
        temp = (sqrt(21./(10.*PI))*(10.*pow(cos(theta/2.),4)*pow(sin(theta/2.),2) - 10.*pow(cos(theta/2.),2)*pow(sin(theta/2.),4)))/2.
    elif (l==3 and m==1):
        temp = -(exp(I*phi)*sqrt(7./(10.*PI))*(-5.*pow(cos(theta/2.),5)*sin(theta/2.) + 10.*pow(cos(theta/2.),3)*pow(sin(theta/2.),3)))
    elif (l==3 and m==2):
        temp = (exp(2.*I*phi)*sqrt(7./PI)*(pow(cos(theta/2.),6) - 5.*pow(cos(theta/2.),4)*pow(sin(theta/2.),2)))/2.
    elif (l==3 and m==3):
        temp = -(exp(3.*I*phi)*sqrt(21./(2.*PI))*pow(cos(theta/2.),5)*sin(theta/2.))
    elif (l==4 and m==-4):
        temp = (3.*sqrt(7./PI)*pow(cos(theta/2.),2)*pow(sin(theta/2.),6))/exp(4.*I*phi)
    elif (l==4 and m==-3):
        temp = (-3.*sqrt(7./(2.*PI))*(-6.*pow(cos(theta/2.),3)*pow(sin(theta/2.),5) + 2.*cos(theta/2.)*pow(sin(theta/2.),7)))/(2.*exp(3.*I*phi))
    elif (l==4 and m==-2):
        temp = (3.*(15.*pow(cos(theta/2.),4)*pow(sin(theta/2.),4) - 12.*pow(cos(theta/2.),2)*pow(sin(theta/2.),6) + pow(sin(theta/2.),8)))/(2.*exp(2.*I*phi)*sqrt(PI))
    elif (l==4 and m==-1):
        temp = (-3.*(-20.*pow(cos(theta/2.),5)*pow(sin(theta/2.),3) + 30.*pow(cos(theta/2.),3)*pow(sin(theta/2.),5) - 6.*cos(theta/2.)*pow(sin(theta/2.),7)))/(2.*exp(I*phi)*sqrt(2.*PI))
    elif (l==4 and m==0):
        temp = (3.*(15.*pow(cos(theta/2.),6)*pow(sin(theta/2.),2) - 40.*pow(cos(theta/2.),4)*pow(sin(theta/2.),4) + 15.*pow(cos(theta/2.),2)*pow(sin(theta/2.),6)))/sqrt(10.*PI)
    elif (l==4 and m==1):
        temp = (-3.*exp(I*phi)*(-6.*pow(cos(theta/2.),7)*sin(theta/2.) + 30.*pow(cos(theta/2.),5)*pow(sin(theta/2.),3) - 20.*pow(cos(theta/2.),3)*pow(sin(theta/2.),5)))/(2.*sqrt(2.*PI))
    elif (l==4 and m==2):
        temp = (3.*exp(2.*I*phi)*(pow(cos(theta/2.),8) - 12.*pow(cos(theta/2.),6)*pow(sin(theta/2.),2) + 15.*pow(cos(theta/2.),4)*pow(sin(theta/2.),4)))/(2.*sqrt(PI))
    elif (l==4 and m==3):
        temp = (-3.*exp(3.*I*phi)*sqrt(7./(2.*PI))*(2.*pow(cos(theta/2.),7)*sin(theta/2.) - 6.*pow(cos(theta/2.),5)*pow(sin(theta/2.),3)))/2.
    elif (l==4 and m==4):
        temp = 3.*exp(4.*I*phi)*sqrt(7./PI)*pow(cos(theta/2.),6)*pow(sin(theta/2.),2)
    elif (l==5 and m==-5):
        temp = (sqrt(330./PI)*pow(cos(theta/2.),3)*pow(sin(theta/2.),7))/exp(5.*I*phi)
    elif (l==5 and m==-4):
        temp = (sqrt(33./PI)*(7.*pow(cos(theta/2.),4)*pow(sin(theta/2.),6) - 3.*pow(cos(theta/2.),2)*pow(sin(theta/2.),8)))/exp(4.*I*phi)
    elif (l==5 and m==-3):
        temp = -((sqrt(22./(3.*PI))*(-21.*pow(cos(theta/2.),5)*pow(sin(theta/2.),5) + 21.*pow(cos(theta/2.),3)*pow(sin(theta/2.),7) - 3.*cos(theta/2.)*pow(sin(theta/2.),9)))/exp(3.*I*phi))
    elif (l==5 and m==-2):
        temp = (sqrt(11./PI)*(35.*pow(cos(theta/2.),6)*pow(sin(theta/2.),4) - 63.*pow(cos(theta/2.),4)*pow(sin(theta/2.),6) + 21.*pow(cos(theta/2.),2)*pow(sin(theta/2.),8) - pow(sin(theta/2.),10)))/(2.*exp(2.*I*phi))
    elif (l==5 and m==-1):
        temp = -((sqrt(11./(7.*PI))*(-35.*pow(cos(theta/2.),7)*pow(sin(theta/2.),3) + 105.*pow(cos(theta/2.),5)*pow(sin(theta/2.),5) - 63.*pow(cos(theta/2.),3)*pow(sin(theta/2.),7) + 7.*cos(theta/2.)*pow(sin(theta/2.),9)))/exp(I*phi))
    elif (l==5 and m==0):
        temp = sqrt(55./(42.*PI))*(21.*pow(cos(theta/2.),8)*pow(sin(theta/2.),2) - 105.*pow(cos(theta/2.),6)*pow(sin(theta/2.),4) + 105.*pow(cos(theta/2.),4)*pow(sin(theta/2.),6) - 21.*pow(cos(theta/2.),2)*pow(sin(theta/2.),8))
    elif (l==5 and m==1):
        temp = -(exp(I*phi)*sqrt(11./(7.*PI))*(-7.*pow(cos(theta/2.),9)*sin(theta/2.) + 63.*pow(cos(theta/2.),7)*pow(sin(theta/2.),3) - 105.*pow(cos(theta/2.),5)*pow(sin(theta/2.),5) + 35.*pow(cos(theta/2.),3)*pow(sin(theta/2.),7)))
    elif (l==5 and m==2):
        temp = (exp(2.*I*phi)*sqrt(11./PI)*(pow(cos(theta/2.),10) - 21.*pow(cos(theta/2.),8)*pow(sin(theta/2.),2) + 63.*pow(cos(theta/2.),6)*pow(sin(theta/2.),4) - 35.*pow(cos(theta/2.),4)*pow(sin(theta/2.),6)))/2.
    elif (l==5 and m==3):
        temp = -(exp(3.*I*phi)*sqrt(22./(3.*PI))*(3.*pow(cos(theta/2.),9)*sin(theta/2.) - 21.*pow(cos(theta/2.),7)*pow(sin(theta/2.),3) + 21.*pow(cos(theta/2.),5)*pow(sin(theta/2.),5)))
    elif (l==5 and m==4):
        temp = exp(4.*I*phi)*sqrt(33./PI)*(3.*pow(cos(theta/2.),8)*pow(sin(theta/2.),2) - 7.*pow(cos(theta/2.),6)*pow(sin(theta/2.),4))
    elif (l==5 and m==5):
        temp = -(exp(5.*I*phi)*sqrt(330./PI)*pow(cos(theta/2.),7)*pow(sin(theta/2.),3))
    elif (l==6 and m==-6):
        temp = (3.*sqrt(715./PI)*pow(cos(theta/2.),4)*pow(sin(theta/2.),8))/(2.*exp(6.*I*phi))
    elif (l==6 and m==-5):
        temp = -(sqrt(2145./PI)*(-8.*pow(cos(theta/2.),5)*pow(sin(theta/2.),7) + 4.*pow(cos(theta/2.),3)*pow(sin(theta/2.),9)))/(4.*exp(5.*I*phi))
    elif (l==6 and m==-4):
        temp = (sqrt(195./(2.*PI))*(28.*pow(cos(theta/2.),6)*pow(sin(theta/2.),6) - 32.*pow(cos(theta/2.),4)*pow(sin(theta/2.),8) + 6.*pow(cos(theta/2.),2)*pow(sin(theta/2.),10)))/(2.*exp(4.*I*phi))
    elif (l==6 and m==-3):
        temp = (-3.*sqrt(13./PI)*(-56.*pow(cos(theta/2.),7)*pow(sin(theta/2.),5) + 112.*pow(cos(theta/2.),5)*pow(sin(theta/2.),7) - 48.*pow(cos(theta/2.),3)*pow(sin(theta/2.),9) + 4.*cos(theta/2.)*pow(sin(theta/2.),11)))/(4.*exp(3.*I*phi))
    elif (l==6 and m==-2):
        temp = (sqrt(13./PI)*(70.*pow(cos(theta/2.),8)*pow(sin(theta/2.),4) - 224.*pow(cos(theta/2.),6)*pow(sin(theta/2.),6) + 168.*pow(cos(theta/2.),4)*pow(sin(theta/2.),8) - 32.*pow(cos(theta/2.),2)*pow(sin(theta/2.),10) + pow(sin(theta/2.),12)))/(2.*exp(2.*I*phi))
    elif (l==6 and m==-1):
        temp = -(sqrt(65./(2.*PI))*(-56.*pow(cos(theta/2.),9)*pow(sin(theta/2.),3) + 280.*pow(cos(theta/2.),7)*pow(sin(theta/2.),5) - 336.*pow(cos(theta/2.),5)*pow(sin(theta/2.),7) + 112.*pow(cos(theta/2.),3)*pow(sin(theta/2.),9) - 8.*cos(theta/2.)*pow(sin(theta/2.),11)))/(4.*exp(I*phi))
    elif (l==6 and m==0):
        temp = (sqrt(195./(7.*PI))*(28.*pow(cos(theta/2.),10)*pow(sin(theta/2.),2) - 224.*pow(cos(theta/2.),8)*pow(sin(theta/2.),4) + 420.*pow(cos(theta/2.),6)*pow(sin(theta/2.),6) - 224.*pow(cos(theta/2.),4)*pow(sin(theta/2.),8) + 28.*pow(cos(theta/2.),2)*pow(sin(theta/2.),10)))/4.
    elif (l==6 and m==1):
        temp = -(exp(I*phi)*sqrt(65./(2.*PI))*(-8.*pow(cos(theta/2.),11)*sin(theta/2.) + 112.*pow(cos(theta/2.),9)*pow(sin(theta/2.),3) - 336.*pow(cos(theta/2.),7)*pow(sin(theta/2.),5) + 280.*pow(cos(theta/2.),5)*pow(sin(theta/2.),7) - 56.*pow(cos(theta/2.),3)*pow(sin(theta/2.),9)))/4.
    elif (l==6 and m==2):
        temp = (exp(2.*I*phi)*sqrt(13./PI)*(pow(cos(theta/2.),12) - 32.*pow(cos(theta/2.),10)*pow(sin(theta/2.),2) + 168.*pow(cos(theta/2.),8)*pow(sin(theta/2.),4) - 224.*pow(cos(theta/2.),6)*pow(sin(theta/2.),6) + 70.*pow(cos(theta/2.),4)*pow(sin(theta/2.),8)))/2.
    elif (l==6 and m==3):
        temp = (-3.*exp(3.*I*phi)*sqrt(13./PI)*(4.*pow(cos(theta/2.),11)*sin(theta/2.) - 48.*pow(cos(theta/2.),9)*pow(sin(theta/2.),3) + 112.*pow(cos(theta/2.),7)*pow(sin(theta/2.),5) - 56.*pow(cos(theta/2.),5)*pow(sin(theta/2.),7)))/4.
    elif (l==6 and m==4):
        temp = (exp(4.*I*phi)*sqrt(195./(2.*PI))*(6.*pow(cos(theta/2.),10)*pow(sin(theta/2.),2) - 32.*pow(cos(theta/2.),8)*pow(sin(theta/2.),4) + 28.*pow(cos(theta/2.),6)*pow(sin(theta/2.),6)))/2.
    elif (l==6 and m==5):
        temp = -(exp(5.*I*phi)*sqrt(2145./PI)*(4.*pow(cos(theta/2.),9)*pow(sin(theta/2.),3) - 8.*pow(cos(theta/2.),7)*pow(sin(theta/2.),5)))/4.
    elif (l==6 and m==6):
        temp = (3.*exp(6.*I*phi)*sqrt(715./PI)*pow(cos(theta/2.),8)*pow(sin(theta/2.),4))/2.
    elif (l==7 and m==-7):
        temp = (sqrt(15015./(2.*PI))*pow(cos(theta/2.),5)*pow(sin(theta/2.),9))/exp(7.*I*phi)
    elif (l==7 and m==-6):
        temp = (sqrt(2145./PI)*(9.*pow(cos(theta/2.),6)*pow(sin(theta/2.),8) - 5.*pow(cos(theta/2.),4)*pow(sin(theta/2.),10)))/(2.*exp(6.*I*phi))
    elif (l==7 and m==-5):
        temp = -((sqrt(165./(2.*PI))*(-36.*pow(cos(theta/2.),7)*pow(sin(theta/2.),7) + 45.*pow(cos(theta/2.),5)*pow(sin(theta/2.),9) - 10.*pow(cos(theta/2.),3)*pow(sin(theta/2.),11)))/exp(5.*I*phi))
    elif (l==7 and m==-4):
        temp = (sqrt(165./(2.*PI))*(84.*pow(cos(theta/2.),8)*pow(sin(theta/2.),6) - 180.*pow(cos(theta/2.),6)*pow(sin(theta/2.),8) + 90.*pow(cos(theta/2.),4)*pow(sin(theta/2.),10) - 10.*pow(cos(theta/2.),2)*pow(sin(theta/2.),12)))/(2.*exp(4.*I*phi))
    elif (l==7 and m==-3):
        temp = -((sqrt(15./(2.*PI))*(-126.*pow(cos(theta/2.),9)*pow(sin(theta/2.),5) + 420.*pow(cos(theta/2.),7)*pow(sin(theta/2.),7) - 360.*pow(cos(theta/2.),5)*pow(sin(theta/2.),9) + 90.*pow(cos(theta/2.),3)*pow(sin(theta/2.),11) - 5.*cos(theta/2.)*pow(sin(theta/2.),13)))/exp(3.*I*phi))
    elif (l==7 and m==-2):
        temp = (sqrt(15./PI)*(126.*pow(cos(theta/2.),10)*pow(sin(theta/2.),4) - 630.*pow(cos(theta/2.),8)*pow(sin(theta/2.),6) + 840.*pow(cos(theta/2.),6)*pow(sin(theta/2.),8) - 360.*pow(cos(theta/2.),4)*pow(sin(theta/2.),10) + 45.*pow(cos(theta/2.),2)*pow(sin(theta/2.),12) - pow(sin(theta/2.),14)))/(2.*exp(2.*I*phi))
    elif (l==7 and m==-1):
        temp = -((sqrt(5./(2.*PI))*(-84.*pow(cos(theta/2.),11)*pow(sin(theta/2.),3) + 630.*pow(cos(theta/2.),9)*pow(sin(theta/2.),5) - 1260.*pow(cos(theta/2.),7)*pow(sin(theta/2.),7) + 840.*pow(cos(theta/2.),5)*pow(sin(theta/2.),9) - 180.*pow(cos(theta/2.),3)*pow(sin(theta/2.),11) + 9.*cos(theta/2.)*pow(sin(theta/2.),13)))/exp(I*phi))
    elif (l==7 and m==0):
        temp = (sqrt(35./PI)*(36.*pow(cos(theta/2.),12)*pow(sin(theta/2.),2) - 420.*pow(cos(theta/2.),10)*pow(sin(theta/2.),4) + 1260.*pow(cos(theta/2.),8)*pow(sin(theta/2.),6) - 1260.*pow(cos(theta/2.),6)*pow(sin(theta/2.),8) + 420.*pow(cos(theta/2.),4)*pow(sin(theta/2.),10) - 36.*pow(cos(theta/2.),2)*pow(sin(theta/2.),12)))/4.
    elif (l==7 and m==1):
        temp = -(exp(I*phi)*sqrt(5./(2.*PI))*(-9.*pow(cos(theta/2.),13)*sin(theta/2.) + 180.*pow(cos(theta/2.),11)*pow(sin(theta/2.),3) - 840.*pow(cos(theta/2.),9)*pow(sin(theta/2.),5) + 1260.*pow(cos(theta/2.),7)*pow(sin(theta/2.),7) - 630.*pow(cos(theta/2.),5)*pow(sin(theta/2.),9) + 84.*pow(cos(theta/2.),3)*pow(sin(theta/2.),11)))
    elif (l==7 and m==2):
        temp = (exp(2.*I*phi)*sqrt(15./PI)*(pow(cos(theta/2.),14) - 45.*pow(cos(theta/2.),12)*pow(sin(theta/2.),2) + 360.*pow(cos(theta/2.),10)*pow(sin(theta/2.),4) - 840.*pow(cos(theta/2.),8)*pow(sin(theta/2.),6) + 630.*pow(cos(theta/2.),6)*pow(sin(theta/2.),8) - 126.*pow(cos(theta/2.),4)*pow(sin(theta/2.),10)))/2.
    elif (l==7 and m==3):
        temp = -(exp(3.*I*phi)*sqrt(15./(2.*PI))*(5.*pow(cos(theta/2.),13)*sin(theta/2.) - 90.*pow(cos(theta/2.),11)*pow(sin(theta/2.),3) + 360.*pow(cos(theta/2.),9)*pow(sin(theta/2.),5) - 420.*pow(cos(theta/2.),7)*pow(sin(theta/2.),7) + 126.*pow(cos(theta/2.),5)*pow(sin(theta/2.),9)))
    elif (l==7 and m==4):
        temp = (exp(4.*I*phi)*sqrt(165./(2.*PI))*(10.*pow(cos(theta/2.),12)*pow(sin(theta/2.),2) - 90.*pow(cos(theta/2.),10)*pow(sin(theta/2.),4) + 180.*pow(cos(theta/2.),8)*pow(sin(theta/2.),6) - 84.*pow(cos(theta/2.),6)*pow(sin(theta/2.),8)))/2.
    elif (l==7 and m==5):
        temp = -(exp(5.*I*phi)*sqrt(165./(2.*PI))*(10.*pow(cos(theta/2.),11)*pow(sin(theta/2.),3) - 45.*pow(cos(theta/2.),9)*pow(sin(theta/2.),5) + 36.*pow(cos(theta/2.),7)*pow(sin(theta/2.),7)))
    elif (l==7 and m==6):
        temp = (exp(6.*I*phi)*sqrt(2145./PI)*(5.*pow(cos(theta/2.),10)*pow(sin(theta/2.),4) - 9.*pow(cos(theta/2.),8)*pow(sin(theta/2.),6)))/2.
    elif (l==7 and m==7):
        temp = -(exp(7.*I*phi)*sqrt(15015./(2.*PI))*pow(cos(theta/2.),9)*pow(sin(theta/2.),5))
    elif (l==8 and m==-8):
        temp = (sqrt(34034./PI)*pow(cos(theta/2.),6)*pow(sin(theta/2.),10))/exp(8.*I*phi)
    elif (l==8 and m==-7):
        temp = -(sqrt(17017./(2.*PI))*(-10.*pow(cos(theta/2.),7)*pow(sin(theta/2.),9) + 6.*pow(cos(theta/2.),5)*pow(sin(theta/2.),11)))/(2.*exp(7.*I*phi))
    elif (l==8 and m==-6):
        temp = (sqrt(17017./(15.*PI))*(45.*pow(cos(theta/2.),8)*pow(sin(theta/2.),8) - 60.*pow(cos(theta/2.),6)*pow(sin(theta/2.),10) + 15.*pow(cos(theta/2.),4)*pow(sin(theta/2.),12)))/(2.*exp(6.*I*phi))
    elif (l==8 and m==-5):
        temp = -(sqrt(2431./(10.*PI))*(-120.*pow(cos(theta/2.),9)*pow(sin(theta/2.),7) + 270.*pow(cos(theta/2.),7)*pow(sin(theta/2.),9) - 150.*pow(cos(theta/2.),5)*pow(sin(theta/2.),11) + 20.*pow(cos(theta/2.),3)*pow(sin(theta/2.),13)))/(2.*exp(5.*I*phi))
    elif (l==8 and m==-4):
        temp = (sqrt(187./(10.*PI))*(210.*pow(cos(theta/2.),10)*pow(sin(theta/2.),6) - 720.*pow(cos(theta/2.),8)*pow(sin(theta/2.),8) + 675.*pow(cos(theta/2.),6)*pow(sin(theta/2.),10) - 200.*pow(cos(theta/2.),4)*pow(sin(theta/2.),12) + 15.*pow(cos(theta/2.),2)*pow(sin(theta/2.),14)))/exp(4.*I*phi)
    elif (l==8 and m==-3):
        temp = -(sqrt(187./(6.*PI))*(-252.*pow(cos(theta/2.),11)*pow(sin(theta/2.),5) + 1260.*pow(cos(theta/2.),9)*pow(sin(theta/2.),7) - 1800.*pow(cos(theta/2.),7)*pow(sin(theta/2.),9) + 900.*pow(cos(theta/2.),5)*pow(sin(theta/2.),11) - 150.*pow(cos(theta/2.),3)*pow(sin(theta/2.),13) + 6.*cos(theta/2.)*pow(sin(theta/2.),15)))/(2.*exp(3.*I*phi))
    elif (l==8 and m==-2):
        temp = (sqrt(17./PI)*(210.*pow(cos(theta/2.),12)*pow(sin(theta/2.),4) - 1512.*pow(cos(theta/2.),10)*pow(sin(theta/2.),6) + 3150.*pow(cos(theta/2.),8)*pow(sin(theta/2.),8) - 2400.*pow(cos(theta/2.),6)*pow(sin(theta/2.),10) + 675.*pow(cos(theta/2.),4)*pow(sin(theta/2.),12) - 60.*pow(cos(theta/2.),2)*pow(sin(theta/2.),14) + pow(sin(theta/2.),16)))/(2.*exp(2.*I*phi))
    elif (l==8 and m==-1):
        temp = -(sqrt(119./(10.*PI))*(-120.*pow(cos(theta/2.),13)*pow(sin(theta/2.),3) + 1260.*pow(cos(theta/2.),11)*pow(sin(theta/2.),5) - 3780.*pow(cos(theta/2.),9)*pow(sin(theta/2.),7) + 4200.*pow(cos(theta/2.),7)*pow(sin(theta/2.),9) - 1800.*pow(cos(theta/2.),5)*pow(sin(theta/2.),11) + 270.*pow(cos(theta/2.),3)*pow(sin(theta/2.),13) - 10.*cos(theta/2.)*pow(sin(theta/2.),15)))/(2.*exp(I*phi))
    elif (l==8 and m==0):
        temp = (sqrt(119./(5.*PI))*(45.*pow(cos(theta/2.),14)*pow(sin(theta/2.),2) - 720.*pow(cos(theta/2.),12)*pow(sin(theta/2.),4) + 3150.*pow(cos(theta/2.),10)*pow(sin(theta/2.),6) - 5040.*pow(cos(theta/2.),8)*pow(sin(theta/2.),8) + 3150.*pow(cos(theta/2.),6)*pow(sin(theta/2.),10) - 720.*pow(cos(theta/2.),4)*pow(sin(theta/2.),12) + 45.*pow(cos(theta/2.),2)*pow(sin(theta/2.),14)))/3.
    elif (l==8 and m==1):
        temp = -(exp(I*phi)*sqrt(119./(10.*PI))*(-10.*pow(cos(theta/2.),15)*sin(theta/2.) + 270.*pow(cos(theta/2.),13)*pow(sin(theta/2.),3) - 1800.*pow(cos(theta/2.),11)*pow(sin(theta/2.),5) + 4200.*pow(cos(theta/2.),9)*pow(sin(theta/2.),7) - 3780.*pow(cos(theta/2.),7)*pow(sin(theta/2.),9) + 1260.*pow(cos(theta/2.),5)*pow(sin(theta/2.),11) - 120.*pow(cos(theta/2.),3)*pow(sin(theta/2.),13)))/2.
    elif (l==8 and m==2):
        temp = (exp(2.*I*phi)*sqrt(17./PI)*(pow(cos(theta/2.),16) - 60.*pow(cos(theta/2.),14)*pow(sin(theta/2.),2) + 675.*pow(cos(theta/2.),12)*pow(sin(theta/2.),4) - 2400.*pow(cos(theta/2.),10)*pow(sin(theta/2.),6) + 3150.*pow(cos(theta/2.),8)*pow(sin(theta/2.),8) - 1512.*pow(cos(theta/2.),6)*pow(sin(theta/2.),10) + 210.*pow(cos(theta/2.),4)*pow(sin(theta/2.),12)))/2.
    elif (l==8 and m==3):
        temp = -(exp(3.*I*phi)*sqrt(187./(6.*PI))*(6.*pow(cos(theta/2.),15)*sin(theta/2.) - 150.*pow(cos(theta/2.),13)*pow(sin(theta/2.),3) + 900.*pow(cos(theta/2.),11)*pow(sin(theta/2.),5) - 1800.*pow(cos(theta/2.),9)*pow(sin(theta/2.),7) + 1260.*pow(cos(theta/2.),7)*pow(sin(theta/2.),9) - 252.*pow(cos(theta/2.),5)*pow(sin(theta/2.),11)))/2.
    elif (l==8 and m==4):
        temp = exp(4.*I*phi)*sqrt(187./(10.*PI))*(15.*pow(cos(theta/2.),14)*pow(sin(theta/2.),2) - 200.*pow(cos(theta/2.),12)*pow(sin(theta/2.),4) + 675.*pow(cos(theta/2.),10)*pow(sin(theta/2.),6) - 720.*pow(cos(theta/2.),8)*pow(sin(theta/2.),8) + 210.*pow(cos(theta/2.),6)*pow(sin(theta/2.),10))
    elif (l==8 and m==5):
        temp = -(exp(5.*I*phi)*sqrt(2431./(10.*PI))*(20.*pow(cos(theta/2.),13)*pow(sin(theta/2.),3) - 150.*pow(cos(theta/2.),11)*pow(sin(theta/2.),5) + 270.*pow(cos(theta/2.),9)*pow(sin(theta/2.),7) - 120.*pow(cos(theta/2.),7)*pow(sin(theta/2.),9)))/2.
    elif (l==8 and m==6):
        temp = (exp(6.*I*phi)*sqrt(17017./(15.*PI))*(15.*pow(cos(theta/2.),12)*pow(sin(theta/2.),4) - 60.*pow(cos(theta/2.),10)*pow(sin(theta/2.),6) + 45.*pow(cos(theta/2.),8)*pow(sin(theta/2.),8)))/2.
    elif (l==8 and m==7):
        temp = -(exp(7.*I*phi)*sqrt(17017./(2.*PI))*(6.*pow(cos(theta/2.),11)*pow(sin(theta/2.),5) - 10.*pow(cos(theta/2.),9)*pow(sin(theta/2.),7)))/2.
    elif (l==8 and m==8):
        temp = exp(8.*I*phi)*sqrt(34034./PI)*pow(cos(theta/2.),10)*pow(sin(theta/2.),6)
    elif (l==9 and m==-9):
        temp = (6.*sqrt(4199./PI)*pow(cos(theta/2.),7)*pow(sin(theta/2.),11))/exp(9.*I*phi)
    elif (l==9 and m==-8):
        temp = (sqrt(8398./PI)*(11.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10) - 7.*pow(cos(theta/2.),6)*pow(sin(theta/2.),12)))/exp(8.*I*phi)
    elif (l==9 and m==-7):
        temp = (-2.*sqrt(247./PI)*(-55.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9) + 77.*pow(cos(theta/2.),7)*pow(sin(theta/2.),11) - 21.*pow(cos(theta/2.),5)*pow(sin(theta/2.),13)))/exp(7.*I*phi)
    elif (l==9 and m==-6):
        temp = (sqrt(741./PI)*(165.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8) - 385.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10) + 231.*pow(cos(theta/2.),6)*pow(sin(theta/2.),12) - 35.*pow(cos(theta/2.),4)*pow(sin(theta/2.),14)))/(2.*exp(6.*I*phi))
    elif (l==9 and m==-5):
        temp = -((sqrt(247./(5.*PI))*(-330.*pow(cos(theta/2.),11)*pow(sin(theta/2.),7) + 1155.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9) - 1155.*pow(cos(theta/2.),7)*pow(sin(theta/2.),11) + 385.*pow(cos(theta/2.),5)*pow(sin(theta/2.),13) - 35.*pow(cos(theta/2.),3)*pow(sin(theta/2.),15)))/exp(5.*I*phi))
    elif (l==9 and m==-4):
        temp = (sqrt(247./(14.*PI))*(462.*pow(cos(theta/2.),12)*pow(sin(theta/2.),6) - 2310.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8) + 3465.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10) - 1925.*pow(cos(theta/2.),6)*pow(sin(theta/2.),12) + 385.*pow(cos(theta/2.),4)*pow(sin(theta/2.),14) - 21.*pow(cos(theta/2.),2)*pow(sin(theta/2.),16)))/exp(4.*I*phi)
    elif (l==9 and m==-3):
        temp = -((sqrt(57./(7.*PI))*(-462.*pow(cos(theta/2.),13)*pow(sin(theta/2.),5) + 3234.*pow(cos(theta/2.),11)*pow(sin(theta/2.),7) - 6930.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9) + 5775.*pow(cos(theta/2.),7)*pow(sin(theta/2.),11) - 1925.*pow(cos(theta/2.),5)*pow(sin(theta/2.),13) + 231.*pow(cos(theta/2.),3)*pow(sin(theta/2.),15) - 7.*cos(theta/2.)*pow(sin(theta/2.),17)))/exp(3.*I*phi))
    elif (l==9 and m==-2):
        temp = (sqrt(19./PI)*(330.*pow(cos(theta/2.),14)*pow(sin(theta/2.),4) - 3234.*pow(cos(theta/2.),12)*pow(sin(theta/2.),6) + 9702.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8) - 11550.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10) + 5775.*pow(cos(theta/2.),6)*pow(sin(theta/2.),12) - 1155.*pow(cos(theta/2.),4)*pow(sin(theta/2.),14) + 77.*pow(cos(theta/2.),2)*pow(sin(theta/2.),16) - pow(sin(theta/2.),18)))/(2.*exp(2.*I*phi))
    elif (l==9 and m==-1):
        temp = -((sqrt(38./(11.*PI))*(-165.*pow(cos(theta/2.),15)*pow(sin(theta/2.),3) + 2310.*pow(cos(theta/2.),13)*pow(sin(theta/2.),5) - 9702.*pow(cos(theta/2.),11)*pow(sin(theta/2.),7) + 16170.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9) - 11550.*pow(cos(theta/2.),7)*pow(sin(theta/2.),11) + 3465.*pow(cos(theta/2.),5)*pow(sin(theta/2.),13) - 385.*pow(cos(theta/2.),3)*pow(sin(theta/2.),15) + 11.*cos(theta/2.)*pow(sin(theta/2.),17)))/exp(I*phi))
    elif (l==9 and m==0):
        temp = 3.*sqrt(19./(55.*PI))*(55.*pow(cos(theta/2.),16)*pow(sin(theta/2.),2) - 1155.*pow(cos(theta/2.),14)*pow(sin(theta/2.),4) + 6930.*pow(cos(theta/2.),12)*pow(sin(theta/2.),6) - 16170.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8) + 16170.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10) - 6930.*pow(cos(theta/2.),6)*pow(sin(theta/2.),12) + 1155.*pow(cos(theta/2.),4)*pow(sin(theta/2.),14) - 55.*pow(cos(theta/2.),2)*pow(sin(theta/2.),16))
    elif (l==9 and m==1):
        temp = -(exp(I*phi)*sqrt(38./(11.*PI))*(-11.*pow(cos(theta/2.),17)*sin(theta/2.) + 385.*pow(cos(theta/2.),15)*pow(sin(theta/2.),3) - 3465.*pow(cos(theta/2.),13)*pow(sin(theta/2.),5) + 11550.*pow(cos(theta/2.),11)*pow(sin(theta/2.),7) - 16170.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9) + 9702.*pow(cos(theta/2.),7)*pow(sin(theta/2.),11) - 2310.*pow(cos(theta/2.),5)*pow(sin(theta/2.),13) + 165.*pow(cos(theta/2.),3)*pow(sin(theta/2.),15)))
    elif (l==9 and m==2):
        temp = (exp(2.*I*phi)*sqrt(19./PI)*(pow(cos(theta/2.),18) - 77.*pow(cos(theta/2.),16)*pow(sin(theta/2.),2) + 1155.*pow(cos(theta/2.),14)*pow(sin(theta/2.),4) - 5775.*pow(cos(theta/2.),12)*pow(sin(theta/2.),6) + 11550.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8) - 9702.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10) + 3234.*pow(cos(theta/2.),6)*pow(sin(theta/2.),12) - 330.*pow(cos(theta/2.),4)*pow(sin(theta/2.),14)))/2.
    elif (l==9 and m==3):
        temp = -(exp(3.*I*phi)*sqrt(57./(7.*PI))*(7.*pow(cos(theta/2.),17)*sin(theta/2.) - 231.*pow(cos(theta/2.),15)*pow(sin(theta/2.),3) + 1925.*pow(cos(theta/2.),13)*pow(sin(theta/2.),5) - 5775.*pow(cos(theta/2.),11)*pow(sin(theta/2.),7) + 6930.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9) - 3234.*pow(cos(theta/2.),7)*pow(sin(theta/2.),11) + 462.*pow(cos(theta/2.),5)*pow(sin(theta/2.),13)))
    elif (l==9 and m==4):
        temp = exp(4.*I*phi)*sqrt(247./(14.*PI))*(21.*pow(cos(theta/2.),16)*pow(sin(theta/2.),2) - 385.*pow(cos(theta/2.),14)*pow(sin(theta/2.),4) + 1925.*pow(cos(theta/2.),12)*pow(sin(theta/2.),6) - 3465.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8) + 2310.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10) - 462.*pow(cos(theta/2.),6)*pow(sin(theta/2.),12))
    elif (l==9 and m==5):
        temp = -(exp(5.*I*phi)*sqrt(247./(5.*PI))*(35.*pow(cos(theta/2.),15)*pow(sin(theta/2.),3) - 385.*pow(cos(theta/2.),13)*pow(sin(theta/2.),5) + 1155.*pow(cos(theta/2.),11)*pow(sin(theta/2.),7) - 1155.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9) + 330.*pow(cos(theta/2.),7)*pow(sin(theta/2.),11)))
    elif (l==9 and m==6):
        temp = (exp(6.*I*phi)*sqrt(741./PI)*(35.*pow(cos(theta/2.),14)*pow(sin(theta/2.),4) - 231.*pow(cos(theta/2.),12)*pow(sin(theta/2.),6) + 385.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8) - 165.*pow(cos(theta/2.),8)*pow(sin(theta/2.),10)))/2.
    elif (l==9 and m==7):
        temp = -2.*exp(7.*I*phi)*sqrt(247./PI)*(21.*pow(cos(theta/2.),13)*pow(sin(theta/2.),5) - 77.*pow(cos(theta/2.),11)*pow(sin(theta/2.),7) + 55.*pow(cos(theta/2.),9)*pow(sin(theta/2.),9))
    elif (l==9 and m==8):
        temp = exp(8.*I*phi)*sqrt(8398./PI)*(7.*pow(cos(theta/2.),12)*pow(sin(theta/2.),6) - 11.*pow(cos(theta/2.),10)*pow(sin(theta/2.),8))
    elif (l==9 and m==9):
        temp = -6.*exp(9.*I*phi)*sqrt(4199./PI)*pow(cos(theta/2.),11)*pow(sin(theta/2.),7)
    elif (l==10 and m==-10):
        temp = (3.*sqrt(146965./(2.*PI))*pow(cos(theta/2.),8)*pow(sin(theta/2.),12))/exp(10.*I*phi)
    elif (l==10 and m==-9):
        temp = (-3.*sqrt(29393./(2.*PI))*(-12.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) + 8.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13)))/(2.*exp(9.*I*phi))
    elif (l==10 and m==-8):
        temp = (3.*sqrt(1547./PI)*(66.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) - 96.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12) + 28.*pow(cos(theta/2.),6)*pow(sin(theta/2.),14)))/(2.*exp(8.*I*phi))
    elif (l==10 and m==-7):
        temp = -(sqrt(4641./(2.*PI))*(-220.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) + 528.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) - 336.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13) + 56.*pow(cos(theta/2.),5)*pow(sin(theta/2.),15)))/(2.*exp(7.*I*phi))
    elif (l==10 and m==-6):
        temp = (sqrt(273./(2.*PI))*(495.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) - 1760.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) + 1848.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12) - 672.*pow(cos(theta/2.),6)*pow(sin(theta/2.),14) + 70.*pow(cos(theta/2.),4)*pow(sin(theta/2.),16)))/exp(6.*I*phi)
    elif (l==10 and m==-5):
        temp = -(sqrt(1365./(2.*PI))*(-792.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) + 3960.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) - 6160.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) + 3696.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13) - 840.*pow(cos(theta/2.),5)*pow(sin(theta/2.),15) + 56.*pow(cos(theta/2.),3)*pow(sin(theta/2.),17)))/(4.*exp(5.*I*phi))
    elif (l==10 and m==-4):
        temp = (sqrt(273./PI)*(924.*pow(cos(theta/2.),14)*pow(sin(theta/2.),6) - 6336.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) + 13860.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) - 12320.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12) + 4620.*pow(cos(theta/2.),6)*pow(sin(theta/2.),14) - 672.*pow(cos(theta/2.),4)*pow(sin(theta/2.),16) + 28.*pow(cos(theta/2.),2)*pow(sin(theta/2.),18)))/(4.*exp(4.*I*phi))
    elif (l==10 and m==-3):
        temp = -(sqrt(273./(2.*PI))*(-792.*pow(cos(theta/2.),15)*pow(sin(theta/2.),5) + 7392.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) - 22176.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) + 27720.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) - 15400.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13) + 3696.*pow(cos(theta/2.),5)*pow(sin(theta/2.),15) - 336.*pow(cos(theta/2.),3)*pow(sin(theta/2.),17) + 8.*cos(theta/2.)*pow(sin(theta/2.),19)))/(4.*exp(3.*I*phi))
    elif (l==10 and m==-2):
        temp = (sqrt(21./PI)*(495.*pow(cos(theta/2.),16)*pow(sin(theta/2.),4) - 6336.*pow(cos(theta/2.),14)*pow(sin(theta/2.),6) + 25872.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) - 44352.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) + 34650.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12) - 12320.*pow(cos(theta/2.),6)*pow(sin(theta/2.),14) + 1848.*pow(cos(theta/2.),4)*pow(sin(theta/2.),16) - 96.*pow(cos(theta/2.),2)*pow(sin(theta/2.),18) + pow(sin(theta/2.),20)))/(2.*exp(2.*I*phi))
    elif (l==10 and m==-1):
        temp = (-3.*sqrt(7./PI)*(-220.*pow(cos(theta/2.),17)*pow(sin(theta/2.),3) + 3960.*pow(cos(theta/2.),15)*pow(sin(theta/2.),5) - 22176.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) + 51744.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) - 55440.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) + 27720.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13) - 6160.*pow(cos(theta/2.),5)*pow(sin(theta/2.),15) + 528.*pow(cos(theta/2.),3)*pow(sin(theta/2.),17) - 12.*cos(theta/2.)*pow(sin(theta/2.),19)))/(4.*exp(I*phi))
    elif (l==10 and m==0):
        temp = (3.*sqrt(35./(22.*PI))*(66.*pow(cos(theta/2.),18)*pow(sin(theta/2.),2) - 1760.*pow(cos(theta/2.),16)*pow(sin(theta/2.),4) + 13860.*pow(cos(theta/2.),14)*pow(sin(theta/2.),6) - 44352.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) + 64680.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) - 44352.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12) + 13860.*pow(cos(theta/2.),6)*pow(sin(theta/2.),14) - 1760.*pow(cos(theta/2.),4)*pow(sin(theta/2.),16) + 66.*pow(cos(theta/2.),2)*pow(sin(theta/2.),18)))/2.
    elif (l==10 and m==1):
        temp = (-3.*exp(I*phi)*sqrt(7./PI)*(-12.*pow(cos(theta/2.),19)*sin(theta/2.) + 528.*pow(cos(theta/2.),17)*pow(sin(theta/2.),3) - 6160.*pow(cos(theta/2.),15)*pow(sin(theta/2.),5) + 27720.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) - 55440.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) + 51744.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) - 22176.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13) + 3960.*pow(cos(theta/2.),5)*pow(sin(theta/2.),15) - 220.*pow(cos(theta/2.),3)*pow(sin(theta/2.),17)))/4.
    elif (l==10 and m==2):
        temp = (exp(2.*I*phi)*sqrt(21./PI)*(pow(cos(theta/2.),20) - 96.*pow(cos(theta/2.),18)*pow(sin(theta/2.),2) + 1848.*pow(cos(theta/2.),16)*pow(sin(theta/2.),4) - 12320.*pow(cos(theta/2.),14)*pow(sin(theta/2.),6) + 34650.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) - 44352.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) + 25872.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12) - 6336.*pow(cos(theta/2.),6)*pow(sin(theta/2.),14) + 495.*pow(cos(theta/2.),4)*pow(sin(theta/2.),16)))/2.
    elif (l==10 and m==3):
        temp = -(exp(3.*I*phi)*sqrt(273./(2.*PI))*(8.*pow(cos(theta/2.),19)*sin(theta/2.) - 336.*pow(cos(theta/2.),17)*pow(sin(theta/2.),3) + 3696.*pow(cos(theta/2.),15)*pow(sin(theta/2.),5) - 15400.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) + 27720.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) - 22176.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) + 7392.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13) - 792.*pow(cos(theta/2.),5)*pow(sin(theta/2.),15)))/4.
    elif (l==10 and m==4):
        temp = (exp(4.*I*phi)*sqrt(273./PI)*(28.*pow(cos(theta/2.),18)*pow(sin(theta/2.),2) - 672.*pow(cos(theta/2.),16)*pow(sin(theta/2.),4) + 4620.*pow(cos(theta/2.),14)*pow(sin(theta/2.),6) - 12320.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) + 13860.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) - 6336.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12) + 924.*pow(cos(theta/2.),6)*pow(sin(theta/2.),14)))/4.
    elif (l==10 and m==5):
        temp = -(exp(5.*I*phi)*sqrt(1365./(2.*PI))*(56.*pow(cos(theta/2.),17)*pow(sin(theta/2.),3) - 840.*pow(cos(theta/2.),15)*pow(sin(theta/2.),5) + 3696.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) - 6160.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) + 3960.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11) - 792.*pow(cos(theta/2.),7)*pow(sin(theta/2.),13)))/4.
    elif (l==10 and m==6):
        temp = exp(6.*I*phi)*sqrt(273./(2.*PI))*(70.*pow(cos(theta/2.),16)*pow(sin(theta/2.),4) - 672.*pow(cos(theta/2.),14)*pow(sin(theta/2.),6) + 1848.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) - 1760.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10) + 495.*pow(cos(theta/2.),8)*pow(sin(theta/2.),12))
    elif (l==10 and m==7):
        temp = -(exp(7.*I*phi)*sqrt(4641./(2.*PI))*(56.*pow(cos(theta/2.),15)*pow(sin(theta/2.),5) - 336.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) + 528.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9) - 220.*pow(cos(theta/2.),9)*pow(sin(theta/2.),11)))/2.
    elif (l==10 and m==8):
        temp = (3.*exp(8.*I*phi)*sqrt(1547./PI)*(28.*pow(cos(theta/2.),14)*pow(sin(theta/2.),6) - 96.*pow(cos(theta/2.),12)*pow(sin(theta/2.),8) + 66.*pow(cos(theta/2.),10)*pow(sin(theta/2.),10)))/2.
    elif (l==10 and m==9):
        temp = (-3.*exp(9.*I*phi)*sqrt(29393./(2.*PI))*(8.*pow(cos(theta/2.),13)*pow(sin(theta/2.),7) - 12.*pow(cos(theta/2.),11)*pow(sin(theta/2.),9)))/2.
    elif (l==10 and m==10):
        temp = 3.*exp(10.*I*phi)*sqrt(146965./(2.*PI))*pow(cos(theta/2.),12)*pow(sin(theta/2.),8)

    else:
        raise ValueError("(l, m) > 10 are not supported.")

    return temp

# fmt: on


@njit(fastmath=False)
def _ylm_kernel(
    out: np.ndarray, l: np.ndarray, m: np.ndarray, theta: float, phi: float
):
    for k in range(len(out)):
        out[k] = _ylm_kernel_inner(l[k], m[k], theta, phi)

    return out


class GetYlms(ParallelModuleBase):
    r"""(-2) Spin-weighted Spherical Harmonics

    The class generates (-2) spin-weighted spherical harmonics,
    :math:`Y_{lm}(\Theta,\phi)`.

    args:
        include_minus_m: Set True if only providing :math:`m\geq0`,
            it will return twice the number of requested modes with the second
            half as modes with :math:`m<0` for array inputs of :math:`l,m`. **Warning**: It will also duplicate
            the :math:`m=0` modes. Default is False.
        **kwargs: Optional keyword arguments for the base class:
            :class:`few.utils.baseclasses.ParallelModuleBase`.
    """

    def __init__(self, include_minus_m: bool = False, **kwargs: Optional[dict]):
        ParallelModuleBase.__init__(self, **kwargs)
        self.include_minus_m = include_minus_m

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    # These are the spin-weighted spherical harmonics with s=-2
    def __call__(
        self,
        l_in: Union[int, np.ndarray],
        m_in: Union[int, np.ndarray],
        theta: float,
        phi: float,
    ) -> np.ndarray:
        """Call method for Ylms.

        This returns ylms based on requested :math:`(l,m)` values and viewing
        angles.

        args:
            l_in: :math:`l` values requested.
            m_in: :math:`m` values requested.
            theta: Polar viewing angle.
            phi: Azimuthal viewing angle.

        Returns:
            Complex array of Ylm values.
        """
        if isinstance(l_in, int) or isinstance(m_in, int):
            assert isinstance(l_in, int) and isinstance(m_in, int)
            return _ylm_kernel_inner(l_in, m_in, theta, phi)

        # if assuming positive m, repeat entries for negative m
        # this will duplicate m = 0
        if self.include_minus_m:
            l = self.xp.zeros(2 * l_in.shape[0], dtype=int)
            m = self.xp.zeros(2 * l_in.shape[0], dtype=int)

            l[: l_in.shape[0]] = l_in
            l[l_in.shape[0] :] = l_in

            m[: l_in.shape[0]] = m_in
            m[l_in.shape[0] :] = -m_in

        # if not, just l_in, m_in
        else:
            l = l_in
            m = m_in

        # the function only works with CPU allocated arrays
        # if l and m are cupy arrays, turn into numpy arrays
        try:
            l = l.get()
            m = m.get()

        except AttributeError:
            pass

        out = np.zeros(len(l), dtype=np.complex128)
        # get ylm arrays and cast back to cupy if using cupy/GPUs
        return self.xp.asarray(
            _ylm_kernel(out, l.astype(np.int32), m.astype(np.int32), theta, phi)
        )
