
# coding: utf-8

### Calculate f_L and the overlap matrix element first when everything is unevolved and later when everything except the current is evolved 

#### Author: Sushant N More     Date: Aug 15, 2014

##### Notes: The kinematics used is the one at quasi-free ridge. Yang and Phillips find that FSI contribution to f_L is least at the quasi-free ridge. This is explained by the observation that, at the quasi-free ridge, both nucleons are on the mass shell after the virtual photon strikes the deuteron, and there is no need for the FSI in order to make the final state particles real. We would like to check if the same holds if we SRG evolve the initial and final state but not the current operator (If everything is evolved the answer would be invariant).

###### Revision history:  August 15, 2014: Started writing

# In[1]:

import math
import cmath
import numpy as np
#from scipy import *
from sympy.physics.quantum.cg import CG
from sympy import S
from scipy.special import sph_harm
#import os
#import pylab
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
#import matplotlib
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
from scipy.integrate import quad
from ntg_util.mesh import Gauss_Legendre_Mesh
import timeit


# In[2]:

#Global constants
hbar_c = 197.326;  # in MeV-fm 
Mnp = (938.272 + 939.565)/2.0; # average of nucleon mass in MeV 
Md = 1875.6; # mass of deuteron in MeV
Mp = 938.272; # mass of proton in MeV

Pi = math.pi
I = complex(0,1)

qq =  math.sqrt(1); # momentum transferred in fm^-1
qqE = qq*hbar_c; # 3 momentum transferred in MeV
Enp = 10; # Energy of the final neutron-proton system
Qsq = qqE**2 - (Enp - math.sqrt(Md**2 + qqE**2) + 2*Mp)**2; # the second term in \[Omega]^2. cf. Eq. 6.85 Yang thesis
ppE = math.sqrt(Mnp*Enp); # final 3 momentum of proton in MeV
pp = ppE/hbar_c ; # final 3 momemtum of the proton in final p-n COM frame in fm^-1
Ed = Md + qqE**2/(2*Md);
alpha = 1.0/137; # the fine structure constant
Ep = Mp + ppE**2/(2*Mp);
QsqG = Qsq/(1000*1000); # in GeV^2*
mDsq = 0.71;
GD = (1 + QsqG/mDsq)**(-2);
GEp = 0.985*GD; # electric form factor for deuteron at (Q^2) ~ 0.04 (GeV^2) [PRC 75.035202]
GEn = 0.02; # electric form factor for neutron at (Q^2) ~ 0.04 (GeV^2) [PRC 75.035202]


# In[3]:

wvfnDir = '../Mathematica/SRG/wave_functions/'

psi_S = np.loadtxt(wvfnDir + 'wavefunction_S_lambda_inf.dat', skiprows=0, unpack=True)
psi_D = np.loadtxt(wvfnDir + 'wavefunction_D_lambda_inf.dat', skiprows=0, unpack=True)

psi_S_I = interpolate.InterpolatedUnivariateSpline(psi_S[0,:], psi_S[1,:]) 
#interpolate the wave function to get values for points not on the mesh
psi_D_I = interpolate.InterpolatedUnivariateSpline(psi_D[0,:], psi_D[1,:])

norm =   quad(lambda x: (2.0/Pi)*x*x*(abs(psi_S_I(x))**2 + abs(psi_D_I(x))**2), 0, 30)[0]

psi_interpolated = np.array([psi_S_I, 0 ,  psi_D_I])


# In[4]:

wvfnDir = '../Mathematica/SRG/wave_functions/'

psi_S_lambda2 = np.loadtxt(wvfnDir + 'wavefunction_S_lambda_2.0.dat', skiprows=0, unpack=True)
psi_D_lambda2 = np.loadtxt(wvfnDir + 'wavefunction_D_lambda_2.0.dat', skiprows=0, unpack=True)

psi_S_I_lambda2 = interpolate.InterpolatedUnivariateSpline(psi_S_lambda2[0,:], psi_S_lambda2[1,:]) 
#interpolate the wave function to get values for points not on the mesh
psi_D_I_lambda2 = interpolate.InterpolatedUnivariateSpline(psi_D_lambda2[0,:], psi_D_lambda2[1,:])

norm2 =   quad(lambda x: (2.0/Pi)*x*x*(abs(psi_S_I_lambda2(x))**2 + abs(psi_D_I_lambda2(x))**2), 0, 30)[0]

psi_interpolated2 = np.array([psi_S_I_lambda2, 0 ,  psi_D_I_lambda2])


# In[5]:

def theta_prime(p, theta):
    return np.arcsin((p* np.sin(theta))/(np.sqrt(p**2 - p* qq* np.cos(theta)+ (qq**2)/4))) 

def theta_prime2(p, theta):
    return np.arcsin((p* np.sin(theta))/(np.sqrt(p**2 + p* qq* np.cos(theta)+ (qq**2)/4))) 

def sph_harmY(l, m , theta, phi):
    if abs(m) <= abs(l):
       return sph_harm(m, l, phi, theta)
    else:
       return 0.0 


# In[6]:

def overlap_pmq(pp, theta, Ms, MJ, phi, lam): #overlap of p-q/2
    if lam == 100: # unevolved
       return math.sqrt(2./Pi)* sum(
                                   sph_harmY(L, MJ-Ms, theta_prime(pp, theta), phi)*\
                                   float(CG(S(L), S(MJ - Ms), S(1), S(Ms), 1, MJ).doit())*(I**L)*\
                                   psi_interpolated[L](np.sqrt(pp**2 - pp* qq* np.cos(theta)+ qq**2/4))/norm
                                   for L in range(0,3,2)
                                   )
    if lam == 2:
       return math.sqrt(2./Pi)* sum(
                                   sph_harmY(L, MJ-Ms, theta_prime(pp, theta), phi)*\
                                   float(CG(S(L), S(MJ - Ms), S(1), S(Ms), 1, MJ).doit())*(I**L)*\
                                   psi_interpolated2[L](np.sqrt(pp**2 - pp* qq* np.cos(theta)+ qq**2/4))/norm2
                                   for L in range(0,3,2)
                                   )

def overlap_mpmq(pp, theta, Ms, MJ, phi,lam): #overlap of -p-q/2
    if lam == 100:
        return math.sqrt(2./Pi)* sum(
                                    sph_harmY(L, MJ-Ms, theta_prime2(pp, theta), phi+Pi)*\
                                    float(CG(S(L), S(MJ - Ms), S(1), S(Ms), 1, MJ).doit())*(I**L)*\
                                    psi_interpolated[L](np.sqrt(pp**2 + pp* qq* np.cos(theta)+ qq**2/4))/norm
                                    for L in range(0,3,2)
                                    )
    if lam == 2:
        return math.sqrt(2./Pi)* sum(
                                    sph_harmY(L, MJ-Ms, theta_prime2(pp, theta), phi+Pi)*\
                                    float(CG(S(L), S(MJ - Ms), S(1), S(Ms), 1, MJ).doit())*(I**L)*\
                                    psi_interpolated2[L](np.sqrt(pp**2 + pp* qq* np.cos(theta)+ qq**2/4))/norm2
                                    for L in range(0,3,2)
                                    )

def overlap_IA(theta, Ms, MJ, lam):
    return GEp*overlap_pmq(pp, theta, Ms, MJ, Pi/6, lam) + GEn*overlap_mpmq(pp, theta, Ms, MJ, Pi/6, lam) #f_L is independent of phi

C1 = -Pi*np.sqrt(2*alpha*pp*(Ep/hbar_c)*(Ed/hbar_c)/(Md/hbar_c))

def fl_IA(theta,lam):
    return sum(
               C1**2*overlap_IA(theta, Ms, MJ, lam)*overlap_IA(theta, Ms, MJ, lam).conjugate()
                for Ms in range(-1,2)
                  for MJ in range(-1,2)
              )


# In[7]:

fl_IA(0,100)


# In[8]:

start = timeit.default_timer()
fl_IA(0,100)
stop = timeit.default_timer()
print stop - start


# In[17]:

Theta_plt = np.linspace(0,180,19)


# In[18]:

Theta_plt


# In[20]:

fl_IA(Theta_plt*Pi/180,100)


# In[21]:

fl_IA(Theta_plt*Pi/180,2)


# In[19]:

plt.plot(Theta_plt, fl_IA(Theta_plt*Pi/180,100),marker = '+')
plt.plot(Theta_plt, fl_IA(Theta_plt*Pi/180,2),marker = 'x')
plt.xlabel(r'$\theta$',fontsize = 24)
plt.ylabel(r'$f_L^{IA}$',fontsize = 24)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('$E_{np} = 10$ $MeV$, $q_{cm}^2 = 10$ $fm^{-2} $', fontsize= 16)
plt.show()


###### Matches Mathematica answer. 

# In[13]:

plt.plot(Theta_plt, fl_IA(Theta_plt*Pi/180,2),marker = 'o')
plt.xlabel(r'$\theta$',fontsize = 24)
plt.ylabel(r'$f_L^{IA}$',fontsize = 24)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('$E_{np} = 10$ $MeV$, $q_{cm}^2 = 10$ $fm^{-2} $', fontsize= 16)
plt.show()


# In[14]:

pp


###### There is hardly any discernible effect if we just use the evolved (initial state) deuteron wave function. This is because for the kinematics under consideration, we are probing the wave function at the momentum ~ 0.5 fm^-1 and this low momentum, the evolved and the unevolved wave functions hardly differ.

# In[15]:

k_plot = np.linspace(0,10,100)
plt.plot(k_plot,psi_interpolated[0](k_plot)/norm)
plt.plot(k_plot,psi_interpolated2[0](k_plot)/norm)
plt.xlim([0,3])
plt.show()


# In[16]:

plt.plot(k_plot,psi_interpolated[2](k_plot),linestyle = 'dashed')
plt.plot(k_plot,psi_interpolated2[2](k_plot))
plt.xlim([0,8])
plt.show()


# In[16]:



