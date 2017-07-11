# file: check_basics.py 
# Programmer: Sushant N More more.13@osu.edu
# 
# Revision history: 
#    July 11, 2014: Original version
#
# Goals:
#   To check if:
#   1. Clebsh-Gordan work as expected
#   2. Spherical Harmonics work as expected
#   3. Input/ Output of files works as expected
#   4. Interpolation works as expected
#
#------------------------------------------------------------
#
# Global constants

import math
import numpy as np
#from scipy import *
from sympy.physics.quantum.cg import CG
from sympy import S
from scipy.special import sph_harm
import os
#import pylab
import matplotlib.pyplot as plt
#import matplotlib
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.integrate import quad


hbarc = 197.326 # in MeV-fm
Mp = 938.272 # proton mass
Mn = 939.565 # neutron mass
Mnp = (Mn + Mp)/2.0 # avearge nucleon mass
Pi = math.pi

#print 'Mnp = ', Mnp 
cg = float(CG(S(1), S(2), S(3), S(4), 5, 6).doit())

#cg1 = cg.doit()

#print 'CG = ', cg

# Thus Clebsch-Gordan coefficients are reproduced correctly

Y = sph_harm(1,2 , Pi/2, Pi/2) # The spherical harmonic Y_lm (theta, phi) is represented in python as sph_harm(m, l, phi, theta)

Y_real = Y.real
Y_imag = Y.imag

#print 'Ylm_real = ', Y_real, 'Ylm_imag = ', Y_imag

# Thus spherical Harmonics are reproduced correctly

#print (os.getcwd()) #This gives the current directory


#fileName = '/home/sushant/Dropbox/kai_sushant_electrodisintegration/Mathematica/SRG/wave_functions/wavefunction_S_lambda_inf.dat'

filepath_psi_S = '../Mathematica/SRG/wave_functions/wavefunction_S_lambda_inf.dat'
filepath_psi_D = '../Mathematica/SRG/wave_functions/wavefunction_D_lambda_inf.dat'


psi_S = np.loadtxt(filepath_psi_S, skiprows=0, unpack=True)


plt.figure(figsize=(5.,5.))

plt.plot( psi_S[0,:], psi_S[1,:], 'r')

#pylab.legend()
#pylab.title("Title of Plot")
#pylab.xlabel("X Axis Label")
#pylab.ylabel("Y Axis Label")

plt.savefig("wave_fn_S_unevolved.pdf")

#plt.show()

psi_D = np.loadtxt(filepath_psi_D, skiprows=0, unpack=True)


#plt.figure(figsize=(5.,5.))

#plt.plot( psi_D[0,:], psi_D[1,:], 'b')


#plt.savefig("wave_fn_D_unevolved.pdf")

#plt.show()

# Now let us check if interpolation works as expected

psi_S_I_c = interp1d(psi_S[0,:], psi_S[1,:], kind = 'cubic')
psi_S_I_new = interpolate.InterpolatedUnivariateSpline(psi_S[0,:], psi_S[1,:])


#tck = interpolate.splrep(psi_S[0,:], psi_S[1,:], s=0)
#ynew = interpolate.splev(1, tck, der=0)

#print "psi_S_cubic = ", psi_S_I_c(1), "psi_S_spline = ", psi_S_I_new(1), "psi_S_c_spline = ", ynew

#print 'psi_S[q[2]] = ', psi_S[1,1]

psi_D_I_c = interp1d(psi_D[0,:], psi_D[1,:], kind = 'cubic')
psi_D_I_new = interpolate.InterpolatedUnivariateSpline(psi_D[0,:], psi_D[1,:])


#tck = interpolate.splrep(psi_D[0,:], psi_D[1,:], s=0)
#ynew = interpolate.splev(1, tck, der=0)

#print "psi_D_cubic = ", psi_D_I_c(1), "psi_D_spline = ", psi_D_I_new(1), "psi_D_c_spline = ", ynew

#print 'psi_D[q[2]] = ', psi_D[1,1]

# All the different methods of interpolation produce the same result. The result
# matches to 8 or 9 decimal places to the result obtained in Mathematica when the
# method 'Spline' is specified in Mathematica

norm =   quad(lambda x: (2.0/Pi)*(x*x*psi_S_I_new(x)*psi_S_I_new(x) + x*x*psi_D_I_new(x)*psi_D_I_new(x)), 0, 30)

#result = quad(lambda x: math.sin(x), 0, Pi)

#print norm

#def psi(0,k):
#    return (1./norm)* psi_S_I(k)

#for l in range(0,6):
#  psi(l, k)  = 0

print (1./norm[0])* psi_D_I_new(1.)

#psi[0] = (1./norm[0])* psi_S_I_new
#psi(2, k) = (1./norm)* psi_D_I_new(k)

#print "psi_s = ", psi(0,1)

psi = np.array([ psi_S_I_new, 0 ,  psi_D_I_new])

print psi[2](1)

I = complex(0,1)

def sph_harmY(l, m , theta, phi):
    if abs(m) <= abs(l):
       return sph_harm(m, l, phi, theta)
    else:
       return 0.0 


#print I**2 + 2
#print I + complex(5,3)

#for i in range(0,5,2):
#   print "i = ", i
theta_cm = Pi/6
phi_cm = Pi/3



#sum1 = 0.
#for L1 in range(0,5,2): 
#    for J1 in range(abs(L1-1),L1+2):
#           for MJ1 in range(-J1, J1+1): 
#               sum1 = sum1 + sph_harmY(L1, MJ1 , theta_cm, phi_cm)
               #print "L1 =", L1, "J1 =", J1, "MJ1 =", MJ1

#print "sum =", sum1
