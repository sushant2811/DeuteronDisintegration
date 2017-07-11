
# coding: utf-8

## Attempt to vectorize the calculations.  Author: Sushant N More  

#### Revision history:       August 6, 2014: Started writing. Taking off from where the earlier notebook check_vectorization left.

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
#import matplotlib.pyplot as plt
#import matplotlib
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
from scipy.integrate import quad
from ntg_util.mesh import Gauss_Legendre_Mesh
import timeit


# In[3]:

hbarc = 197.326 # in MeV-fm
Mp = 938.272 # proton mass
Mn = 939.565 # neutron mass
Mnp = (Mn + Mp)/2.0 # avearge nucleon mass
Pi = math.pi
I = complex(0,1)


# In[4]:

filepath = '../vsrg_Bonn/' # file path for the potentials
filepathU = '../Mathematica/SRG/U_matrices/' # file path for the U matrices


# In[5]:

pot = 'kvnn_07' # Choose the potential
# kvnn_07: CD-Bonn
# kvnn_02: Bonn-A

Lambda = 2.0 # Choose the SRG \lambda.

fileID = str(pot) + '_lam' + str(Lambda) + '_reg_0_3_0.out'

meshdata = np.loadtxt(filepath + 'vsrg_1S0_' + str(pot) + '_lam' + str(Lambda) + '_reg_0_3_0_mesh.out')

q = meshdata[:,0]  # momentum mesh points
w = meshdata[:,1]  # weights

wvfnDir = '../Mathematica/SRG/wave_functions/'


# In[6]:

psi_S = np.loadtxt(wvfnDir + 'wavefunction_S_lambda_inf.dat', skiprows=0, unpack=True)
psi_D = np.loadtxt(wvfnDir + 'wavefunction_D_lambda_inf.dat', skiprows=0, unpack=True)

psi_S_I = interpolate.InterpolatedUnivariateSpline(psi_S[0,:], psi_S[1,:]) 
#interpolate the wave function to get values for points not on the mesh
psi_D_I = interpolate.InterpolatedUnivariateSpline(psi_D[0,:], psi_D[1,:])


# In[7]:

norm =   quad(lambda x: (2.0/Pi)*x*x*(abs(psi_S_I(x))**2 + abs(psi_D_I(x))**2), 0, 30)[0]


# In[8]:

norm


# In[9]:

psi_S_I(([2,3]))


# In[10]:

psi_S_I(([3,2,4])) # Thus 1D interpolation does not need input data to be sorted in ascending order


# In[11]:

U3S1data = np.loadtxt(filepathU + 'Umat_3S1_' + fileID)
U1P1data = np.loadtxt(filepathU + 'Umat_1P1_' + fileID)
U3P1data = np.loadtxt(filepathU + 'Umat_3P1_' + fileID)


# In[11]:

U3S1data.shape


##### A.shape returns a tuple (m, n) where m is the number of rows and n is the number of columns.

# In[12]:

I1P1bare = interpolate.interp2d(q, q, U1P1data[:, 2], kind = 'cubic')

I3P1bare = interpolate.interp2d(q, q, U3P1data[:, 2], kind = 'cubic')

I3S11bare = interpolate.interp2d(q, q, U3S1data[:, 2], kind = 'cubic')
I3S12bare = interpolate.interp2d(q, q, U3S1data[:, 3], kind = 'cubic')
I3S13bare = interpolate.interp2d(q, q, U3S1data[:, 4], kind = 'cubic')
I3S14bare = interpolate.interp2d(q, q, U3S1data[:, 5], kind = 'cubic')


# In[13]:

U1P1reshaped = np.reshape(U1P1data[:, 2],(100,100))
U3P1reshaped = np.reshape(U3P1data[:, 2],(100,100))

U3S11reshaped = np.reshape(U3S1data[:, 2],(100,100))
U3S12reshaped = np.reshape(U3S1data[:, 3],(100,100))
U3S13reshaped = np.reshape(U3S1data[:, 4],(100,100))
U3S14reshaped = np.reshape(U3S1data[:, 5],(100,100))


# In[14]:

I1P1bivariate = RectBivariateSpline(q,q,U1P1reshaped)
I3P1bivariate = RectBivariateSpline(q,q,U3P1reshaped)

I3S11bivariate = RectBivariateSpline(q,q,U3S11reshaped)
I3S12bivariate = RectBivariateSpline(q,q,U3S12reshaped)
I3S13bivariate = RectBivariateSpline(q,q,U3S13reshaped)
I3S14bivariate = RectBivariateSpline(q,q,U3S14reshaped)


# In[15]:

start = timeit.default_timer()
U1P1reshaped = np.reshape(U1P1data[:, 2],(100,100))
for i in range(0,100000):
   I1P1bivariate(2.3,([1,3]))
stop = timeit.default_timer()  
print stop - start


# In[16]:

I1P1bivariate(2.3,([1.5,3.4]))


# In[17]:

I1P1bare(2.3,([1.5,3.4]))


###### RectBivariate Spline and interp2d give the same answer

# In[18]:

start = timeit.default_timer()
for i in range(0,100000):
   I1P1bare(2.3,([1,3]))
stop = timeit.default_timer()  
print stop - start


# In[19]:

1.55/0.466


###### Thus, RectBivariate spline is about 3 times faster than interp2d

# In[15]:

def U_tilde(k, kk, L, LL, J, S, T): 
   
    if L == LL and L == J:
       #if J == 0 and S == 0 and T == 1:
       #   return I1S0bare(k, kk)[0]
       if J == 1 and S == 0 and T == 0:
          return I1P1bivariate(k, kk)[:]
          #return I1P1bivariate(k, kk)[:]
       #if J == 0 and S == 1 and T == 1:
       #   return I3P0bare(k, kk)[0]
       if J == 1 and S == 1 and T == 1:
          return I3P1bivariate(k, kk)[:] 
          #return I3P1bivariate(k, kk)[:]  
    elif abs(L - J) == 1:
       if L == 0 and LL == 0 and S == 1 and T == 0 and J == 1:
          return I3S11bivariate(k, kk)
       if L == 0 and LL == 2 and S == 1 and T == 0 and J == 1:
          return I3S12bivariate(k, kk)[:]
       if L == 2 and LL == 0 and S == 1 and T == 0 and J == 1:
          return I3S13bivariate(k, kk)[:]
       if L == 2 and LL == 2 and S == 1 and T == 0 and J == 1:
          return I3S14bivariate(k, kk)[:]
    else:
       return 0.


# In[16]:

# Kinematic parameters (Chosen here to match inputs in the earlier mathematica notebooks)
qq = math.sqrt(0.25)
pp = q[22]
theta_cm = Pi/6
phi_cm = Pi/3


# In[22]:

I3S11bare(2.3,3.4)


# In[29]:

I3S12bare(ks_mesh,1)


# In[30]:

I3S12bivariate(ks_mesh,1)[:,0]


# In[31]:

I3S11bivariate(2.3,([3,4]))


# In[32]:

I3S11bivariate(2.3,([3,4]))[0]


# In[33]:

I3S11bivariate(2.3,([3,4]))[:,0]


# In[34]:

I3S11bare(2.3,([3,4]))[:,0]


###### Python conventions for theta and phi are switched. Also, python does not return zero for unphysical values of l and m.

# In[17]:

def sph_harmY(l, m , theta, phi):
    if abs(m) <= abs(l):
       return sph_harm(m, l, phi, theta)
    else:
       return 0.0 


# In[18]:

def theta_prime(p, theta):
    return np.arcsin((p* np.sin(theta))/(np.sqrt(p**2 - p* qq* np.cos(theta)+ (qq**2)/4))) 

def theta_prime2(p, theta):
    return np.arcsin((p* np.sin(theta))/(np.sqrt(p**2 + p* qq* np.cos(theta)+ (qq**2)/4))) 


# In[19]:

theta_mesh, theta_wt = Gauss_Legendre_Mesh([0, Pi],[10])
phi_mesh, phi_wt = Gauss_Legendre_Mesh([0, 2*Pi],[10])
#print theta_wt[0]

ks_mesh, ks_wt = Gauss_Legendre_Mesh([0, 2, 30],[10, 10])


# In[35]:

theta_prime(pp, theta_mesh)


###### When numpy definitions of trignometric functions are used, the input can also be an array. 

# In[21]:

psi_new_mesh = np.array([psi_S_I, 0 ,  psi_D_I])


# In[37]:

psi_new_mesh[0](ks_mesh)/norm


# In[38]:

np.dot(psi_new_mesh[0](ks_mesh)/norm, psi_new_mesh[2](ks_mesh)/norm)


####### Dot product matches Mathematica answer

# In[39]:

b = np.zeros((5,5,2,3))


####### It's possible to define mutlidimensional arrays in python

####### 'sum' works as expected

# In[40]:

Lk = 0
sum(
    ks_wt[k3]* (ks_mesh[k3]**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh[k3])*(1./norm)*\
    ( sph_harmY(0, 0, theta_prime(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh[k3], math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0) + sph_harmY(0, 0, theta_prime2(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh[k3], math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0))  
    for k3 in range(0, len(ks_mesh))
    )


###### Operations with arrays work as expected

##### Summing using a for loop

# In[22]:

start = timeit.default_timer()
for i in range(0, 10000):
 Lk = 0
 kf = sum(
        ks_wt[k3]* (ks_mesh[k3]**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh[k3])*\
        ( sph_harmY(0, 0, theta_prime(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh[k3], math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0) + sph_harmY(0, 0, theta_prime2(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh[k3], math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0))  
        for k3 in range(0, len(ks_mesh))
        )
stop = timeit.default_timer()
print stop - start


# In[42]:

kf


# In[43]:

Lk = 0
ki = sum(
      ks_wt[k3]* (ks_mesh[k3]**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh[k3])*\
       ( sph_harmY(0, 0, theta_prime(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh[k3], math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0) + sph_harmY(0, 0, theta_prime2(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh[k3], math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0))  
       for k3 in range(0, len(ks_mesh))
       )


# In[44]:

ki


##### Summing the elements of the array

# In[24]:

start = timeit.default_timer()
for i in range(0, 10000):
 Lk = 0
 kf1 = sum(
    ks_wt* (ks_mesh**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh)*\
    ( sph_harmY(0, 0, theta_prime(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh, math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0) + sph_harmY(0, 0, theta_prime2(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh, math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0))  
    )
stop = timeit.default_timer()
print stop-start


# In[27]:

sum(kf1)


##### Summing using the dot product

# In[31]:

start = timeit.default_timer()
for i in range(0, 10000):
 Lk = 0
 kfd = np.dot(ks_wt, (ks_mesh**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh)*    ( sph_harmY(0, 0, theta_prime(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh, math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0)*1. + sph_harmY(0, 0, theta_prime2(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh, math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0)*1.)  
 )
stop = timeit.default_timer()
print stop-start    


# In[32]:

kfd


##### All the three ways of summing give the same answer. Successful vectorization of a part of a program

#### Doing it through the dot product way is the fastest. Notice that vectorization makes the program 20 times faster! Push harder to convert things into an array form

# In[47]:

ks_mesh.shape


##### Let's now extend the vectorization further

# In[48]:

sum( np.dot(ks_wt, (ks_mesh**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh)*    ( sph_harmY(0, 0, theta_prime(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh, math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0) + sph_harmY(0, 0, theta_prime2(pp, theta_mesh[0]), phi_mesh[0])* U_tilde(ks_mesh, math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[0])+ qq**2/4), Lk, 0, 1, 1, 0))  
)  
    for Lk in range(0,3,2)
    )


####### The above answer matches Mathematica answer

####### Because of the way psi and U is defined it might be tricky to vectorize Lk. Let's vectorize theta and phi and return to this later.

##### For some reason, the 2D interpolated function complains when asked to evaluate entries in an array which are not in ascending order. This is a concern for the theta integral. However, not a problem for phi integral. Let us do the phi integral using the outer product method first and return to the theta integral later.

# In[52]:

sum(
    phi_mesh[tp]*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])*\
U_tilde(ks_mesh[k], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)
   for tp in range(0, len(phi_mesh))
    for k in range(0, len(ks_mesh))
   )


####### This matches Mathematica answer. See if the result using the outer product depends on interpolation

# In[50]:

def U_tilde(k, kk, L, LL, J, S, T): 
    
    if abs(L - J) == 1:
       if L == 0 and LL == 0 and S == 1 and T == 0 and J == 1:
          return I3S11bivariate(k, kk)       # 1.Gives the correct answer. U_tilde shape- (20,1) 
          #return I3S11bivariate(k, kk)[:,0] # 2. Gives the correct answer. U_tilde shape- (20,)
          #return I3S11bivariate(k, kk)[0,:] # 3. Considers only the first element - incorrect answer
          #return I3S11bivariate(k, kk)[0]   # 4. Considers only the first element - incorrect answer
          #return I3S11bare(k, kk)[0]        # 5. Considers only the first element - incorrect answer     
          #return I3S11bare(k, kk)           # 6. Gives the correct answer. U_tilde shape- (20,)
          #return I3S11bare(k, kk)[:,0]      # 7. Gives error message 
          #return I3S11bare(k, kk)[0,:]      # 8. Gives error message
    else:
       return 0.


# In[53]:

tt = 0
sum(sum(
    phi_mesh[tp]*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])*\
U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]
   for tp in range(0, len(phi_mesh))
   ))


# In[54]:

U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0).shape


# In[55]:

U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0].shape


# In[56]:

sum(sum(np.outer(phi_mesh*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh),U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])))


# In[57]:

U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0).shape


###### Let us add ks_mesh and see how the results change

# In[58]:

sum(
    phi_mesh[tp]*ks_mesh[k]*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])*\
U_tilde(ks_mesh[0], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)
   for tp in range(0, len(phi_mesh))
    for k in range(0, len(ks_mesh))
   )


# In[59]:

sum(
    phi_mesh[tp]*ks_mesh[k]*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])*\
U_tilde(ks_mesh[k], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]
   for tp in range(0, len(phi_mesh))
    for k in range(0, len(ks_mesh))
   )


# In[60]:

def U_tilde(k, kk, L, LL, J, S, T): 
    
    if abs(L - J) == 1:
       if L == 0 and LL == 0 and S == 1 and T == 0 and J == 1:
          return I3S11bivariate(k, kk)       # 1.Same incorrect answer for both when extra sum is used in the first case. U_tilde shape- (20,1) 
          #return I3S11bivariate(k, kk)[:,0] # 2. Gives the correct answer for both. U_tilde shape- (20,)
          #return I3S11bivariate(k, kk)[0,:] # 3. Considers only the first element in U_tilde for both - incorrect answer
          #return I3S11bivariate(k, kk)[0]   # 4. Considers only the first element in U_tilde for both - incorrect answer
          #return I3S11bare(k, kk)[0]        # 5. Considers only the first element in U_tilde for both - incorrect answer     
          #return I3S11bare(k, kk)           # 6. Gives the correct answer for both.. U_tilde shape- (20,)
          #return I3S11bare(k, kk)[:,0]      # 7. Gives error message 
          #return I3S11bare(k, kk)[0,:]      # 8. Gives error message
    else:
       return 0.


# In[61]:

sum(sum(
    phi_mesh[tp]*ks_mesh*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])*\
U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]
   for tp in range(0, len(phi_mesh))
   ))


# In[62]:

sum(sum(np.outer(phi_mesh*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh),ks_mesh*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])))


###### Thus, for interpolation either use bivariate()[:,0] or bare() or use bivariate() and stick [:,0] in front of U_tilde

##### Thus, using outer product does give the expected answer. Now let us do the phi integral with all the bells and whistles

# In[63]:

Lk = 0
tt = 4

sum(
    theta_wt[tt]* phi_wt[tp]* np.sin(theta_mesh[tt])*\
     sph_harmY(1, 1, theta_mesh[tt], phi_mesh[tp]).conjugate()*\
    ks_wt[k]* (ks_mesh[k]**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh[k])*\
    ( sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh[k], math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0) \
     + sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh[k], math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0))  
    for tp in range(0, len(phi_mesh)) 
      for k in range(0, len(ks_mesh))
    )


###### Above answer matches Mathematica answer

# In[64]:

Lk = 0
tt = 4

sum(
    theta_wt[tt]* phi_wt[tp]* math.sin(theta_mesh[tt])*\
     sph_harmY(1, 1, theta_mesh[tt], phi_mesh[tp]).conjugate()*\
         sum(
    ks_wt* (ks_mesh**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh)*\
    ( sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh, math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0] \
     + sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh, math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])  
            )
    for tp in range(0, len(phi_mesh)) 
    )


# In[65]:

sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])+ np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])
))


# In[66]:

Lk = 0
tt = 4

sum(
    theta_wt[tt]* phi_wt[tp]* math.sin(theta_mesh[tt])*\
     sph_harmY(1, 1, theta_mesh[tt], phi_mesh[tp]).conjugate()*\
         sum(
    ks_wt* (ks_mesh**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh)*\
    ( sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh, math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0) \
     + sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh, math.sqrt(pp**2 + pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0))  
            )
    for tp in range(0, len(phi_mesh)) 
    )


# In[67]:

sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]) +np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0] )
))


###### All roads lead to Rome! Phi vectorization successful.

# In[68]:

start = timeit.default_timer()
for l in range(0,1000):
 sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)) 
))
stop = timeit.default_timer()
print stop-start


# In[69]:

Lk = 0
tt = 4
start = timeit.default_timer()
for l in range(0,1000):
 sum(
    theta_wt[tt]* phi_wt[tp]* math.sin(theta_mesh[tt])*\
     sph_harmY(1, 1, theta_mesh[tt], phi_mesh[tp]).conjugate()*\
         sum(
    ks_wt[k]* (ks_mesh[k]**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh[k])*\
    ( sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh[k], math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0) \
     )  
     for k in range(0, len(ks_mesh))  
            )
       for tp in range(0, len(phi_mesh)) 
    )
stop = timeit.default_timer()
print stop - start


# In[70]:

Lk = 0
tt = 4
start = timeit.default_timer()
for l in range(0,1000):
 sum(
    theta_wt[tt]* phi_wt[tp]* math.sin(theta_mesh[tt])*\
     sph_harmY(1, 1, theta_mesh[tt], phi_mesh[tp]).conjugate()*\
         sum(
    ks_wt* (ks_mesh**2)*(I**Lk)*psi_new_mesh[Lk](ks_mesh)*\
    ( sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh[tp])* U_tilde(ks_mesh, math.sqrt(pp**2 - pp* qq* math.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0) \
     )  
     #for k in range(0, len(ks_mesh))  
            )
    for tp in range(0, len(phi_mesh)) 
    )
stop = timeit.default_timer()
print stop - start


# In[71]:

1.3375/0.48137


###### Vectorizing phi integral makes it about three times faster.

###### Now let us take up vectorizing theta integral

# In[72]:

a = np.random.randint(0,10,3)
aa = np.argsort(a)
aaa = np.argsort(aa)


# In[73]:

a


# In[74]:

aa


# In[75]:

aaa


# In[76]:

a[aa]


# In[77]:

a[np.argsort(a)][np.argsort(np.argsort(a))]


###### It works! Thanks to the person on stackoverflow. Unfortunately cannot upvote his answer till I 15 points on stackoverflow. Also, thanks to Chun for making me consider this work around

# In[78]:

U_tilde(ks_mesh[0], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0)


# In[79]:

def U_tilde(k, kk, L, LL, J, S, T): 
    
    if abs(L - J) == 1:
       if L == 0 and LL == 0 and S == 1 and T == 0 and J == 1:
          return I3S11bivariate(k, kk)       # 1.Same incorrect answer for both when extra sum is used in the first case. U_tilde shape- (20,1) 
          #return (I3S11bivariate(k, kk)[:,0]) # 2. Gives the correct answer for both. U_tilde shape- (20,)
          #return I3S11bivariate(k, kk)[0,:] # 3. Considers only the first element in U_tilde for both - incorrect answer
          #return I3S11bivariate(k, kk)[0]   # 4. Considers only the first element in U_tilde for both - incorrect answer
          #return I3S11bare(k, kk)[0]        # 5. Considers only the first element in U_tilde for both - incorrect answer     
          #return I3S11bare(k, kk)           # 6. Gives the correct answer for both.. U_tilde shape- (20,)
          #return I3S11bare(k, kk)[:,0]      # 7. Gives error message 
          #return I3S11bare(k, kk)[0,:]      # 8. Gives error message
    else:
       return 0.


# In[80]:

U_tilde(([2,3]),([5,6]) , Lk, 0, 1, 1, 0)


# In[81]:

U_tilde(([2,3]),([5,6]) , Lk, 0, 1, 1, 0).shape


# In[82]:

U_tilde(ks_mesh[0], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0)


# In[83]:

U_tilde(([ks_mesh[0], ks_mesh[1]]), np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0)


# In[84]:

np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4)


# In[85]:

np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)


# In[86]:

np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)[np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4))]


# In[87]:

U_tilde(([ks_mesh[0], ks_mesh[1]]), np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)[np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4))], Lk, 0, 1, 1, 0)[:,np.argsort(np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)))]


##### Great! So, np.argsort provides a work around regarding the fact that 2D interpolation takes data only in ascending order. Since, the call to np.argsort has been made multiple times, let us check if this is faster than having a for loop.

# In[88]:

for i in range(0, len(theta_mesh)):
  print U_tilde(ks_mesh[0], np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[i])+ qq**2/4), Lk, 0, 1, 1, 0)


# In[89]:

U_tilde(ks_mesh[0], np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)[np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4))], Lk, 0, 1, 1, 0)[:,np.argsort(np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)))]


###### check that the answer matches

# In[90]:

start = timeit.default_timer()
for t in range(0,100000):
 for i in range(0, len(theta_mesh)):
   U_tilde(ks_mesh[0], np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[i])+ qq**2/4), Lk, 0, 1, 1, 0)
stop = timeit.default_timer()
print stop - start


# In[91]:

start = timeit.default_timer()
for t in range(0,100000):
   U_tilde(ks_mesh[0], np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)[np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4))], Lk, 0, 1, 1, 0)[:,np.argsort(np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)))]
stop = timeit.default_timer()
print stop - start


# In[92]:

8.91807985306/3.43754005432


# In[93]:

U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)[np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4))], Lk, 0, 1, 1, 0)[:,np.argsort(np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)))].shape


# In[94]:

a = np.ones((3,3))


# In[95]:

b = np.ones((3,3))


# In[96]:

np.outer(a,b)


# In[97]:

c = np.ones((3,3))


# In[98]:

np.outer(a,b,c)


# In[99]:

np.tensordot(a,b)


# In[101]:

np.outer(ks_mesh,np.outer(theta_mesh, phi_mesh)).shape


# In[102]:

np.outer(theta_mesh, phi_mesh).shape


###### outer product is for vectors (arrays fo form (n,)). Will have to use tensor product to do theta, phi and momentum integral simulataneously.

# In[103]:

sum_try = 0
for i in range(0, len(theta_mesh)):
    for j in range(0, len(phi_mesh)):
        for k in range(0, len(ks_mesh)):
            sum_try = sum_try + sph_harmY(1, 1, theta_mesh[i], phi_mesh[j])*U_tilde(ks_mesh[k], pp + theta_mesh[i], Lk, 0, 1, 1, 0)[:,0]
print sum_try


# In[104]:

sph_harmY(1, 1,theta_mesh, phi_mesh)


# In[105]:

U_tilde(ks_mesh, pp + theta_mesh, Lk, 0, 1, 1, 0).shape


# In[106]:

sph_harmY(1, 1,([2,3,6]), 7)


# In[107]:

for i in range(0, len(theta_mesh)):
    print sph_harmY(1, 1, theta_mesh[i], phi_mesh[i])


# In[108]:

sph_harm(1, 1,phi_mesh, theta_mesh)


##### sph_harm does not produce a matrix over theta and phi as we would like, but instead takes corresponding element from theta array and the phi array. To get around this, precalculate and store the spherical harmonics. This will also speed up the program compared to calculating spherical harmonics each time on fly. As has been previously established, calculation of spherical harmonics is time intensive.

# In[109]:

sph_harm_theta_phi = np.zeros((6, 11, 10 ,10), dtype = 'complex128')


# In[110]:

sph_harm_1 = np.zeros(10)


# In[111]:

sph_harm_1[:] = sph_harm(1, 1, phi_mesh,theta_mesh[1])


# In[112]:

for L in range(0,6):
    for m in range(-L, L+1):
        for t in range(0, len(theta_mesh)):
                sph_harm_theta_phi[L, m+L, t, :] = sph_harm(m, L, phi_mesh, theta_mesh[t])


# In[113]:

sph_harm_theta_phi[4,3+4, 2,5].conjugate()


# In[114]:

sph_harmY(4,3, theta_mesh[2], phi_mesh[5])


# In[115]:

start = timeit.default_timer()
for n in range(0,10000):
    sph_harmY(4,3, theta_mesh[2], phi_mesh[5])
stop = timeit.default_timer()
print stop - start


# In[116]:

start = timeit.default_timer()
sph_harm_theta_phi = np.zeros((6, 11, len(theta_mesh) , len(phi_mesh)), dtype = 'complex128')
for L in range(0,6):
    for m in range(-L, L+1):
        for t in range(0, len(theta_mesh)):
                sph_harm_theta_phi[L, m+L, t, :] = sph_harm(m, L, phi_mesh, theta_mesh[t])
 
for n in range(0,10000):
    sph_harm_theta_phi[4,3+4, 2,5]
    
stop = timeit.default_timer()
print stop - start


##### Thus calling the array gives the expected answer. Let us define similar arrays for other Spherical Harmonics that occur.

# In[117]:

sph_harm_cm = np.zeros((6, 11, 1 ,1), dtype = 'complex128')
sph_harm_Pi_cm = np.zeros((6, 11, 1 ,1), dtype = 'complex128')

for L in range(0, 6):
    for m in range(-L, L+1):
        sph_harm_cm[L, m+L, 0, 0] = sph_harm(m, L, phi_cm, theta_cm)
        sph_harm_Pi_cm[L, m+L, 0, 0] = sph_harm(m, L, phi_cm + Pi, Pi - theta_cm)


# In[118]:

sph_harm_theta_prime = np.zeros((3, 5, len(theta_mesh), len(phi_mesh)), dtype = 'complex128')
sph_harm_theta_prime2 = np.zeros((3, 5, len(theta_mesh), len(phi_mesh)), dtype = 'complex128')

for L in range(0,3):
    for m in range(-L, L+1):
        for t in range(0, len(theta_mesh)):
                sph_harm_theta_prime[L, m+L, t, :] = sph_harm(m, L, phi_mesh, theta_prime(pp, theta_mesh[tt]))
                sph_harm_theta_prime2[L, m+L, t, :] = sph_harm(m, L, phi_mesh, theta_prime2(pp, theta_mesh[tt]))


####### Let us use the Spherical Harmonics from array and try to reproduce the phi integration result

# In[119]:

sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]) +np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0] )
))


# In[120]:

int(np.linspace(0,9,10))


# In[121]:

NB = np.arange(10)


# In[122]:

sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()


# In[123]:

tt


# In[124]:

sph_harmY(1,1, theta_mesh[tt], phi_mesh).conjugate()


# In[125]:

sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*     (I**Lk)*sph_harm_theta_prime[1,1+1, tt, np.arange(10)],     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]) +np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*     (I**Lk)*sph_harm_theta_prime2[1,1+1, tt, np.arange(10)],     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0] )
))


# In[126]:

start = timeit.default_timer()
for i in range(0,10000):
  sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]) +   np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harmY(1, 1, theta_mesh[tt], phi_mesh).conjugate()*     (I**Lk)*sph_harmY(1,1, theta_prime2(pp, theta_mesh[tt]), phi_mesh),     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0] )
    ))    
stop = timeit.default_timer()
print stop - start


# In[127]:

start = timeit.default_timer()
for i in range(0,10000):
  sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*     (I**Lk)*sph_harm_theta_prime[1,1+1, tt, np.arange(10)],     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0]) +  np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*     (I**Lk)*sph_harm_theta_prime2[1,1+1, tt, np.arange(10)],     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0] )
  ))
stop = timeit.default_timer()
print stop - start


####### Thus, the array look up of Spherical Harmonics reproduces the same answer and is much faster. Now, we can go ahead with the real deal - vectorizing theta integral. For the sake of avoiding clutter, let us just look at one term.

# In[128]:

sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*     (I**Lk)*sph_harm_theta_prime[1,1+1, tt, np.arange(10)],     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])
        ))


# In[129]:

sph_harm_theta_prime[1,1+1, np.arange(10), 0]


# In[130]:

sph_harm_theta_prime[1,1+1, np.arange(10)[:], np.arange(10)]


# In[131]:

np.arange((10,10))


# In[132]:

A= np.array([[1,1],[2,2]])


# In[133]:

A.shape


# In[134]:

print A


# In[135]:

B = np.array([[3,4],[3,4]])


# In[136]:

B.shape


# In[137]:

print B


# In[138]:

sph_harm_theta_prime[1,1+1, A, B]


# In[139]:

sph_harmY(1, 1, A, B)


# In[140]:

sph_harmY(1, 1, 1, 3)


# In[141]:

sph_harmY(1, 1, 1, 4)


# In[142]:

sph_harmY(1, 1, 2, 3)


# In[143]:

sph_harmY(1, 1, 2, 4)


# In[144]:

sph_harm_theta_phi[1, 1+1, A, B]


# In[145]:

A_theta = np.array([[theta_mesh[1],theta_mesh[1]], [theta_mesh[2], theta_mesh[2]]])


# In[146]:

B_phi = np.array([[phi_mesh[3], phi_mesh[4]], [phi_mesh[3], phi_mesh[4]]])


# In[147]:

sph_harmY(1, 1, A_theta, B_phi)


# In[148]:

Phi = np.zeros((10,10), dtype = 'int16')


# In[149]:

Phi[:,:] = np.arange(10)


# In[150]:

print Phi


# In[151]:

print np.transpose(Phi)


# In[152]:

sph_harm_theta_prime[1, 1 + 1, np.transpose(Phi), Phi]


# In[153]:

sph_harm_theta_prime[1, 1 + 1, np.transpose(Phi), Phi].shape


#### Finally, I was able to get the output from sph_harm that I desired. Thanks to Kenny for his input. At least now, can I successfully do the vectorization of theta integration without further hurdles?

# In[154]:

sum(
 sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*\
     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*\
     (I**Lk)*sph_harm_theta_prime[1,1+1, tt, np.arange(10)],\
     ks_wt* psi_new_mesh[Lk](ks_mesh)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])
        ))
       for tt in range(0, len(theta_mesh)) 
   )


# In[159]:

U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0).shape


# In[162]:

U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0)[:,0].shape


# In[163]:

range(1,9)


# In[164]:

np.array(range(1,9))


# In[165]:

np.array(range(1,9)).shape


# In[166]:

a = np.array(range(1,9)).reshape(2,2,2)


# In[167]:

a.shape


# In[168]:

print a


# In[174]:

a1 = np.array(range(1,9))


# In[175]:

print a1


# In[176]:

A1 = np.array(('a', 'b', 'c', 'd'), dtype = object)


# In[177]:

print A1


# In[178]:

a1*A1


# In[179]:

np.tensordot(a1,A1)


# In[208]:

a = np.arange(4.).reshape(2,2)
b = np.arange(4.).reshape(2,2)
c = np.tensordot(a,b,[1,1])


# In[193]:

a


# In[194]:

b


# In[195]:

c


# In[198]:

sum(sum(np.outer(a,b)))


# In[199]:

Lk


# In[206]:

sum(
   theta_wt[i]*phi_wt[j]*ks_wt[k]*(psi_new_mesh[Lk](ks_mesh[k])/norm)*sph_harmY(1, 0, theta_mesh[i], phi_mesh[j])*\
   U_tilde(ks_mesh[k], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[i])+ qq**2/4), Lk, 0, 1, 1, 0)
   for i in range(0, len(theta_mesh))
     for j in range(0, len(phi_mesh))
       for k in range(0, len(ks_mesh))
   )


###### Matches Mathematica answer

# In[ ]:

np.tensordot()


# In[211]:

sph_harm_theta_phi[1,1+1, np.transpose(Phi), Phi].shape


# In[215]:

Phi


# In[216]:

phi_wt


# In[221]:

print np.ones((10,10))*phi_wt


# In[223]:

A = np.ones((10,10))


# In[224]:

A[:,1]


# In[225]:

np.dot(A,phi_wt)


# In[229]:

A*np.transpose(phi_wt)


# In[230]:

A*(phi_wt)


# In[231]:

(phi_wt)*A


# In[232]:

np.transpose(phi_wt)*A


# In[233]:

theta_wt


# In[234]:

phi_wt


# In[240]:

Theta_Wt = np.zeros((10,10))


# In[241]:

Theta_Wt[:,:] = np.transpose(theta_wt)


# In[242]:

Theta_Wt


# In[243]:

np.transpose(Theta_Wt)


# In[244]:

np.transpose(Theta_Wt)[2,0]


# In[245]:

Phi_Wt = np.zeros((10,10))


# In[246]:

Phi_Wt[:,:] = phi_wt 


# In[248]:

print Phi_Wt


# In[249]:

K = np.transpose(Theta_Wt)*sph_harm_theta_phi[1,1+1, np.transpose(Phi), Phi]*Phi_Wt


# In[250]:

K[2,3]


# In[251]:

theta_wt[2]*sph_harmY(1, 1, theta_mesh[2], phi_mesh[3])* phi_wt[3]


###### Works as expected! :-)

# In[261]:

np.dot(np.tensordot( U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0),             np.dot(sph_harm_theta_phi[1,0+1, np.transpose(Phi), Phi]*np.transpose(Theta_Wt),phi_wt),([1,0])),       ks_wt*(psi_new_mesh[Lk](ks_mesh)/norm))


# In[263]:

sum(
    theta_wt[i]*phi_wt[j]*ks_wt[k]*(psi_new_mesh[Lk](ks_mesh[k])/norm)*sph_harmY(1, 0, theta_mesh[i], phi_mesh[j])*\
    U_tilde(ks_mesh[k], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[i])+ qq**2/4), Lk, 0, 1, 1, 0)
    for i in range(0, len(theta_mesh))
      for j in range(0, len(phi_mesh))
        for k in range(0, len(ks_mesh))
     )


# In[273]:

start = timeit.default_timer()
for ttt in range(100):
 sum(
    theta_wt[i]*phi_wt[j]*ks_wt[k]*(psi_new_mesh[Lk](ks_mesh[k])/norm)*sph_harm_theta_phi[1, 0+1, i, j]*\
    U_tilde(ks_mesh[k], np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[i])+ qq**2/4), Lk, 0, 1, 1, 0)
    for i in range(0, len(theta_mesh))
      for j in range(0, len(phi_mesh))
        for k in range(0, len(ks_mesh))
     )
stop = timeit.default_timer()
print stop - start


# In[274]:

start = timeit.default_timer()
for ttt in range(100):
  np.dot(np.tensordot( U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0),             np.dot(sph_harm_theta_phi[1,0+1, np.transpose(Phi), Phi]*np.transpose(Theta_Wt),phi_wt),([1,0])),       ks_wt*(psi_new_mesh[Lk](ks_mesh)/norm))
stop = timeit.default_timer()    
print stop - start


# In[275]:

5.94348812103/0.00807809829712


##### Hooray! Vectorization gives the same answer and is about 1000 times faster!

# In[284]:

sum(
 sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*\
     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*\
     (I**Lk)*sph_harm_theta_prime[1,1+1, tt, np.arange(10)],\
     ks_wt* (psi_new_mesh[Lk](ks_mesh)/norm)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])\
        +np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*\
     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*\
     (I**Lk)*sph_harm_theta_prime2[1,1+1, tt, np.arange(10)],\
     ks_wt* (psi_new_mesh[Lk](ks_mesh)/norm)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])
        ))
       for tt in range(0, len(theta_mesh)) 
   )


# In[278]:

Theta_Mesh = np.zeros((10,10))
Theta_Mesh[:,:] = np.transpose(theta_mesh)


# In[283]:

np.dot(np.tensordot( U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0),             np.dot(sph_harm_theta_phi[1,1+1, np.transpose(Phi), Phi].conjugate()*np.transpose(Theta_Wt)*                    sph_harm_theta_prime[1, 1+1, np.transpose(Phi), Phi]*np.sin(np.transpose(Theta_Mesh)),phi_wt),([1,0])),       ks_wt*(I**Lk)*(psi_new_mesh[Lk](ks_mesh)/norm)*ks_mesh**2)+np.dot(np.tensordot( U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)[np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4))], Lk, 0, 1, 1, 0)[:,np.argsort(np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)))],             np.dot(sph_harm_theta_phi[1,1+1, np.transpose(Phi), Phi].conjugate()*np.transpose(Theta_Wt)*                    sph_harm_theta_prime2[1, 1+1, np.transpose(Phi), Phi]*np.sin(np.transpose(Theta_Mesh)),phi_wt),([1,0])),       ks_wt*(I**Lk)*(psi_new_mesh[Lk](ks_mesh)/norm)*ks_mesh**2)


####### Same answer!

# In[285]:

start = timeit.default_timer()
for n in range(1000):
 sum(
  sum(sum(np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*\
     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*\
     (I**Lk)*sph_harm_theta_prime[1,1+1, tt, np.arange(10)],\
     ks_wt* (psi_new_mesh[Lk](ks_mesh)/norm)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])\
        +np.outer(theta_wt[tt]* phi_wt* np.sin(theta_mesh[tt])*\
     sph_harm_theta_phi[1, 1+1, tt, np.arange(10)].conjugate()*\
     (I**Lk)*sph_harm_theta_prime2[1,1+1, tt, np.arange(10)],\
     ks_wt* (psi_new_mesh[Lk](ks_mesh)/norm)*(ks_mesh**2)*U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh[tt])+ qq**2/4), Lk, 0, 1, 1, 0)[:,0])
        ))
       for tt in range(0, len(theta_mesh)) 
   )
stop = timeit.default_timer()
print stop -start


# In[290]:

start = timeit.default_timer()
for n in range(1):
  np.dot(np.tensordot( U_tilde(ks_mesh, np.sqrt(pp**2 - pp* qq* np.cos(theta_mesh)+ qq**2/4), Lk, 0, 1, 1, 0),             np.dot(sph_harm_theta_phi[1,1+1, np.transpose(Phi), Phi].conjugate()*np.transpose(Theta_Wt)*                    sph_harm_theta_prime[1, 1+1, np.transpose(Phi), Phi]*np.sin(np.transpose(Theta_Mesh)),phi_wt),([1,0])),       ks_wt*(I**Lk)*(psi_new_mesh[Lk](ks_mesh)/norm)*ks_mesh**2)+  np.dot(np.tensordot( U_tilde(ks_mesh, np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)[np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4))], Lk, 0, 1, 1, 0)   [:,np.argsort(np.argsort(np.sqrt(pp**2 + pp* qq* np.cos(theta_mesh)+ qq**2/4)))],             np.dot(sph_harm_theta_phi[1,1+1, np.transpose(Phi), Phi].conjugate()*np.transpose(Theta_Wt)*                    sph_harm_theta_prime2[1, 1+1, np.transpose(Phi), Phi]*np.sin(np.transpose(Theta_Mesh)),phi_wt),([1,0])),       ks_wt*(I**Lk)*(psi_new_mesh[Lk](ks_mesh)/norm)*ks_mesh**2)
stop = timeit.default_timer()
print stop -start    


# In[289]:

1.49293708801/0.246959209442


###### Thus, vectorization of theta integral successful! However, the answer differs from Mathematica answer at the third decimal place. Understand why this is the case.

# In[291]:

pp


# In[292]:

qq


# In[ ]:



