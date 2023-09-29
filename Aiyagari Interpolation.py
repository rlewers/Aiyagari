#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:46:52 2023

@author: rileylewers
"""

import numpy as np
import math
import random
from scipy.interpolate import interp1d
import time 
import matplotlib.pyplot as plt

start = time.time()

# Model Parameters
beta = 0.96 # discount rate
alpha = 0.33 # capital share
delta = 0.1 # depreciation rate 
gamma = 1.25 # risk aversion

ygrid = [0.3, 1] # productivity "shock" values 
ny = len(ygrid)
TransM = [[0.75, 0.25], [0.25, 0.75]] # transition probabilities matrix 

#piinfty = TransM**1000 # stationary distribution for aggregate labor supply

piinfty = TransM
for i in range(100):
    piinfty=np.matmul(piinfty, TransM)

# first and second rows of the array are the same now
# extract first row to be piinfty
piinfty=piinfty[0]

# first element is fraction of workers that are low productivity 
# second element is fraction of workers that are high productivity
# so multiply by productivity of each type of worker to get aggregate 
# labor supply
L = np.matmul(piinfty,ygrid) # aggregate labor supply

# AGrid 
na = 200
a_min = 0 
a_max = 10 
agrid = np.linspace(a_min, a_max, na)

# Iteration Tolerance 
tol_for_vfi = 1e-6
tol_for_loop = 1e-6

# Setting up some stuff to use simulation method 
T = 10000 # this will be the number of agents 
burn_in = 5000 
np.random.seed(1020) # set this so simulation results will be the same each time
ysimul = np.zeros(T) # col vector w/ 10000 rows 
# don't understand why he does this
# I guess it sets the first agent to low productivity
ysimul[0]=ygrid[0] 

# filling the vector of productivity realizations for all agents in the economy 
# not sure why he does it this way...
for t in range(1, T):
    #prestate = ysimul[t-1] == ygrid 
    prestate = np.where(ysimul[t-1] == ygrid)[0][0]
    generate = np.random.uniform(0, 1)
    if generate < TransM[prestate][1]: # bit confused by this if condition
        ysimul[t] = ygrid[0]
    else:
        ysimul[t] = ygrid[1]
        

# VFI Set up 
# discounted maximized lifetime value given assets and productivity
V = np.zeros([na, ny]) 
# optimal action to max discounted lifetime value given assets and productivity
pol_func = np.zeros([na, ny])
Vnew = np.zeros([na, ny])

# Solution Algorithm 
'''We have the HH problem (ex-ante identical, but HA after income realization), 
the representative firm's problem, and the defintion of a stationary equilibrium 
(what we want to find). To find eqm, we need to implement the following algorithm:
    1. Guess K
    2. Using firm's FOCs, solve for r and w
    3. Given r and w, determine HH optimal savings (use vfi)
    4. From HH savings, aggregate up to get K (DK uses simulation method)
    5. If this K agrees w/ our initial guess, we are done. Otherwise, repeat.'''

Kinterv = [2, 6]
lambd = np.zeros([1,na])
diff_in_main = 1 # tells us whether our guess of K is close enough

for i in range(10000): # I guess this is like setting max iterations?
    #K=(Kinterv[0]+Kinterv[1])/2 # guess aggregate capital level 
    K=np.mean(Kinterv)
    # representative firm's optimality conditions determine R and W
    R = (alpha)*(K/L)**(alpha-1)+1-delta 
    W = (1-alpha)*(K/L)**(alpha)
    
    diff_in_valfun=1 # tells us whether our value function is close enough
    k=0
    while diff_in_valfun > tol_for_vfi:
        k+=1
        
        Vnew = V
        V = np.zeros([na, ny])
        pol_func = np.zeros([na, ny])
        con = np.zeros([na, ny])
        
        # loop over productivity levels 
        for y_ind in range(ny):
            # loop over asset choices 
            for a_ind in range(na):
                wealth = R*agrid[a_ind] + W*ygrid[y_ind]
                #wealth = wealth*na
                #max1 = max(np.subtract(wealth,agrid))
                #if y_ind==0 and a_ind==0:
                    #print(max1)
                # the below is a 200x1 row vector. It is like 1 row in Carlos's program
                V_max=np.maximum(np.subtract(wealth,agrid), 1.0e-15)**(1-gamma)/(1-gamma)+beta*(np.matmul(Vnew, TransM[y_ind][:]))
                
                # place stuff into the correct spot in the value, policy, and consumption matrices
                V[a_ind, y_ind] = np.max(V_max)
                #if y_ind==0 and a_ind==0 and i==0:
                    #print(np.max(V_max))
                temp = np.argmax(V_max)
                pol_func[a_ind, y_ind] = agrid[temp]
                con[a_ind, y_ind] = wealth - pol_func[a_ind, y_ind]
        
        diff_in_valfun = np.max(np.max(abs(V-Vnew))) # thanks Elke!
       # if i==0 and k==1:
           #print( V)
        
    # Do simulation method to get K 
    
    # my attempt at doing the simulation in a way that makes more sense to me

    # T = 10000 # 10000 agents in the economy

    # make a guess at the initial number of agents in each state
    #Kgrid = np.zeros([na, ny]) #matrix to keep track of number of agents in each state
    #Kgrid.fill(25) # start with agents evenly divided among the states 
    
    na_finer=400
    Kgrid = np.zeros([na_finer, ny]) #matrix to keep track of number of agents in each state
    # 10 people in each state, 800 states --> 8000 people in this economy
    T=8000
    Kgrid.fill(10) # start with agents evenly divided among the states 
    
    # make policy function finer as well by interpolating the function values from 
    # pol_func onto a_gridfiner 
    a_gridfiner =np.linspace(a_min,a_max,na_finer)
    #pol_func_finer = np.interp(agrid, pol_func, a_gridfiner)
    #f = interp1d(agrid, pol_func, axis=0, kind='linear')
    #pol_func_finer = f(a_gridfiner)
    #pol_func_finer = np.interp(a_gridfiner, agrid, pol_func_high, axis=0)
    
    # separate policy function into 2 columns 
    pol_func_low = pol_func[:,0]
    pol_func_high = pol_func[:,1]
    
    #interpolate separately for low and high policy functions 
    pol_func_finer_high = np.interp(a_gridfiner, agrid, pol_func_high)
    pol_func_finer_low = np.interp(a_gridfiner, agrid, pol_func_low)
    pol_func_finer = np.column_stack((pol_func_finer_low, pol_func_finer_high))

    # now iterate using the transition matrix and the policy function 
    # until a stationary distribution is reached 
    diff_in_Kgrid=1
    k = 0
    while diff_in_Kgrid > tol_for_vfi:
            k+=1
            #print(k)
            Kgridnew = np.zeros([na_finer, ny])
            for y_ind in range(ny):
                    for a_ind in range(na_finer):
                        # use policy function to get agent's optimal choice 
                        # of assets next period assets
                        next_a=pol_func_finer[a_ind][y_ind]
                        # tells us at which index is the value closest to next_a
                        a_index = (np.abs(a_gridfiner - next_a)).argmin()
                        #a_index = np.where(a_gridfiner==next_a)[0][0]
                        
                        # use transition probability matrix to get 
                        # the fraction that move to each productivity outcome next period
                        produc_vec_0=float(Kgrid[a_ind][y_ind])*TransM[y_ind][0]
                        produc_vec_1=float(Kgrid[a_ind][y_ind])*TransM[y_ind][1]
                        
                        # add the mass of agents coming from this a and y combo
                        # to the new Kgrid
                        Kgridnew[a_index][0]+=produc_vec_0
                        Kgridnew[a_index][1]+=produc_vec_1
                        
            diff_in_Kgrid = np.max(np.max(abs(Kgrid-Kgridnew)))
            #print(diff_in_Kgrid)
            Kgrid=Kgridnew
    
    # sum all the elements to get aggregate K
    Kagg=0
    for a_ind in range(na_finer):
        Kagg+=a_gridfiner[a_ind]*Kgridnew[a_ind][0]+a_gridfiner[a_ind]*Kgridnew[a_ind][1]
    Kprime=Kagg/T
    
    # apply bisection method to get a new guess for K that is updated to move our 
    # K interval closer to what makes sense given the outcome of the simulation 
    if Kprime>K:
        Kinterv[0] = K 
    else:
        Kinterv[1] = K 
        
    diff_in_main = np.diff(Kinterv)
    print(f'Iteration #: {iter}, Capital Interval Difference: {diff_in_main}')
    
    # if we have narrowed in on a K i.e. K interval is very small, we are done
    if diff_in_main<tol_for_loop: 
        break

# now that algorithm is complete, get final values for K and R
Kstar = np.mean(Kinterv);
Rstar = (alpha)*(Kstar/L)**(alpha-1)+1-delta

# display final values 
print(f'Equilibrium Capital: {Kstar}')
print(f'Equilibrium Interest Rate: {Rstar}')

##############
# Make Plots #
##############

# Value Function 
plt.subplot(1, 2, 1)
plt.plot(agrid, V[:,0], label="Low")
plt.plot(agrid, V[:,1], label="High")
plt.legend()
plt.xlabel("a")
plt.ylabel("V")

# Policy Function
plt.subplot(1, 2, 2)
plt.plot(agrid, pol_func[:,0], label="Low")
plt.plot(agrid, pol_func[:,1], label="High")
plt.legend()
plt.xlabel("a")
plt.ylabel("g(a,e)")

plt.tight_layout()
plt.show()

# Unconditional Wealth Distribution 

# make unconditional wealth vector 
Kadd=np.zeros(na_finer)
for k in range(na_finer):
    Kadd[k]=Kgrid[k][0]+Kgrid[k][1]

plt.subplot(1,2,1)    
plt.plot(a_gridfiner, Kadd/8000)
plt.legend()
plt.xlabel("a")
plt.ylabel("lambda")
plt.title("Unconditional Wealth Distribution")    
    
# Conditional Wealth Distribution 
plt.subplot(1,2,2)
plt.plot(a_gridfiner, Kgrid[:,0], label="Low")
plt.plot(a_gridfiner, Kgrid[:,1], label="High")
plt.legend()
plt.xlabel("a")
plt.ylabel("lambda(a,e)")
plt.title("Conditional Wealth Distribution")
plt.tight_layout()
plt.show()

# Aiyagari Diagram

# get asset supply by doing vfi and k simulation without making firms focs firm
# asset demand match to HH asset supply

Rplot=np.linspace(1, 1.1, 100) # list of R options 
KSupp=np.zeros(len(Rplot))
KDem=np.zeros(len(Rplot))
for r in range(len(Rplot)):
    K=(((Rplot[r]+delta-1)/alpha)**(1/(alpha-1)))*L
    #K=((Rplot[r]+delta-1)**(1/(alpha-1)))*(L/alpha)
    # firm asset demand vector
    KDem[r]=K
    W =(1-alpha)*(K/L)**(alpha)
    pol_func=np.zeros([na,ny]) # How much should the agent save?
    Vnew=np.zeros([na,ny])
    
    # vfi is the same as before
    
    diff_in_valfun=1 # tells us whether our value function is close enough
    k=0
    while diff_in_valfun > tol_for_vfi:
        Vnew = V
        V = np.zeros([na, ny])
        pol_func = np.zeros([na, ny])
        con = np.zeros([na, ny])
        k+=1
        for y_ind in range(ny):
            # loop over asset choices 
            for a_ind in range(na):
                wealth = Rplot[r]*agrid[a_ind] + W*ygrid[y_ind]
                #wealth = wealth*na
                #max1 = max(np.subtract(wealth,agrid))
                #if y_ind==0 and a_ind==0:
                        #print(max1)
                # the below is a 200x1 row vector. It is like 1 row in Carlos's program
                V_max=np.maximum(np.subtract(wealth,agrid), 1.0e-15)**(1-gamma)/(1-gamma)+beta*(np.matmul(Vnew, TransM[y_ind][:]))
                    
                # place stuff into the correct spot in the value, policy, and consumption matrices
                V[a_ind, y_ind] = np.max(V_max)
                #if y_ind==0 and a_ind==0 and i==0:
                        #print(np.max(V_max))
                temp = np.argmax(V_max)
                pol_func[a_ind, y_ind] = agrid[temp]
                con[a_ind, y_ind] = wealth - pol_func[a_ind, y_ind]
            
        diff_in_valfun = np.max(np.max(abs(V-Vnew)))
        
    # now do simulation to get asset supply
    # need to know for each R, what will HH asset supply be
    
    # make a guess at the initial number of agents in each state
    Kgrid = np.zeros([na, ny]) #matrix to keep track of number of agents in each state
    Kgrid.fill(25) # start with agents evenly divided among the states 
    
    diff_in_Kgrid=1
    #k = 0
    while diff_in_Kgrid > tol_for_vfi:
            #k+=1
            #print(k)
            Kgridnew = np.zeros([na, ny])
            for y_ind in range(ny):
                    for a_ind in range(na):
                        # use policy function to get agent's optimal choice 
                        # of assets next period assets
                        next_a=pol_func[a_ind][y_ind]
                        a_index = np.where(agrid==next_a)[0][0]
                        
                        # use transition probability matrix to get 
                        # the fraction that move to each productivity outcome next period
                        produc_vec_0=float(Kgrid[a_ind][y_ind])*TransM[y_ind][0]
                        produc_vec_1=float(Kgrid[a_ind][y_ind])*TransM[y_ind][1]
                        
                        # add the mass of agents coming from this a and y combo
                        # to the new Kgrid
                        Kgridnew[a_index][0]+=produc_vec_0
                        Kgridnew[a_index][1]+=produc_vec_1
                        
            diff_in_Kgrid = np.max(np.max(abs(Kgrid-Kgridnew)))
            #print(diff_in_Kgrid)
            Kgrid=Kgridnew
    
    # sum all the elements to get aggregate K
    Kagg=0
    for a_ind in range(na):
        Kagg+=agrid[a_ind]*Kgridnew[a_ind][0]+agrid[a_ind]*Kgridnew[a_ind][1]
    Kprime=Kagg/10000
    KSupp[r]=Kprime
    
# Asset Supply 
plt.plot(KSupp,Rplot)
plt.plot(KDem,Rplot)
plt.xlabel("a")
plt.ylabel("R")
plt.title("Aiyagari Diagram")
plt.show()  

end = time.time()
run_time=end - start
print(f'Program takes {run_time} seconds')

        
        
    


