#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:35:35 2023

# based on DK's matlab code from second quarter

@author: rileylewers
"""

import numpy as np
import math
import random
from scipy.interpolate import interp1d

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
        if i==0 and k==1:
           print( V)
        
    # Do simulation method to get K 
    
    # my attempt at doing the simulation in a way that makes more sense to me

    # T = 10000 # 10000 agents in the economy

    # make a guess at the initial number of agents in each state
    Kgrid = np.zeros([na, ny]) #matrix to keep track of number of agents in each state
    Kgrid.fill(25) # start with agents evenly divided among the states 

    # now iterate using the transition matrix and the policy function 
    # until a stationary distribution is reached 
    diff_in_Kgrid=1
    k = 0
    while diff_in_Kgrid > tol_for_vfi:
            k+=1
            print(k)
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
            print(diff_in_Kgrid)
            Kgrid=Kgridnew
    
    # sum all the elements to get aggregate K
    Kagg=0
    for a_ind in range(na):
        Kagg+=agrid[a_ind]*Kgridnew[a_ind][0]+agrid[a_ind]*Kgridnew[a_ind][1]
    Kprime=Kagg/10000
    '''
    # get a finer agrid 
    a_gridfiner = np.linspace(a_min,a_max,int(1.5*na))
    
    # Use linear interpolation to get finer policy function corresponding 
    # to our finer agrid
    # we need to specify axis=0 b/c rows in the policy function array 
    # correspond to agrid (asset) values
    # I think this is the corresponding python syntax to do what is in DK's code
    # but I'm not 100% sure
    f = interp1d(agrid, pol_func, axis=0, kind='linear')
    pol_func_finer = f(a_gridfiner)
    
    # initialize a vector to put our simulated asset values into
    a_sim = np.zeros([T,1])
    a_sim[0]=a_gridfiner[0]
    
    # Loop to update a_sim
    # need to better understand what is going on here
    for t in range(T-1):
        y_idx = np.where(ysimul[t] == ygrid)[0][0]  # Find the index where ysimul(t) matches ygrid
        a_sim[t + 1] = interp1d(a_gridfiner, pol_func_finer[:, y_idx], kind='linear', fill_value='extrapolate')(a_sim[t])
    
    # also confused about why this is necessary 
    #a_sim = a_sim[burn_in+1:T]
    a_sim = a_sim[burn_in:T]
    y_sim = ysimul[burn_in:T]
    
    # make new asset vectors splitting the original a_sim into high and low 
    # productivity groups 
    a_sim1 = a_sim[y_sim == ygrid[0]]
    a_sim2 = a_sim[y_sim == ygrid[1]]
    
    # use numpy histogram to count the number of occurences of assets that fall 
    # into each of the grid points in a_gridfiner
    a_grid_hist = np.append(a_gridfiner, math.inf) # need a final bin 
    ncount, _ = np.histogram(a_sim, a_grid_hist)
    ncount1, _ = np.histogram(a_sim1, a_grid_hist)
    ncount2, _ = np.histogram(a_sim2, a_grid_hist)
    
    # get the probability of being in each of the bins
    lambd = np.divide(ncount, len(a_sim))
    lambd1 = np.divide(ncount1, len(a_sim))
    lambd2 = np.divide(ncount2, len(a_sim))
    
    # multiply element by element the probability of being in each asset bin 
    # times the amount of assets for someone in that bin 
    # then sum all the elements 
    Kprime = np.sum(np.multiply(a_gridfiner, lambd))
    '''
    
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

        
        
    



    
