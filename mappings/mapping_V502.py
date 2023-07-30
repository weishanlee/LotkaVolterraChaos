# -*- coding: utf-8 -*-
"""
This script is intended to plot
1. bifurcation
2. lyapunov exponents

V5: mapping of Version 5
    f(x,y) = mu0 * x * ( 1 - x ) - mu1 * x * y
    g(x,y) =-nu0 * y * ( 1 - y ) + nu1 * x * y

Modification Note: 
      1. Discard vpython and make use of matplotlib
      2. Slice mu0ListLP, sumLPX, and sumLPY in the `if RoEck == True:` block
      3. Plot the x vs.y
      4. Add try-except statement 
      5. Remove Plot of x vs y
      6. discard gridspec.GridSpec for plots
      7. Modify def newAttractor, def jacobian, and def func from pplane.py
      8. Change simCase instructions a little bit.
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
#import nolds
from pathlib import Path
from numpy import linalg as LA
import warnings
warnings.filterwarnings("error")
import sys

tolerance = 1e-8
 
mu0_min = 1
mu0_max = 4

# Calculate Lyapunov for Rosenstein and Eckmann Algorithm
RoEck = True  

# zoom in along x direction?
xZoomIn = False

# initX and initY are numbers between 0 and 1
# alpha, beta, and gamma are all greater than 0
# For E3 lies in the 1st quardrant, either {alpha<1 or gamma<beta}
# or either {alpha>1 or gamma>beta}
###############################################################################
simCase = "Standard"

if simCase == "Trial":
   initX, initY, alpha, beta, gamma = 0.500, 0.100, 0.000, 0.000, 0.000
   step  = 0.005 
   RoEck = False 
   dirname = "{:.5f}".format(initX)+\
              "_{:.5f}".format(initY)+\
              "_{:.5f}".format(alpha)+\
              "_{:.5f}".format( beta)+\
              "_{:.5f}".format(gamma)
#elif simCase == "Paraclete":
#    initX, initY, alpha, beta, gamma = 0.010, 0.100, 5.000, 0.010, 0.900 # paraclete
#    if xZoomIn == True:
#       step  = 0.001
#    else:
#       step  = 0.005
elif simCase == "Normal": # ok
    initX, initY, alpha, beta, gamma = 0.200, 0.200, 1.000, 0.001, 0.500 # normal 
    if xZoomIn == True:
       step  = 0.001
    else:
       step  = 0.005
elif simCase == "Standard":   
    initX, initY, alpha, beta, gamma = 0.100, 0.500, 1.000, 0.100, 0.500 # standard 
    if xZoomIn == True:
       step  = 0.001
    else:
       step  = 0.005
elif simCase == "Extinction": # ok
    initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.500 # extinction
    step  = 0.001
elif simCase == "Vorticella": 
    initX, initY, alpha, beta, gamma = 0.010, 0.100, 5.000, 0.001, 0.900 # vorticella (bell-shaped)
    if xZoomIn == True:
       step  = 0.001
    else:
       step  = 0.005
#elif simCase == "VS":
#    initX, initY, alpha, beta, gamma = 0.100, 0.500, 0.875, 0.018, 1.000 # vorticella strange
#    step  = 0.001
else:
    sys.exit("Wrong simCase!")
dirname = simCase
###############################################################################

Path(dirname).mkdir(parents=True, exist_ok=True)
cwd = "./"+str(dirname)+"/"
logFile = str(cwd)+"log.txt"

import os #, psutil
# If previous log.txt file exists, remove it.
if os.path.exists(logFile):
    os.remove(logFile)

def writeLog(msg):
    with open(logFile, 'a+') as the_file:
        print(msg)
        the_file.write(msg)
        the_file.close()

writeLog("simCase: {}\n".format(simCase))
writeLog(r'init X = '+ "{:.5f}\n".format(initX))
writeLog(r'init Y = '+ "{:.5f}\n".format(initY))
writeLog(r'alpha = '+ "{:.5f}\n".format(alpha))
writeLog(r'beta = '+ "{:.5f}\n".format(beta))
writeLog(r'gamma = '+ "{:.5f}\n".format(gamma))

mu0ListLM = []  # mu0 list for logistic map
#mu0ListLP = []  # mu0 list for lyapunov
ptsX = []       # points for logistic map X
ptsY = []       # points for logistic map Y

sumLPX = []     # sum of liapunov for X
sumLPY = []     # sum of liapunov for Y

if RoEck == True:
    import nolds
    lya_e_X1 = []   # lyaponov exponents for X (Eckmann)
    lya_e_X2 = []  # lyaponov exponents for X (Eckmann) # Bad result
    lya_e_Y1 = []  # lyaponov exponents for Y (Eckmann) # Bad result
    lya_e_Y2 = []   # lyaponov exponents for Y (Eckmann)
    lya_r_X = []    # lyaponov exponents for X (Rosenstein)
    lya_r_Y = []    # lyaponov exponents for Y (Rosenstein)

def newAttractor(x, y, mu0):
    xx =  mu0 * x * ( 1 - x ) - mu0 * alpha * x * y
    yy = -mu0 * beta * y * ( 1 - y ) + mu0 * gamma * x * y
    return xx, yy

def func(variables,mu0):
    x, y = variables
    dx, dy = newAttractor(x, y, mu0)
    return (dx, dy)
 
def jacobian(fs, xs, mu0, h=1e-4):
    """
    Reference: Gezerlis, Numerical Methods in Phyisics with Python, p.284
    """
    n = np.asarray(xs).size
    iden = np.identity(n)
    Jf = np.zeros((n,n))
    fs0 = fs(xs,mu0)
    for j in range(n):  # through columns to allow for vector addition
        fs1 = fs(xs+iden[:,j]*h,mu0)
        Jf[:,j] = ( np.asarray(fs1) - np.asarray(fs0) )/h
    return Jf    
    

def plotFiguresSameX(mu0ListLM, ptsX, ptsY):
    fig, ax = plt.subplots(2, 1, figsize=(16,9), constrained_layout=True, sharex=True)
    
    # the first subplot
    ax[0].scatter(mu0ListLM,ptsX,color = 'k' , s = 0.1 , marker = '.')
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax[0].set_ylabel("$x$",size = 16)
    ax[0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    if xZoomIn == True:
       ax[0].set_xlim(2.8,4.0)
    ax[0].grid(True)
    
    # the second subplot
    # shared axis X
    ax[1].scatter(np.array(mu0ListLM),ptsY,color = 'k', s = 0.1, marker = '.')
    ax[1].set_xlabel(r'$\mu_{0}$',size = 16)
    ax[1].set_ylabel("$y$",size = 16)
    ax[1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    if xZoomIn == True:
       ax[1].set_xlim(2.8,4.0)
    ax[1].grid(True)

    plt.savefig(str(cwd)+str(simCase)+"_"+"LogisticMap.png")
    
def outPutLyaExp(mu0ListLP, EqnLx, RLx, ELx1, ELx2, EqnLy, RLy, ELy1, ELy2, omitRosenstein):
    if omitRosenstein == False:
       data = {'mu0ListLP': mu0ListLP,'EqnLx': EqnLx,'RLx':RLx, 'ELx1':ELx1, 'ELx2':ELx2,
                                      'EqnLy': EqnLy,'RLy':RLy, 'ELy1':ELy1, 'ELy2':ELy2}
       df = pd.DataFrame(data)
       df_file = open(str(cwd)+"LyapunovExponents.csv",'w',newline='')
       df.to_csv(df_file, sep=',', encoding='utf-8',index=False)
       df_file.close()
    else:
       data = {'mu0ListLP': mu0ListLP,'EqnLx': EqnLx, 'ELx1':ELx1, 'ELx2':ELx2,
                                      'EqnLy': EqnLy, 'ELy1':ELy1, 'ELy2':ELy2}
       df = pd.DataFrame(data)
       df_file = open(str(cwd)+str(simCase)+"LyapunovExponents.csv",'w',newline='')
       df.to_csv(df_file, sep=',', encoding='utf-8',index=False)
       df_file.close() 
   
nIterations = 2500

mu = np.arange(mu0_min, mu0_max, step)
muList = []
# Find suitable muList
for mu_ in mu: 
    traceX = np.zeros(nIterations)
    traceY = np.zeros(nIterations)
    traceX[0] = initX
    traceY[0] = initY

    # version 2
    i = 0 
    exception = False
    while (i < nIterations - 1):
        try:
            x_next, y_next = newAttractor(traceX[i], traceY[i], mu_)
        except RuntimeWarning:
            exception = True
            if x_next > 0:
                print("Warning Line 192! x_next positive infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                break
            else:
                print("Warning Line 195! x_next negative infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                break
            if y_next > 0:
                print("Warning Line 198! y_next positive infinitle! y_next = {} at mu_ = {}".format(y_next,mu_))
                break
            else:
                print("Warning Line 201! y_next negative infinite! y_next = {} at mu_ = {}".format(y_next,mu_))
                break
        else:
            if x_next < 0 or x_next >1:
                exception = True
                print("Warning Line 208! x_next out of bound x_next = {} at mu_ = {}".format(x_next, mu_))
                break
            if y_next < 0 or y_next >1:
                exception = True
                print("Warning Line 214! y_next out of bound y_next = {} at mu_ = {}".format(y_next, mu_))
                break
            traceX[i+1] = x_next
            traceY[i+1] = y_next                 
            i += 1
    if exception == False:
        muList += [mu_]
    else:
        writeLog(r'mu0_max = '+ "{:.5f}\n".format(muList[-1]))
        break

mu = muList 
# end of find suitable muList 

# plot bifurcation
nIterations = 600 # Reduce nIterations if the black area is large. 
avoidTransient = 200
for mu_ in mu:
    x_next = initX
    y_next = initY
    i = 0 
    exception = False     
    while (i<nIterations-1):
        try:
            x_next, y_next = newAttractor(x_next, y_next, mu_)
        except RuntimeWarning:
            exception = True 
            if x_next > 0:
                print("Warning Line 242! x_next positive infinite! mu_ = {}, x_next = {}".format(mu_,x_next))
                break
            else:
                print("Warning Line 245! x_next negative infinite! mu_ = {}, x_next = {}".format(mu_, x_next))
                break
            if y_next > 0:
                print("Warning Line 248! y_next positive infinitle! mu_ = {}, y_next = {}".format(mu_, y_next))
                break
            else:
                print("Warning Line 251! y_next negative infinite! mu_ = {}, y_next = {}".format(mu_, y_next))
                break
        else:
            if x_next < 0 or x_next>1:
                exception = True
                print("Warning Line 258! x_next out of bound x_next = {} at mu_ = {}".format(x_next, mu_))
                break
            if y_next < 0 or y_next>1:
                exception = True
                print("Warning Line 264! y_next out of bound y_next = {} at mu_ = {}".format(y_next, mu_))
                break
            if i >= avoidTransient:
                mu0ListLM += [mu_]
                ptsX += [x_next]
                ptsY += [y_next]
                w = LA.eigvals( jacobian(func,(x_next, y_next), mu_ ) )
            i += 1
    if exception == True:
        break
    else:
        try:
            np.log( abs((w[0]) ) )
        except RuntimeWarning:
            sumLPX += [-float('inf')]
        else:
            sumLPX += [np.log( abs((w[0]) ) ) / np.log(2)]
        try:
            np.log( abs((w[1]) ) )
        except RuntimeWarning:
            sumLPY += [-float('inf')]
        else:
            sumLPY += [np.log( abs((w[1]) ) ) / np.log(2)]

plotFiguresSameX(mu0ListLM, ptsX, ptsY)
# end of plot of bifurcation
#%%
omitRosenstein = False
if RoEck == True:
    warnings.filterwarnings('ignore') #"default" 
    step1 = step
    step = 0.01 
    nIterations = 2500
    
    gap = int(step / step1)
    mu0ListLP = mu[::gap] # step changes from step1 to step
    sumLPX = sumLPX[::gap]
    sumLPY = sumLPY[::gap]
    
    mu = mu0ListLP
    for mu_ in mu:
        X = np.zeros(nIterations)
        Y = np.zeros(nIterations)
        X[0] = initX
        Y[0] = initY
        
        i = 0
        exception = False
        while (i < nIterations - 1):
            try:
                x_next, y_next = newAttractor(X[i], Y[i], mu_)
            except RuntimeWarning:
                exception = True
                if x_next > 0:
                    print("Warning Line 315! x_next positive infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                    break
                else:
                    print("Warning Line 318! x_next negative infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                    break
                if y_next > 0:
                    print("Warning Line 321! y_next positive infinitle! y_next = {} at mu_ = {}".format(y_next,mu_))
                    break
                else:
                    print("Warning Line 324! y_next negative infinite! y_next = {} at mu_ = {}".format(y_next,mu_))
                    break
            else:
                if x_next < 0 or x_next>1:
                    exception = True
                    print("Warning Line 331! x_next out of bound x_next = {} at mu_ = {}".format(x_next, mu_))
                    break
                if y_next < 0 or y_next>1:
                    exception = True
                    print("Warning Line 337! y_next out of bound y_next = {} at mu_ = {}".format(y_next, mu_))
                    break
                X[i+1] = x_next
                Y[i+1] = y_next
                i += 1 
        if exception == True: 
            print("Warning! mu0List error! Line 343")
            break
        else:
            try:
                lya_r_X += [ nolds.lyap_r(X, emb_dim=10)/np.log(2) ]
                lya_r_Y += [ nolds.lyap_r(Y, emb_dim=10)/np.log(2) ]
            except np.linalg.LinAlgError:
                if omitRosenstein == False:
                   writeLog("SVD did not converge in linear least square. Rosenstein omitted.\n")
                   print("SVD did not converge in linear least square. Rosenstein omitted.")
                   omitRosenstein = True
                lya_r_X = []
                lya_r_Y = []
            finally:
                lya_e_X1 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                lya_e_X2 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]
                lya_e_Y1 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                lya_e_Y2 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]
    # Write Lyaponov Exponents to csv file
    outPutLyaExp(mu0ListLP, sumLPX, lya_r_X, lya_e_X1, lya_e_X2, 
                            sumLPY, lya_r_Y, lya_e_Y1, lya_e_Y2, omitRosenstein)
#%%
# plot lyapunov exponents
fig, ax = plt.subplots(2, 1, figsize=(16,9), constrained_layout=True, sharex=True)

if RoEck == True:
    # the first subplot
    ax[0].plot(np.array(mu0ListLP),np.array(sumLPX),'b-.')
    if omitRosenstein == False:
       ax[0].plot(np.array(mu0ListLP),np.array(lya_r_X),'k:')
       ax[0].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
       ax[0].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')
       ax[0].legend(["Equation", "Rosenstein", "Eckmann X", "Eckmann Y"], loc ="lower right")
    else:
       if simCase == "Standard":
          ax[0].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
          ax[0].legend(["Equation", "Eckmann X"], loc ="lower right")
       else:
          ax[0].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
          ax[0].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')
          ax[0].legend(["Equation", "Eckmann X", "Eckmann Y"], loc ="lower right")
else:
    ax[0].plot(np.array(mu),np.array(sumLPX),'b-.')
    
plt.minorticks_on()
minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(5)
#ax0.set_xlabel("$\\mu_{0}$",size = 16)
ax[0].set_ylabel("$\\lambda_{x}$",size = 16)
ax[0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
if xZoomIn == True:
   ax[0].set_xlim(2.8,4.0)
ax[0].set_ylim([-8,2])
ax[0].grid(True)

if RoEck == True:
    # the second subplot
    # shared axis X
    ax[1].plot(np.array(mu0ListLP),np.array(sumLPY),'b-.')
    if omitRosenstein == False:
       ax[1].plot(np.array(mu0ListLP),np.array(lya_r_X),'k:')
       ax[1].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
       ax[1].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')
       ax[1].legend(["Equation", "Rosenstein", "Eckmann X", "Eckmann Y"], loc ="lower right")
    else:
       if simCase == "Standard":
          ax[1].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
          ax[1].legend(["Equation", "Eckmann X"], loc ="lower right")
       else:
          ax[1].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
          ax[1].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')  
          ax[1].legend(["Equation", "Eckmann X", "Eckmann Y"], loc ="lower right")
else:
    ax[1].plot(np.array(mu),np.array(sumLPY),'b-.')
      
ax[1].set_xlabel(r'$\mu_{0}$',size = 16)
ax[1].set_ylabel("$\\lambda_{y}$",size = 16)
ax[1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
if xZoomIn == True:
   ax[1].set_xlim(2.8,4.0)
#plt.ylim(-5.2)
ax[1].grid(True)

# remove vertical gap between subplots
#plt.show()
plt.savefig(str(cwd)+str(simCase)+"_"+"LyapunovExponents.png")