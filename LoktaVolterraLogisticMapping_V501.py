# -*- coding: utf-8 -*-
"""
This script is intended to plot
1. bifurcation
2. lyapunov exponents

V5: mapping of Version 5
    f(x,y) = mu0 * x * ( 1 - x ) - mu1 * x * y
    g(x,y) =-nu0 * y * ( 1 - y ) + nu1 * x * y

Modification Note: 1. Discard vpython and make use of matplotlib
      2. Slice mu0ListLP, sumLPX, and sumLPY in the `if RoEck == True:` block
      3. Plot the x vs.y
      4. Add try-except statement 
      5. Remove Plot of x vs y
      6. discard gridspec.GridSpec for plots
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import nolds
from pathlib import Path
from numpy import linalg as LA
import warnings
warnings.filterwarnings("error")
import sys

tolerance = 1e-8
 
mu0_min = 1
mu0_max = 4
step  = 0.005

# initX and initY are numbers between 0 and 1
# alpha, beta, and gamma are all greater than 0

simCase = "paraclete"

if simCase == "paraclete":
   initX, initY, alpha, beta, gamma = 0.010, 0.100, 5.000, 0.010, 0.900 # paraclete
elif simCase == "normal":
   initX, initY, alpha, beta, gamma = 0.200, 0.200, 1.000, 0.001, 0.500 # normal    
elif simCase == "extinction":
   initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.500 # extinction
elif simCase == "standard":
   initX, initY, alpha, beta, gamma = 0.100, 0.500, 1.000, 0.100, 0.500 # standard 
elif simCase == "vorticella":
   initX, initY, alpha, beta, gamma = 0.100, 0.100, 1.000, 0.100, 2.000 # vorticella (bell-shaped)
elif simCase == "VS":
   initX, initY, alpha, beta, gamma = 0.100, 0.800, 0.500, 0.010, 1.000 # vorticella strange
else:
    sys.exit("Wrong simCase!")

dirname = "{:.5f}".format(initX)+\
          "_{:.5f}".format(initY)+\
          "_{:.5f}".format(alpha)+\
          "_{:.5f}".format( beta)+\
          "_{:.5f}".format(gamma)

#dirname = simCase

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
#%%
# Calculate Lyapunov for Rosenstein and Eckmann Algorithm
RoEck = True 

mu0ListLM = []  # mu0 list for logistic map
#mu0ListLP = []  # mu0 list for lyapunov
ptsX = []       # points for logistic map X
ptsY = []       # points for logistic map Y

sumLPX = []     # sum of liapunov for X
sumLPY = []     # sum of liapunov for Y

if RoEck == True:
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

def Jacobian(x, y, mu0):
     xx = mu0*(1-2*x) - mu0 * alpha * y
     xy = - mu0 * alpha *  x
     yx = mu0 * gamma *  y
     yy = beta*mu0*(-1+2*y) + mu0*gamma*x
     return xx,xy,yx,yy

def plotFiguresSameX(mu0ListLM, ptsX, ptsY):
    fig, ax = plt.subplots(2, 1, figsize=(16,9), constrained_layout=True)
    plt.title("Logistic Map")
    
    # the first subplot
    ax[0].scatter(mu0ListLM,ptsX,color = 'k' , s = 0.1 , marker = '.')
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax[0].set_ylabel("$x$",size = 16)
    ax[0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax[0].grid(True)
    
    # the second subplot
    # shared axis X
    ax[1].scatter(np.array(mu0ListLM),ptsY,color = 'k', s = 0.1, marker = '.')
    #minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    #minorLocatorY = AutoMinorLocator(5)
    ax[1].set_xlabel("$\\mu_{0}$",size = 16)
    ax[1].set_ylabel("$y$",size = 16)
    ax[1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax[1].grid(True)

    # remove vertical gap between subplots
    #plt.show()
    plt.savefig(str(cwd)+"LogisticMap.png")
    
def outPutLyaExp(mu0ListLP, EqnLx, RLx, ELx1, ELx2, EqnLy, RLy, ELy1, ELy2):
    data = {'mu0ListLP': mu0ListLP,'EqnLx': EqnLx,'RLx':RLx, 'ELx1':ELx1, 'ELx2':ELx2,
                                   'EqnLy': EqnLy,'RLy':RLy, 'ELy1':ELy1, 'ELy2':ELy2}
    df = pd.DataFrame(data)
    df_file = open(str(cwd)+"LyapunovExponents.csv",'w',newline='')
    df.to_csv(df_file, sep=',', encoding='utf-8',index=False)
    df_file.close()
#%%    
nIterations = 2500

mu = np.arange(mu0_min, mu0_max, step)
muList = [mu0_min]
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
                print("Warning Line 169! x_next positive infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                break
            else:
                print("Warning Line 172! x_next negative infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                break
            if y_next > 0:
                print("Warning Line 175! y_next positive infinitle! y_next = {} at mu_ = {}".format(y_next,mu_))
                break
            else:
                print("Warning Line 178! y_next negative infinite! y_next = {} at mu_ = {}".format(y_next,mu_))
                break
        else:
            if x_next < 0 and abs(x_next) < tolerance:
                x_next = 0.0
            if x_next < 0 :
                exception = True
                print("Warning Line 185! x_next negative x_next = {} at mu_ = {}".format(x_next, mu_))
                break
            if y_next < 0 and abs(y_next) < tolerance:
                y_next = 0.0
            if y_next < 0:
                exception = True
                print("Warning Line 191! y_next negative y_next = {} at mu_ = {}".format(y_next, mu_))
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
#%%
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
                print("Warning Line 218! x_next positive infinite! mu_ = {}, x_next = {}".format(mu_,x_next))
                break
            else:
                print("Warning Line 221! x_next negative infinite! mu_ = {}, x_next = {}".format(mu_, x_next))
                break
            if y_next > 0:
                print("Warning Line 224! y_next positive infinitle! mu_ = {}, y_next = {}".format(mu_, y_next))
                break
            else:
                print("Warning Line 227! y_next negative infinite! mu_ = {}, y_next = {}".format(mu_, y_next))
                break
        else:
            if x_next < 0 and abs(x_next) < tolerance:
                x_next = 0.0
            if x_next < 0 :
                exception = True
                print("Warning Line 234! x_next negative x_next = {} at mu_ = {}".format(x_next, mu_))
                break
            if y_next < 0 and abs(y_next) < tolerance:
                y_next = 0.0
            if y_next < 0:
                exception = True
                print("Warning Line 240! y_next negative y_next = {} at mu_ = {}".format(y_next, mu_))
                break
            if i >= avoidTransient:
                mu0ListLM += [mu_]
                ptsX += [x_next]
                ptsY += [y_next]
                x11, x12, x21, x22 = Jacobian(x_next, y_next, mu_)
                w = LA.eigvals( np.array( [ [x11,x12], [x21,x22] ] ) )
            i += 1
    if exception == True:
        break
    else:
        #mu0ListLP += [mu_]
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
    mu0ListLP = mu[::gap] #mu0ListLP[::gap]  # step changes from step1 to step
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
                    print("Warning Line 294! x_next positive infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                    break
                else:
                    print("Warning Line 297! x_next negative infinite! x_next = {} at mu_ = {}".format(x_next,mu_))
                    break
                if y_next > 0:
                    print("Warning Line 300! y_next positive infinitle! y_next = {} at mu_ = {}".format(y_next,mu_))
                    break
                else:
                    print("Warning Line 303! y_next negative infinite! y_next = {} at mu_ = {}".format(y_next,mu_))
                    break
            else:
                if x_next < 0 and abs(x_next) < tolerance:
                    x_next = 0.0
                if x_next < 0 :
                    exception = True
                    print("Warning Line 310! x_next negative x_next = {} at mu_ = {}".format(x_next, mu_))
                    break
                if y_next < 0 and abs(y_next) < tolerance:
                    y_next = 0.0
                if y_next < 0:
                    exception = True
                    print("Warning Line 316! y_next negative y_next = {} at mu_ = {}".format(y_next, mu_))
                    break
                X[i+1] = x_next
                Y[i+1] = y_next
                i += 1 
        if exception == True: 
            print("Warning! mu0List error! Line 322")
            break
        else:               
            try:
                lya_r_X += [ nolds.lyap_r(X, emb_dim=10)/np.log(2) ]
                lya_r_Y += [ nolds.lyap_r(Y, emb_dim=10)/np.log(2) ]
            except np.linalg.LinAlgError:
                if omitRosenstein == False:
                   print("SVD did not converge in least square. Rosenstein omitted.")
                   omitRosenstein = True
                lya_e_X1 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                lya_e_X2 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]
                lya_e_Y1 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                lya_e_Y2 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]
            else:
                if omitRosenstein == False:
                    lya_e_X1 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                    lya_e_X2 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]
                    lya_e_Y1 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                    lya_e_Y2 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]
                    lya_r_X += [ nolds.lyap_r(X, emb_dim=10)/np.log(2) ]
                    lya_r_Y += [ nolds.lyap_r(Y, emb_dim=10)/np.log(2) ]
                else:
                    lya_e_X1 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                    lya_e_X2 += [ nolds.lyap_e(X, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]
                    lya_e_Y1 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[0]/np.log(2) ]
                    lya_e_Y2 += [ nolds.lyap_e(Y, emb_dim=10, matrix_dim=2)[1]/np.log(2) ]

# Write Lyaponov Exponents to csv file
    outPutLyaExp(mu0ListLP, sumLPX, lya_r_X, lya_e_X1, lya_e_X2, 
                            sumLPY, lya_r_Y, lya_e_Y1, lya_e_Y2)
#%%
# plot lyapunov exponents
fig, ax = plt.subplots(2, 1, figsize=(16,9), constrained_layout=True)
plt.title("Lyapunov Exponents")
    
# the first subplot
ax[0].plot(np.array(mu0ListLP),np.array(sumLPX),'b-.')

if RoEck == True:
    if omitRosenstein == False:
       ax[0].plot(np.array(mu0ListLP),np.array(lya_r_X),'k:')
       ax[0].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
       ax[0].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')  
       ax[0].legend(["Equation", "Rosenstein", "Eckmann X", "Eckmann Y"], loc ="lower right")
    else:
       ax[0].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
       ax[0].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')  
       ax[0].legend(["Equation", "Eckmann X", "Eckmann Y"], loc ="lower right")
    
plt.minorticks_on()
minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(5)
#ax0.set_xlabel("$\\mu_{0}$",size = 16)
ax[0].set_ylabel("$\\lambda_{x}$",size = 16)
ax[0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
#plt.xlim(1,4.0)
ax[0].set_ylim([-8,2])
ax[0].grid(True)
    
# the second subplot
# shared axis X
ax[1].plot(np.array(mu0ListLP),np.array(sumLPY),'b-.')

if RoEck == True:
    if omitRosenstein == False:
       ax[1].plot(np.array(mu0ListLP),np.array(lya_r_X),'k:')
       ax[1].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
       ax[1].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')  
       ax[1].legend(["Equation", "Rosenstein", "Eckmann X", "Eckmann Y"], loc ="lower right")
    else:
       ax[1].plot(np.array(mu0ListLP),np.array(lya_e_X1),'r-')
       ax[1].plot(np.array(mu0ListLP),np.array(lya_e_Y1),'g--')  
       ax[1].legend(["Equation", "Eckmann X", "Eckmann Y"], loc ="lower right")

# remove last tick label for the second subplot
minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(5)
ax[1].set_xlabel("$\\mu_{0}$",size = 16)
ax[1].set_ylabel("$\\lambda_{y}$",size = 16)
ax[1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
#plt.xlim(1.0,4.0)
#plt.ylim(-5.2)
ax[1].grid(True)

# remove vertical gap between subplots
#plt.show()
plt.savefig(str(cwd)+"LyapunovExponents.png")