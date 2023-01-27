# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 09:01:13 2022

@author: weishan Lee

This script is intended to plot
1. population vs iteration
2. phase portrait (y vs x)
3. phase diagram
4. eigenvalus w0 and w1 and types of Jacobian at fixed points

Trajectories of x vs. y
# Reference: https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots

V5_1: trajectories of Version 5_1
    f(x,y) = mu0 * x * ( 1 - x ) - mu1 * x * y
    g(x,y) =-nu0 * y * ( 1 - y ) + nu1 * x * y
    
Modification Note: 
    1. Add x vs. n and y vs. n related calculations
    2. plot x_{n+1} - x_{n} vs x_{n}, and y_{n+1} - y_{n} vs y_{n} , i.e, phase spaces of x and y
    3. Plot fixed points
    4. Plot fixed points E1, E2, and E3, and indicate types of the fixed points.
    5. Indicate value of mu0 in fixed point plots E1, E2, and E3. (pending)
    6. Plot |omega2| vs mu0 and |omega1| vs mu0 for fixed points E1, E2, and E3
    7. Rewrite duplicates in calculating traceX and traceY
"""
import numpy as np
import matplotlib as mpl  # for continuous color map
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(5)
from pathlib import Path
from numpy import linalg as LA
import warnings
import pandas as pd
import sys
warnings.filterwarnings("error")

tolerance = 1e-8

mu0_min = 1
mu0_max = 4
step  = 0.005

# initX and initY are numbers between 0 and 1
# alpha, beta, and gamma are all greater than 0
# xlimValue and ylimValue are for plots of x vs n or v vs n 

simCase = "VS"

if simCase == "Paraclete":
   initX, initY, alpha, beta, gamma = 0.010, 0.100, 5.000, 0.010, 0.900 # paraclete
   xlimValue, ylimValue = 1.0, 0.15 # paraclete
elif simCase == "Normal":
   initX, initY, alpha, beta, gamma = 0.200, 0.200, 1.000, 0.001, 0.500 # normal    
   xlimValue, ylimValue = 1.0, 0.30 # normal 
elif simCase == "Extinction":
   initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.500 # extinction
   xlimValue, ylimValue = 1.0, 0.30 # extinction
elif simCase == "Standard":
   initX, initY, alpha, beta, gamma = 0.100, 0.500, 1.000, 0.100, 0.500 # standard 
   xlimValue, ylimValue = 1.0, 0.50 # standard
elif simCase == "Vorticella": # deprecated in this study
   initX, initY, alpha, beta, gamma = 0.100, 0.100, 1.000, 0.100, 2.000 # vorticella (bell-shaped)
   xlimValue, ylimValue = 0.6, 0.70 # vorticella (bell-shaped)
elif simCase == "VS":
   initX, initY, alpha, beta, gamma = 0.100, 0.500, 0.875, 0.018, 1.000 # vorticella strange
   xlimValue, ylimValue = 1.0, 1.00 # vorticella strange
else:
   sys.exit("Wrong simCase!")

#dirname = "{:.5f}".format(initX)+\
#          "_{:.5f}".format(initY)+\
#          "_{:.5f}".format(alpha)+\
#          "_{:.5f}".format( beta)+\
#          "_{:.5f}".format(gamma)

dirname = simCase

# parent folder
Path(dirname).mkdir(parents=True, exist_ok=True)
cwd = "./" + str(dirname) +"/"
logFile = str(cwd)+"log.txt"

# folder of plots of x and y (coordinate space)
dirnameTraj = dirname + "/" + str("trajectory")
Path(dirnameTraj).mkdir(parents=True, exist_ok=True)
folderTraj = "./" + str(dirnameTraj) + "/"

# folder of plots of imaginary vs real part, and absolute values of
# eigenvalues
dirnameEigVals = dirname + "/" + str("EigVals")
Path(dirnameEigVals).mkdir(parents=True, exist_ok=True)
folderEigVals = "./" + str(dirnameEigVals) + "/"

# folder of plots of x vs n or v vs n
dirnamePopVsN = dirname + "/" + str("PopVsN")
Path(dirnamePopVsN).mkdir(parents=True, exist_ok=True)
folderPopVsN = "./" + str(dirnamePopVsN) + "/"

# folder of plots of X phase space
dirnameXPhaseSpace = dirname + "/" + str("XPhaseSpace")
Path(dirnameXPhaseSpace).mkdir(parents=True, exist_ok=True)
folderXPhaseSpace = "./" + str(dirnameXPhaseSpace) + "/"

# folder of plots of Y phase space
dirnameYPhaseSpace = dirname + "/" + str("YPhaseSpace")
Path(dirnameYPhaseSpace).mkdir(parents=True, exist_ok=True)
folderYPhaseSpace = "./" + str(dirnameYPhaseSpace) + "/"

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

def newAttractor(x, y, mu0):
    xx =  mu0 * x * ( 1 - x ) - mu0 * alpha * x * y
    yy = -mu0 * beta * y * ( 1 - y ) + mu0 * gamma * x * y
    return xx, yy

def plotFigure(mu0, ptsX, ptsY):
    plt.figure(figsize=(16,9), constrained_layout=True)
    plt.title("Trajectory")
    ax = plt.gca()
    plt.text(xlimValue * 0.9, ylimValue * 0.9, r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=12)
    #plt.text(max(ptsX)*0.9, max(ptsY)*0.90, r'$\alpha = $'+"{:.5f}".format(alpha))
    #plt.text(max(ptsX)*0.9, max(ptsY)*0.85, r'$\beta = $'+"{:.5f}".format(beta))
    #plt.text(max(ptsX)*0.9, max(ptsY)*0.80, r'$\gamma = $'+"{:.5f}".format(gamma))
    plt.arrow(ptsX[0], ptsY[0], (ptsX[1]-ptsX[0])/2.0, (ptsY[1]-ptsY[0])/2.0, 
              shape='full', color = 'k', lw=0, length_includes_head=True,head_width=0.01)
    plt.plot(ptsX,ptsY,'k.',markersize=10)
    plt.minorticks_on()
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax.set_xlabel("x",size = 16)
    ax.set_ylabel("y",size = 16)
    ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    plt.xlim(-0.01, xlimValue * 1.01)
    plt.ylim(-0.01, ylimValue * 1.01)
    plt.grid(True)
    plt.savefig(str(folderTraj)+str(simCase)+"_"+"Trajectory"+"_{:.5f}".format(mu0)+".jpg")
    plt.close()

# added to save csv of x vs. n and y vs. n
def outPutPoPVsIteration(nIterations,mu0, ptsX, ptsY):
    data = {'nIterations': nIterations,'prey': ptsX,'ptsY':ptsY}
    df = pd.DataFrame(data)
    df_file = open(str(folderPopVsN)+str(simCase)+"_"+"PopVsN"+"_{:.5f}".format(mu0)+".csv",'w',newline='')
    df.to_csv(df_file, sep=',', encoding='utf-8',index=False)
    df_file.close()

# added to plot x vs. n and y vs. n  
def plotPopVsIteration(nIterations,mu0, ptsX, ptsY):
    fig, ax = plt.subplots(2, 1, figsize=(16,11), constrained_layout=True)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12) 

    # the first subplot
    xAxis = 200
    yAxis = ylimValue * 0.8
    ax[0].text(xAxis*0.8,yAxis,r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=24)
    ax[0].scatter(nIterations,ptsX,color = 'k' , s = 8 , marker = 'o')
    
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax[0].set_ylabel("$x_{n}$",size = 32)
    ax[0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax[0].set_xlim([-10,xAxis]) 
    ax[0].set_ylim([-0.05,xlimValue * 1.01])
    ax[0].grid(True)
    
    # the second subplot
    # shared axis X
    ax[1].scatter(nIterations,ptsY,color = 'k', s = 8, marker = 'o')
    ax[1].set_xlabel("$n$",size = 32)
    ax[1].set_ylabel("$y_{n}$",size = 32)  
    ax[1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax[1].set_xlim([-10,xAxis]) 
    ax[1].set_ylim([-0.05,ylimValue * 1.01])
    ax[1].grid(True)   
           
    #plt.show()
    plt.savefig(str(folderPopVsN)+str(simCase)+"_"+"PopVsN"+"_{:.5f}".format(mu0)+".jpg")
    plt.close()
    
# added to plot phase space
def plotFigurePhaseSpace(XorY, mu0, ptsX, ptsY):
    plt.figure(figsize=(16,9))#, constrained_layout=True)
    if XorY == "XPhaseSpace": 
        plt.title("Phase space of x")
    else:
        plt.title("Phase space of y")
    ax = plt.gca()
    plt.text(0.9, 0.9, r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=12)
    plt.plot(ptsX,ptsY,'k.',markersize=10)
    plt.minorticks_on()
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    if XorY == "XPhaseSpace":
        ax.set_xlabel("$x$",size = 16)
        ax.set_ylabel(r'$\Delta x$',size = 16)
    else:
        ax.set_xlabel("$y$",size = 16)
        ax.set_ylabel(r'$\Delta y$',size = 16)
    ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.grid(True)
    if XorY == "XPhaseSpace": 
        plt.savefig(str(folderXPhaseSpace)+str(simCase)+"_"+"XPhaseSpace"+\
                    "_{:.5f}".format(mu0)+".jpg", bbox_inches = 'tight',pad_inches = 0)
    else:
        plt.savefig(str(folderYPhaseSpace)+str(simCase)+"_"+"YPhaseSpace"+\
                    "_{:.5f}".format(mu0)+".jpg", bbox_inches = 'tight',pad_inches = 0)
    plt.close()

def Jacobian(x, y, mu0):
     xx = mu0*(1-2*x) - mu0 * alpha * y
     xy = - mu0 * alpha *  x
     yx = mu0 * gamma *  y
     yy = beta*mu0*(-1+2*y) + mu0*gamma*x
     return xx,xy,yx,yy

def fixedPointsE1(mu0):
    fx = 0
    fy = 1 + 1 / ( beta * mu0 )
    return fx,fy

def fixedPointsE2(mu0):
    fx = 1-1/mu0
    fy = 0
    return fx,fy

def fixedPointsE3(mu0):
    fx = ( alpha * beta * mu0 - beta * mu0 + alpha + beta) / ( mu0 * ( alpha * gamma - beta ) )
    fy = ( -beta * mu0  + gamma * mu0 - gamma - 1 ) / ( mu0 * ( alpha * gamma - beta ) )
    return fx,fy

def fixedPoints(mu0):
    xE1, yE1 = fixedPointsE1(mu0)
    xE2, yE2 = fixedPointsE2(mu0)
    xE3, yE3 = fixedPointsE3(mu0)
    return ([xE1,yE1],[xE2,yE2],[xE3,yE3])
    
def fixedPointType(fPt,mu0):
    
    x11, x12, x21, x22 = Jacobian(fPt[0], fPt[1], mu0)
    w_ = LA.eigvals( np.array( [ [x11,x12], [x21,x22] ] ) )
    
    if abs(w_[0])<1 and abs(w_[1])<1:
        # sink
        area_ = 5
    elif abs(w_[0])>1 and abs(w_[1])>1: 
        # source
        area_ = 25
    elif ( abs(w_[0])<1 and abs(w_[1])>1 ) or ( abs(w_[0])>1 and abs(w_[1])<1 ): 
        # saddle
        area_ = 125
    else: # abs( abs(w[0])-1 ) < tolerance or abs( abs(w[1])-1 ) < tolerance:
        # non-hyperbolic
        area_ = 300
    return w_, area_

def handlesAndLabels(scatter, area_, handles_, labels_, i10_, i50_, i100_, i200_):
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    if area_ == 5 and i10_ == 0:
        handles_ += handles
        labels_ += ['sink']
        i10_ += 1
    if area_ == 25 and i50_ == 0:
        handles_ += handles
        labels_ += ['source']
        i50_ += 1
    if area_ == 125 and i100_ == 0:
        handles_ += handles
        labels_ += ['saddle']
        i100_ += 1
    if area_ == 300 and i200_ ==0:
        handles_ += handles
        labels_ += ['non-hyperbolic']
        i200_ += 1
    return handles_, labels_, i10_, i50_, i100_, i200_

#%% 
# Find suitable muList
nIterations = 2500

mu = np.arange(mu0_min, mu0_max, step)
muList = [] #[mu0_min]

traceX = np.zeros([len(mu), nIterations])
traceY = np.zeros([len(mu), nIterations])

for ii, mu_ in enumerate(mu):
    traceX[ii,0] = initX
    traceY[ii,0] = initY

    # version 2
    i = 0 
    exception = False
    while (i < nIterations - 1):
        try:
            x_next, y_next = newAttractor(traceX[ii,i], traceY[ii,i], mu_)
        except RuntimeWarning:
            exception = True
            if x_next > 0:
                print("Warning Line 329! x_next positive infinite! x_next = {} at mu = {}".format(x_next,mu_))
                break
            else:
                print("Warning Line 332! x_next negative infinite! x_next = {} at mu = {}".format(x_next,mu_))
                break
            if y_next > 0:
                print("Warning Line 335! y_next positive infinite! y_next = {} at mu = {}".format(y_next,mu_))
                break
            else:
                print("Warning Line 338! y_next negative infinite! y_next = {} at mu = {}".format(y_next,mu_))
                break
        else:
            if x_next < 0 and abs(x_next) < tolerance:
                x_next = 0.0
            if x_next < 0 or x_next >1:
                exception = True
                print("Warning Line 345! x_next out of bound x_next = {} at mu = {}".format(x_next,mu_))
                break
            if y_next < 0 and abs(y_next) < tolerance:
                y_next = 0.0
            if y_next < 0 or y_next >1:
                exception = True
                print("Warning Line 351! y_next out of bound y_next = {} at mu = {}".format(y_next,mu_))
                break
            traceX[ii,i+1] = x_next
            traceY[ii,i+1] = y_next
            i += 1
    if exception == False:
        muList += [mu_]
    else:
        writeLog(r'mu0_max = '+ "{:.5f}\n".format(muList[-1]))
        break

mu = np.array(muList) 
# end of find suitable muList V508
#%%
color_name = 'viridis'
cmap = plt.cm.get_cmap(color_name)
norm = mpl.colors.Normalize(vmin=mu.min(), vmax=mu.max())   # for continuous color map
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)          # for continuous color map
cmap.set_array([])                                          # for continuous color map
#%%
# plot of |omega1| vs mu0 and |omega0| vs mu0 for fixed points E1, E2, and E3
fig, ax = plt.subplots(2, 3, figsize=(16,9), constrained_layout=True)

absOmega0E1 = []
absOmega1E1 = []

absOmega0E2 = []
absOmega1E2 = []

absOmega0E3 = []
absOmega1E3 = []

for mu_ in mu:
    
    fPtE1, fPtE2, fPtE3 = fixedPoints(mu_)
    # type for fixpoint E1
    w, area = fixedPointType(fPtE1, mu_)
    absOmega0E1 += [abs(w[0])]
    absOmega1E1 += [abs(w[1])]
    
    # type for fixpoint E2
    w, area = fixedPointType(fPtE2, mu_)
    absOmega0E2 += [abs(w[0])]
    absOmega1E2 += [abs(w[1])]       

    # type for fixpoint E3
    w, area = fixedPointType(fPtE3, mu_)
    absOmega0E3 += [abs(w[0])]
    absOmega1E3 += [abs(w[1])] 
    
# |omega_{1}|
ax[0,0].plot(np.array(mu),np.array(absOmega1E1),'k-')
#ax[0,0].set_xlabel(r'Re$(\omega_{0})$',size = 10)
ax[0,0].set_ylabel(r'|$\omega_{1}$|',size = 12)
ax[0,0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0,0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[0,0].grid(True)

ax[0,1].plot(np.array(mu),np.array(absOmega1E2),'k-')
#ax[0,1].set_xlabel(r'Re$(\omega_{1})$',size = 10)
#ax[0,1].set_ylabel(r'Im$(\omega_{1})$',size = 12)
ax[0,1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0,1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[0,1].grid(True)

ax[0,2].plot(np.array(mu),np.array(absOmega1E3),'k-')
#ax[0,2].set_xlabel(r'$|\omega_{0}|$',size = 10)
#ax[0,2].set_ylabel(r'$|\omega_{1}|$',size = 12)
ax[0,2].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0,2].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[0,2].grid(True)

# |omega_{0}|
ax[1,0].plot(np.array(mu),np.array(absOmega0E1),'k-')
ax[1,0].set_xlabel(r'$\mu_{0}.$'+str(" Column for ")+r'$E_{1}^{\prime}$',size = 10)
ax[1,0].set_ylabel(r'|$\omega_{0}$|',size = 12)
ax[1,0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1,0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[1,0].grid(True)

ax[1,1].plot(np.array(mu),np.array(absOmega0E2),'k-')
ax[1,1].set_xlabel(r'$\mu_{0}.$'+str(" Column for ")+r'$E_{2}^{\prime}$',size = 10)
#ax[1,1].set_ylabel(r'Im$(\omega_{1})$',size = 12)
ax[1,1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1,1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[1,1].grid(True)

ax[1,2].plot(np.array(mu),np.array(absOmega0E3),'k-')
ax[1,2].set_xlabel(r'$\mu_{0}.$'+str(" Column for ")+r'$E_{3}^{\prime}$',size = 10)
#ax[1,2].set_ylabel(r'$|\omega_{1}|$',size = 12)
ax[1,2].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1,2].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[1,2].grid(True)

fig.savefig(str(folderEigVals)+str(simCase)+"_"+"absEval_vs_mu0.jpg")
# end of plot of |omega1| vs mu0 and |omega0| vs mu0 for fixed points E1, E2, and E3
#%%
# plot of Imaginary part vs. Real part of eigenvalues w0 and w1, and 
# absolute Values of eigenvalues |w1| vs. |w0| for fixed points E1, E2, and E3
handles1 = []
labels1 = []

i10 = 0
i50 = 0
i100 = 0
i200 = 0

fig, ax = plt.subplots(4, 3, figsize=(16,9), constrained_layout=True)

for mu_ in mu:
    alpha_ = 0.2
    
    color = cmap.to_rgba(mu_)    
    fPtE1, fPtE2, fPtE3 = fixedPoints(mu_) 
    
    # type for fixpoint E1
    w, area = fixedPointType(fPtE1, mu_)

    scatter = ax[0,0].scatter(np.real(w[0]),np.imag(w[0]),color=color, s = area, alpha=alpha_)
    handles1, labels1, i10, i50, i100, i200 =\
    handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
      
    ax[0,1].scatter(np.real(w[1]),np.imag(w[1]),color=color, s = area, alpha=alpha_)
    ax[0,2].scatter(abs(w[0]),abs(w[1]),color=color, s = area, alpha=alpha_)
      
    # type for fixpoint E2
    w, area = fixedPointType(fPtE2, mu_)
        
    scatter = ax[1,0].scatter(np.real(w[0]),np.imag(w[0]),color=color, s = area, alpha=alpha_)
    handles1, labels1, i10, i50, i100, i200 =\
    handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
    
    ax[1,1].scatter(np.real(w[1]),np.imag(w[1]),color=color, s = area, alpha=alpha_)
    ax[1,2].scatter(abs(w[0]),abs(w[1]),color=color, s = area, alpha=alpha_)
  
    # type for fixpoint E3
    w, area = fixedPointType(fPtE3, mu_)

    scatter = ax[2,0].scatter(np.real(w[0]),np.imag(w[0]),color=color, s = area, alpha=alpha_)
    handles1, labels1, i10, i50, i100, i200 =\
    handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
    
    ax[2,1].scatter(np.real(w[1]),np.imag(w[1]),color=color, s = area, alpha=alpha_)
    ax[2,2].scatter(abs(w[0]),abs(w[1]),color=color, s = area, alpha=alpha_)
     
# E1
#ax[0,0].set_xlabel(r'Re$(\omega_{0})$',size = 10)
ax[0,0].set_ylabel(r'Im$(\omega_{0})$',size = 12)
ax[0,0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0,0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[0,0].grid(True)

#ax[0,1].set_xlabel(r'Re$(\omega_{1})$',size = 10)
ax[0,1].set_ylabel(r'Im$(\omega_{1})$',size = 12)
ax[0,1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0,1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[0,1].grid(True)

#ax[0,2].set_xlabel(r'$|\omega_{0}|$',size = 10)
ax[0,2].set_ylabel(r'$|\omega_{1}|$',size = 12)
ax[0,2].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[0,2].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[0,2].grid(True)

# E2
#ax[1,0].set_xlabel(r'Re$(\omega_{0})$',size = 10)
ax[1,0].set_ylabel(r'Im$(\omega_{0})$',size = 12)
ax[1,0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1,0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[1,0].grid(True)

#ax[1,1].set_xlabel(r'Re$(\omega_{1})$',size = 10)
ax[1,1].set_ylabel(r'Im$(\omega_{1})$',size = 12)
ax[1,1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1,1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[1,1].grid(True)

#ax[1,2].set_xlabel(r'$|\omega_{0}|$',size = 10)
ax[1,2].set_ylabel(r'$|\omega_{1}|$',size = 12)
ax[1,2].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[1,2].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[1,2].grid(True)

# E3 
ax[2,0].set_xlabel(r'Re$(\omega_{0})$',size = 12)
ax[2,0].set_ylabel(r'Im$(\omega_{0})$',size = 12)
ax[2,0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[2,0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[2,0].grid(True)

ax[2,1].set_xlabel(r'Re$(\omega_{1})$',size = 12)
ax[2,1].set_ylabel(r'Im$(\omega_{1})$',size = 12)
ax[2,1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[2,1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[2,1].grid(True)

ax[2,2].set_xlabel(r'$|\omega_{0}|$',size = 12)
ax[2,2].set_ylabel(r'$|\omega_{1}|$',size = 12)
ax[2,2].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax[2,2].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax[2,2].grid(True)

cbar = fig.colorbar(cmap,ticks = np.arange(1,int(np.ceil(norm.vmax))+1), 
                    ax=ax[:3, :], location='right', aspect = 40)
cbar.ax.get_xaxis().labelpad = 15   # cbar has to follow ax, NOT axes or anything else
cbar.ax.set_xlabel(r'$\mu_{0}$')    # cbar has to follow ax, NOT axes or anything else

# legend (leg)
ax[3,0].legend(handles1, labels1, title="Type")
ax[3,0].axis('off')
# remove frames in the empty ax
ax[3,1].axis('off')
ax[3,2].axis('off')

fig.savefig(str(folderEigVals)+str(simCase)+"_"+"ImagVsRe_AbsVals.jpg")
# end of plot of Imaginary part vs. Real part of eigenvalues w0 and w1, and 
# absolute Values of eigenvalues |w1| vs. |w0| for fixed points E1, E2, and E3
#%%
# plot coordinate space (phase portrait)
# type of fixed points are also included after V506
fig = plt.figure(figsize=(16,9), constrained_layout=True)
ax = fig.add_subplot(111)

nIterations = 600 # Reduce nIterations if the black area is large. 
handles1 = []
labels1 = []

i10 = 0
i50 = 0
i100 = 0
i200 = 0

alpha_ = 0.2    
for ii in range(len(mu)):
    mu_ = mu[ii]
    color = cmap.to_rgba(mu_) 
    plotFigure(mu_, traceX[ii,0:nIterations], traceY[ii,0:nIterations]) # for a single phase portrait black figure for an arrow                 
                
    ax.plot(traceX[ii,0:nIterations],traceY[ii,0:nIterations],'.',color = color,alpha=alpha_,markersize=3)  # set alpha=0.1 for chaos
    plotPopVsIteration(np.arange(nIterations),mu_, traceX[ii,0:nIterations], traceY[ii,0:nIterations])   # added to plot x vs. n and y vs. n
    outPutPoPVsIteration(np.arange(nIterations),mu_, traceX[ii,0:nIterations], traceY[ii,0:nIterations]) # added to save csv of x vs. n and y vs. n

    fPtE1, fPtE2, fPtE3 = fixedPoints(mu_)

    # type for fixpoint E1
    w, area = fixedPointType(fPtE1, mu_)
    if fPtE1[0]>=0.0 and fPtE1[0]<=1.0 and fPtE1[1]>=0.0 and fPtE1[1]<=1.0:
       scatter = ax.scatter(fPtE1[0],fPtE1[1],color = color, s = area, alpha=alpha_)
       handles1, labels1, i10, i50, i100, i200 =\
       handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       ax.plot(fPtE1[0],fPtE1[1],'k.',markersize=1)
        
    # type for fixpoint E2
    w, area = fixedPointType(fPtE2, mu_)
    if fPtE2[0]>=0.0 and fPtE2[0]<=1.0 and fPtE2[1]>=0.0 and fPtE2[1]<=1.0:
       scatter = ax.scatter(fPtE2[0],fPtE2[1],color = color, s = area, alpha=alpha_)
       handles1, labels1, i10, i50, i100, i200 =\
       handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       ax.plot(fPtE2[0],fPtE2[1],'k.',markersize=1)
        
    # type for fixpoint E3
    w, area = fixedPointType(fPtE3, mu_)
    if fPtE3[0]>=0.0 and fPtE3[0]<=1.0 and fPtE3[1]>=0.0 and fPtE3[1]<=1.0:
       scatter = ax.scatter(fPtE3[0],fPtE3[1],color = color, s = area, alpha=alpha_)
       handles1, labels1, i10, i50, i100, i200 =\
       handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       ax.plot(fPtE3[0],fPtE3[1],'k.',markersize=1)         
            
cbar = fig.colorbar(cmap,ticks = np.arange(1,int(np.ceil(norm.vmax))+1),aspect=40)
cbar.ax.get_xaxis().labelpad = 15
cbar.ax.set_xlabel(r'$\mu_{0}$')
plt.minorticks_on()  # no fig.minorticks_on()
ax.set_xlabel("x",size = 16)
ax.set_ylabel("y",size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax.legend(handles1, labels1, title="Type")
ax.grid(True)
#plt.xlim(-0.01,xlimValue * 1.02)
#plt.ylim(-0.01,ylimValue * 1.02)
#plt.show()
fig.savefig(str(folderTraj)+str(simCase)+"_"+"Trajectory_all.jpg")
# end of plot coordinate space (phase portrait) from V506
#%%
# plot phase spaces
## plot phase space of x
fig = plt.figure(figsize=(16,9), constrained_layout=True)
ax = fig.add_subplot(111)
alpha_ = 0.2

for ii in range(len(mu)):
    mu_ = mu[ii]
    diffTraceX = [ (traceX[ii, (i+1)] - traceX[ii, i]) for i in range( nIterations -1 ) ]
    plotFigurePhaseSpace("XPhaseSpace",mu_, traceX[ii, 0:(nIterations-1)], diffTraceX)                   
    ax.plot(traceX[ii, 0:(nIterations-1)],diffTraceX,'.',color=cmap.to_rgba(mu_),alpha=alpha_,markersize=3)  # for continuous color map
            
cbar = fig.colorbar(cmap,ticks = np.arange(1,int(np.ceil(norm.vmax))+1), aspect = 40)
cbar.ax.get_xaxis().labelpad = 15
cbar.ax.set_xlabel(r'$\mu_{0}$')
plt.minorticks_on()
ax.set_xlabel("$x$",size = 16)
ax.set_ylabel(r'$\Delta x$',size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax.grid(True)
#plt.xlim(-0.01,1.02)
#plt.ylim(-0.01,1.02)
#plt.show()
fig.savefig(str(folderXPhaseSpace)+str(simCase)+"_"+"phaseSpaceX_all.jpg")
#%%
## plot phase space of y
fig = plt.figure(figsize=(16,9), constrained_layout=True)
ax = fig.add_subplot(111)

for ii in range(len(mu)):
    mu_ = mu[ii]
    diffTraceY = [ (traceY[ii, (i+1)] - traceY[ii, i]) for i in range( nIterations -1 ) ]
    plotFigurePhaseSpace("YPhaseSpace",mu_, traceY[ii, 0:(nIterations-1)], diffTraceY)                   
    ax.plot(traceY[ii, 0:(nIterations-1)],diffTraceY,'.',color=cmap.to_rgba(mu_),alpha=alpha_,markersize=3)  # for continuous color map
                             
cbar = fig.colorbar(cmap,ticks = np.arange(1,int(np.ceil(norm.vmax))+1), aspect = 40)
cbar.ax.get_xaxis().labelpad = 15
cbar.ax.set_xlabel(r'$\mu_{0}$')
plt.minorticks_on()
ax.set_xlabel("$y$",size = 16)
ax.set_ylabel(r'$\Delta y$',size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax.grid(True)
#plt.xlim(-0.01,1.02)
#plt.ylim(-0.01,1.02)
#plt.show()
fig.savefig(str(folderYPhaseSpace)+str(simCase)+"_"+"phaseSpaceY_all.jpg")
# end of plot phase space
#%%                 
import moviepy.video.io.ImageSequenceClip
fps=10
image_files = [os.path.join(folderTraj,img)
               for img in sorted(os.listdir(folderTraj))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
clip.write_videofile(str(folderTraj)+str(simCase)+"_"+'trajectory.mp4')
print("----------------------------------------------\n")
image_files = [os.path.join(folderPopVsN,img)
               for img in sorted(os.listdir(folderPopVsN))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(str(folderPopVsN)+str(simCase)+"_"+'popVsnIterations.mp4')
print("----------------------------------------------\n")
image_files = [os.path.join(folderXPhaseSpace,img)
               for img in sorted(os.listdir(folderXPhaseSpace))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
clip.write_videofile(str(folderXPhaseSpace)+str(simCase)+"_"+'phaseSpaceX.mp4')
print("----------------------------------------------\n")
image_files = [os.path.join(folderYPhaseSpace,img)
               for img in sorted(os.listdir(folderYPhaseSpace))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
clip.write_videofile(str(folderYPhaseSpace)+str(simCase)+"_"+'phaseSpaceY.mp4')