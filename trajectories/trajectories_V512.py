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
    8. Plot |omega1|, |omega0|, Im(omega1), Re(Omega1), Im(Omega0), Re(Omega0) 
       vs mu0 for fixed points E1, E2, and E3
    9. Restore gridspec package for def plotPopVsIteration that is copied from 
       def plotPopVsSameN from V506 because the old def plotPopVsIteration caused
       trouble for "VS" simCase
   10. Modify def newAttractor, def jacobian, and def func from pplane.py 
   11. Change simCase directions a little bit.   
   12. Apply ax.stream adopted from pplane.py to plot phase portraits 
   13. Modify plt.arrorw  in def plotFigure function. 
   14. Find xlim, ylim for phase portrait, and phase spaces automatically. 
   15. Remove plt.arrow in def plotFigure
   16. Remove outPutPoPVsIteration calculations
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
from scipy.interpolate import griddata

warnings.filterwarnings("error")

# Plot with streamplot? 
# Note: The gradient estimation did not converge. Problems on this still not solved.
#       In streamplot.py, replace line_color = [] with line_color = [[]]
plotStreamplot = False

tolerance = 1e-8

mu0_min = 1
mu0_max = 4
step  = 0.005

# initX and initY are numbers between 0 and 1
# alpha, beta, and gamma are all greater than 0
###############################################################################
simCase = "Vorticella"

if simCase == "Trial":
   initX, initY, alpha, beta, gamma = 0.200, 0.200, 1.000, 0.001, 0.500
   #step  = 0.005  
   dirname = "{:.5f}".format(initX)+\
              "_{:.5f}".format(initY)+\
              "_{:.5f}".format(alpha)+\
              "_{:.5f}".format( beta)+\
              "_{:.5f}".format(gamma)
#elif simCase == "Paraclete":
#    initX, initY, alpha, beta, gamma = 0.010, 0.100, 5.000, 0.010, 0.900 # paraclete
elif simCase == "Normal": 
    initX, initY, alpha, beta, gamma = 0.200, 0.200, 1.000, 0.001, 0.500 # normal 
elif simCase == "Standard":   
    initX, initY, alpha, beta, gamma = 0.100, 0.500, 1.000, 0.100, 0.500 # standard 
elif simCase == "Extinction": 
    initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.500 # extinction
elif simCase == "Vorticella": 
    initX, initY, alpha, beta, gamma = 0.010, 0.100, 5.000, 0.001, 0.900 # vorticella (bell-shaped)
#elif simCase == "VS":
#    initX, initY, alpha, beta, gamma = 0.100, 0.500, 0.875, 0.018, 1.000 # vorticella strange
else:
    sys.exit("Wrong simCase!")
dirname = simCase
###############################################################################

# parent folder
Path(dirname).mkdir(parents=True, exist_ok=True)
cwd = "./" + str(dirname) +"/"
logFile = str(cwd)+"log.txt"

# folder of plots of x and y (coordinate space) phase portrait
dirnameTraj = dirname + "/" + str("trajectory")
Path(dirnameTraj).mkdir(parents=True, exist_ok=True)
folderTraj = "./" + str(dirnameTraj) + "/"

# folder of plots of x and y phase portrait with ax.streamplot # after version V511
dirnamePhasePortrait = dirname + "/" + str("PhasePortrait")
Path(dirnamePhasePortrait).mkdir(parents=True, exist_ok=True)
folderPhasePortrait = "./" + str(dirnamePhasePortrait) + "/"

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

def func(variables,mu0):
    x, y = variables
    xx, yy = newAttractor(x, y, mu0)
    return (xx, yy)

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

def plotFigure(mu0, ptsX, ptsY,xmin,xmax,ymin,ymax):
    plt.figure(figsize=(16,9), constrained_layout=True)
    plt.title("Trajectory")
    ax = plt.gca()
    plt.text(xmax * 0.9, ymax * 0.9, r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=12)
    #plt.arrow(ptsX[0], ptsY[0], (ptsX[1]-ptsX[0])/2.0, (ptsY[1]-ptsY[0])/2.0, 
    #          shape='full', color = 'k', lw=0, length_includes_head=True,head_width=0.01)
    #for i in range( len(ptsX) - 1):
    #    plt.arrow(ptsX[i], ptsY[i], (ptsX[i+1]-ptsX[i])/2.0, (ptsY[i+1]-ptsY[i])/2.0, 
    #              color = 'k', lw=1)  
    plt.plot(ptsX,ptsY,'k.',markersize=10)
    plt.minorticks_on()
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax.set_xlabel("x",size = 16)
    ax.set_ylabel("y",size = 16)
    ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.savefig(str(folderTraj)+str(simCase)+"_"+"Trajectory"+"_{:.5f}".format(mu0)+".jpg")
    plt.close()
   
def plotFigureAxStream(mu0, ptsX, ptsY,xmin,xmax,ymin,ymax):
    plt.figure(figsize=(16,9), constrained_layout=True)
    plt.title("PhasePortrait")
    ax = plt.gca()
    plt.text(xmax * 0.9, ymax * 0.9, r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=12)
    
    xx = ptsX[0:(nIterations-1)]
    yy = ptsY[0:(nIterations-1)]
    u = [ (ptsX[i+1] - ptsX[i]) for i in range( nIterations -1 ) ]
    v = [ (ptsY[i+1] - ptsY[i]) for i in range( nIterations -1 ) ]
    
    u= np.array(u,dtype="float64")
    v= np.array(v,dtype="float64")
    speed = np.sqrt(u**2 + v**2)

    # Note: ax.streamplot can only handle x and y 
    #       to be "evenly spaced strictly increasing arrays" to make a grid.
    # Reference: https://stackoverflow.com/questions/25297375/matplotlib-streamplot-for-unevenly-curvilinear-grid
    
    # regularly spaced grid spanning the domain of x and y 
    x = np.linspace(xx.min(), xx.max(), xx.size)
    y = np.linspace(yy.min(), yy.max(), yy.size)

    xi, yi = np.meshgrid(x,y)

    # Then, interpolate your data onto this grid:

    px = xx.flatten()
    py = yy.flatten()
    pu = u.flatten()
    pv = v.flatten()
    pspeed = speed.flatten()
    
    points = np.r_[ px[None,:], py[None,:] ].T # The purpose of a[None,:] is to add an axis to array a.
    
    gu = griddata(points, pu, (xi,yi),method='cubic')
    gv = griddata(points, pv, (xi,yi),method='cubic')
    gspeed = griddata(points, pspeed, (xi,yi),method='cubic')

    lw = gspeed/np.nanmax(gspeed)
    # Now, you can use x, y, gu, gv and gspeed in streamplot:

    ax.streamplot(x,y,gu,gv, density=1,linewidth=lw, color=gspeed)#, cmap.to_rgba(mu0))
    
    ax.plot(ptsX,ptsY,'k.',markersize=10)
    
    plt.minorticks_on()
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax.set_xlabel("x",size = 16)
    ax.set_ylabel("y",size = 16)
    ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.savefig(str(folderPhasePortrait)+str(simCase)+"_"+"PhasePortrait"+"_{:.5f}".format(mu0)+".jpg")
    plt.close()

# added to save csv of x vs. n and y vs. n
def outPutPoPVsIteration(nIterations,mu0, ptsX, ptsY):
    data = {'nIterations': nIterations,'prey': ptsX,'ptsY':ptsY}
    df = pd.DataFrame(data)
    df_file = open(str(folderPopVsN)+str(simCase)+"PopVsN"+"_{:.5f}".format(mu0)+".csv",'w',newline='')
    df.to_csv(df_file, sep=',', encoding='utf-8',index=False)
    df_file.close()

# added to plot x vs. n and y vs. n  
def plotPopVsIteration(nIterations,mu0, ptsX, ptsY,xmin,xmax,ymin,ymax):
    fig, ax = plt.subplots(2, 1, figsize=(16,11), constrained_layout=True)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12) 

    # the first subplot
    xAxis = 200
    yAxis = ymax * 0.8
    ax[0].text(xAxis*0.8,yAxis,r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=24)
    ax[0].scatter(nIterations,ptsX,color = 'k' , s = 8 , marker = 'o')
    
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax[0].set_ylabel("$x_{n}$",size = 32)
    ax[0].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[0].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax[0].set_xlim([-10,xAxis]) 
    ax[0].set_ylim([xmin,xmax])
    ax[0].grid(True)
    
    # the second subplot
    # shared axis X
    ax[1].scatter(nIterations,ptsY,color = 'k', s = 8, marker = 'o')
    ax[1].set_xlabel("$n$",size = 32)
    ax[1].set_ylabel("$y_{n}$",size = 32)  
    ax[1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax[1].set_xlim([-10,xAxis]) 
    ax[1].set_ylim([ymin,ymax])
    ax[1].grid(True)   
           
    #plt.show()
    plt.savefig(str(folderPopVsN)+str(simCase)+"_"+"PopVsN"+"_{:.5f}".format(mu0)+".jpg")
    plt.close()
# added to plot phase space
def plotFigurePhaseSpace(XorY, mu0, ptsX, ptsY,xmin,xmax,ymin,ymax):
    plt.figure(figsize=(16,9))
    if XorY == "XPhaseSpace": 
        plt.title("Phase space of x")
    else:
        plt.title("Phase space of y")
    ax = plt.gca()
    plt.text(xmax * 0.9, ymax * 0.9, r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=12)
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
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.grid(True)
    if XorY == "XPhaseSpace": 
        plt.savefig(str(folderXPhaseSpace)+str(simCase)+"_"+"XPhaseSpace"+\
                    "_{:.5f}".format(mu0)+".jpg", bbox_inches = 'tight',pad_inches = 0)      
    else:
        plt.savefig(str(folderYPhaseSpace)+str(simCase)+"_"+"YPhaseSpace"+\
                    "_{:.5f}".format(mu0)+".jpg", bbox_inches = 'tight',pad_inches = 0)
    plt.close()  

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
    w_ = LA.eigvals( jacobian(func,( fPt[0], fPt[1] ), mu0 ) )
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
            if x_next < 0 or x_next >1:
                exception = True
                print("Warning Line 345! x_next out of bound x_next = {} at mu = {}".format(x_next,mu_))
                break
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

if len(traceX[:,0])!=len(mu) or len(traceY[:,0])!=len(mu):
   print("Warning Line 444! len(traceX) = {} and len(traceY) = {}, but len(mu) = {}".format(len(traceX[:,0]),len(traceY[:,0]),len(mu)))
   sys.exit()
# end of find suitable muList V508
#%%
# Find xlim, ylim for phase portrait, and phase spaces automatically.

diffTraceX = np.zeros([len(mu), nIterations-1])
diffTraceY = np.zeros([len(mu), nIterations-1])

for ii in range(len(mu)):
    for iii in range(nIterations-1):
        diffTraceX[ii,iii] = traceX[ii, iii+1] - traceX[ii, iii]
        diffTraceY[ii,iii] = traceY[ii, iii+1] - traceY[ii, iii]

xMin = np.min(traceX)
xMax = np.max(traceX)
dxMin = np.min(diffTraceX)
dxMax = np.max(diffTraceX)

yMin = np.min(traceY)
yMax = np.max(traceY)
dyMin = np.min(diffTraceY)
dyMax = np.max(diffTraceY)
#%%
color_name = 'viridis'
cmap = plt.cm.get_cmap(color_name)#mpl.colormaps.get_cmap(color_name)
norm = mpl.colors.Normalize(vmin=mu.min(), vmax=mu.max())   # for continuous color map
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)          # for continuous color map
cmap.set_array([])                                          # for continuous color map
#%% 
# plot Re(Omega0), Im(Omega0), |omega0|, Re(Omega1), Im(omega1), |omega1|   
# vs mu0 for fixed points E1, E2, and E3

w0E1 = []
w1E1 = []

w0E2 = []
w1E2 = []

w0E3 = []
w1E3 = []

for mu_ in mu:
    
    fPtE1, fPtE2, fPtE3 = fixedPoints(mu_)
    # type for fixpoint E1
    w, area = fixedPointType(fPtE1, mu_)
    w0E1 += [w[0]]
    w1E1 += [w[1]]
    
    # type for fixpoint E2
    w, area = fixedPointType(fPtE2, mu_)
    w0E2 += [w[0]]
    w1E2 += [w[1]]     

    # type for fixpoint E3
    w, area = fixedPointType(fPtE3, mu_)
    w0E3 += [w[0]]
    w1E3 += [w[1]]

ReW0E1, ImW0E1 = np.real(w0E1), np.imag(w0E1)# , map(abs,w0E1)
absW0E1 = np.sqrt( ReW0E1**2 + ImW0E1**2 )

ReW1E1, ImW1E1 = np.real(w1E1), np.imag(w1E1)#, map(abs,w1E1)
absW1E1 = np.sqrt( ReW1E1**2 + ImW1E1**2 )

ReW0E2, ImW0E2 = np.real(w0E2), np.imag(w0E2)#, map(abs,w0E2)
absW0E2 = np.sqrt( ReW0E2**2 + ImW0E2**2 )

ReW1E2, ImW1E2 = np.real(w1E2), np.imag(w1E2)#, map(abs,w1E2)
absW1E2 = np.sqrt( ReW1E2**2 + ImW1E2**2 )

ReW0E3, ImW0E3 = np.real(w0E3), np.imag(w0E3)#, map(abs,w0E3)
absW0E3 = np.sqrt( ReW0E3**2 + ImW0E3**2 )

ReW1E3, ImW1E3 = np.real(w1E3), np.imag(w1E3)#, map(abs,w1E3)
absW1E3 = np.sqrt( ReW1E3**2 + ImW1E3**2 )

data = {'mu0': mu,'ReW0E1':  ReW0E1, 'ReW0E2':  ReW0E2, 'ReW0E3':  ReW0E3,
                  'ImW0E1':  ImW0E1, 'ImW0E2':  ImW0E2, 'ImW0E3':  ImW0E3,
                 'absW0E1': absW0E1,'absW0E2': absW0E2,'absW0E3': absW0E3,
                  'ReW1E1':  ReW1E1, 'ReW1E2':  ReW1E2, 'ReW1E3':  ReW1E3,
                  'ImW1E1':  ImW1E1, 'ImW1E2':  ImW1E2, 'ImW1E3':  ImW1E3,
                 'absW1E1': absW1E1,'absW1E2': absW1E2,'absW1E3': absW1E3}
df = pd.DataFrame(data)
df_file = open(str(folderEigVals)+str(simCase)+"ReImAbs.csv",'w',newline='')
df.to_csv(df_file, sep=',', encoding='utf-8',index=False)
df_file.close()

# plots
fig = plt.figure(figsize=(8.27, 11.69))#, constrained_layout=True)

#fig, ax = plt.subplots(6, 3, sharex='col',figsize=(8.27, 11.69))
for i in range(0,18):
    
    ax = plt.subplot(6, 3, i+1)#, sharex=True)
    
    if i!= 15 and i!=16 and i!=17:
       plt.tick_params(
       axis='x',           # changes apply to the x-axis
       which='major',      # both major and minor ticks are affected
       bottom=True,        # ticks along the bottom edge are off
       top=False,          # ticks along the top edge are off
       labelbottom=False)  # labels along the bottom edge are off
    
    ax.plot( df.iloc[:,0], df.iloc[:,i+1], 'k-')
    ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax.grid(True)

    if i == 0:
       ax.set_ylabel(r'Re($\omega_{0})$',size = 12)
    if i == 3:
       ax.set_ylabel(r'Im($\omega_{0})$',size = 12)
    if i == 6:
       ax.set_ylabel(r'|$\omega_{0}$|',size = 12)
    if i == 9:
       ax.set_ylabel(r'Re($\omega_{1})$',size = 12)   
    if i ==12:   
       ax.set_ylabel(r'Im($\omega_{1})$',size = 12)
    if i ==15:
       ax.set_xlabel(r'$\mu_{0}$'+str(" at ")+r'$E^{\prime}_{1}$',size = 12)
       ax.set_ylabel(r'|$\omega_{1}$|',size = 12)
    if i ==16:
       ax.set_xlabel(r'$\mu_{0}$'+str(" at ")+r'$E^{\prime}_{2}$',size = 12)
    if i ==17:
       ax.set_xlabel(r'$\mu_{0}$'+str(" at ")+r'$E^{\prime}_{3}$',size = 12)
       
plt.tight_layout()       
plt.savefig(str(folderEigVals)+str(simCase)+"_"+"ReImAbs.jpg", bbox_inches = 'tight',pad_inches = 0)
# end of plot Re(Omega0), Im(Omega0), |omega0|, Re(Omega1), Im(omega1), |omega1|
# vs mu0 for fixed points E1, E2, and E3
#%%
# plot of Imaginary part vs. Real part of eigenvalues w0 and w1, and 
# absolute Values of eigenvalues |w1| vs. |w0| for fixed points E1, E2, and E3
handles1 = []
labels1 = []

i10 = 0
i50 = 0
i100 = 0
i200 = 0

#fig, ax = plt.subplots(4, 3, figsize=(11.69,8.27), constrained_layout=True, sharex=True) # for landscape 
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

fig.savefig(str(folderEigVals)+str(simCase)+"_"+"ReImAbs_color.jpg")
# end of plot of Imaginary part vs. Real part of eigenvalues w0 and w1, and 
# absolute Values of eigenvalues |w1| vs. |w0| for fixed points E1, E2, and E3
#%%
# plot phase spaces
## plot phase space of x
fig = plt.figure(figsize=(16,9), constrained_layout=True)
ax = fig.add_subplot(111)
alpha_ = 0.2

for ii in range(len(mu)):
    mu_ = mu[ii]
    plotFigurePhaseSpace("XPhaseSpace",mu_, traceX[ii, :-1], diffTraceX[ii,:],
                         xMin, xMax, dxMin, dxMax)
    ax.plot(traceX[ii, :-1], diffTraceX[ii,:],'.',color=cmap.to_rgba(mu_),alpha=alpha_,markersize=3)# for continuous color map
            
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

## plot phase space of y
fig = plt.figure(figsize=(16,9), constrained_layout=True)
ax = fig.add_subplot(111)

for ii in range(len(mu)):
    mu_ = mu[ii]
    plotFigurePhaseSpace("YPhaseSpace",mu_, traceY[ii, :-1], diffTraceY[ii,:],
                         yMin, yMax, dyMin, dyMax)                   
    ax.plot(traceY[ii, :-1], diffTraceY[ii,:],'.',color=cmap.to_rgba(mu_),alpha=alpha_,markersize=3)  # for continuous color map
                             
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
    plotFigure(mu_, traceX[ii,0:nIterations], traceY[ii,0:nIterations],
               xMin, xMax, yMin, yMax) # for a single phase portrait black figure for an arrow                 
                
    ax.plot(traceX[ii,0:nIterations],traceY[ii,0:nIterations],'.',color = color,alpha=alpha_,markersize=3)  # set alpha=0.1 for chaos
    plotPopVsIteration(np.arange(nIterations),mu_, traceX[ii,0:nIterations], traceY[ii,0:nIterations],
                       xMin, xMax, yMin, yMax)   # added to plot x vs. n and y vs. n
    #outPutPoPVsIteration(np.arange(nIterations),mu_, traceX[ii,0:nIterations], traceY[ii,0:nIterations]) # added to save csv of x vs. n and y vs. n
    
    fPtE1, fPtE2, fPtE3 = fixedPoints(mu_)
    
    if plotStreamplot == "True":
       plotFigureAxStream(mu_,traceX[ii,0:nIterations],traceY[ii,0:nIterations],
                          xMin,xMax,yMin,yMax)

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
#plt.show()
fig.savefig(str(folderTraj)+str(simCase)+"_"+"Trajectory_all.jpg")
# end of plot coordinate space (phase portrait) from V506
#%%                 
import moviepy.video.io.ImageSequenceClip
fps=10
image_files = [os.path.join(folderTraj,img)
               for img in sorted(os.listdir(folderTraj))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
clip.write_videofile(str(folderTraj)+str(simCase)+"_"+'trajectory.mp4')
print("------------------------------------------\n")
image_files = [os.path.join(folderPopVsN,img)
               for img in sorted(os.listdir(folderPopVsN))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(str(folderPopVsN)+str(simCase)+"_"+'popVsnIterations.mp4')
print("------------------------------------------\n")
image_files = [os.path.join(folderXPhaseSpace,img)
               for img in sorted(os.listdir(folderXPhaseSpace))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
clip.write_videofile(str(folderXPhaseSpace)+str(simCase)+"_"+'phaseSpaceX.mp4')
print("------------------------------------------\n")
image_files = [os.path.join(folderYPhaseSpace,img)
               for img in sorted(os.listdir(folderYPhaseSpace))
               if img.endswith(".jpg")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
clip.write_videofile(str(folderYPhaseSpace)+str(simCase)+"_"+'phaseSpaceY.mp4')
print("------------------------------------------\n")
if plotStreamplot == True:
   image_files = [os.path.join(folderPhasePortrait,img)
                  for img in sorted(os.listdir(folderPhasePortrait))
                  if img.endswith(".jpg")]
   clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
   clip.write_videofile(str(folderPhasePortrait)+str(simCase)+"_"+'PhasePortrait.mp4')