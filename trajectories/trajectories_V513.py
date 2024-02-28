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
   16. Plot 3d phase portrait vs mu
   17. Rewrite Find suitable muList
   18. Remove ReImAbs_color.jpg plot

"""
import numpy as np
import matplotlib as mpl  # for continuous color map
import matplotlib
#this will prevent the figure from popping up
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(5)
from pathlib import Path
from numpy import linalg as LA
import warnings
import pandas as pd
import sys
#from scipy.interpolate import griddata
from inspect import currentframe
import alphashape
from mpl_toolkits.mplot3d import Axes3D
import os
import moviepy.video.io.ImageSequenceClip
from matplotlib.markers import MarkerStyle

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

warnings.filterwarnings("error")

# Plot with streamplot? 
plotStreamplot = True

# Plot nullclines or not?
plotNullclines = False

# Making animations?
animations = True

# Zoom in streamplot?
streamPlotZoomIn = False
if streamPlotZoomIn:
    xMinStreamPlotZoomIn = 0.2
    yMinStreamPlotZoomIn = 0.2
    xMaxStreamPlotZoomIn = 0.6
    yMaxStreamPlotZoomIn = 0.6
    
mu0_min = 1
mu0_max = 4
pVal_Min = 0
pVal_Max = 1
step  = 0.005

# initX and initY are numbers between 0 and 1
# alpha, beta, and gamma are all greater than 0
###############################################################################
simCase = "Trial_Vorticella"
nsLoops = True # plot Neimark Sackar loops (only for vorticella!)
if simCase == "Trial" or simCase == "Trial_Vorticella" :
   initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.912
   #step  = 0.005  
   dirname = "{:.5f}".format(initX)+\
              "_{:.5f}".format(initY)+\
              "_{:.5f}".format(alpha)+\
              "_{:.5f}".format( beta)+\
              "_{:.5f}".format(gamma)
elif simCase == "Normal": 
    initX, initY, alpha, beta, gamma = 0.200, 0.200, 1.000, 0.001, 0.500 # normal 
elif simCase == "Standard":   
    initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.197 # standard 
elif simCase == "Vorticella": 
    initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.912 # vorticella
    nsLoops = True
    alphaShape = True
elif simCase == "Extinction": 
    initX, initY, alpha, beta, gamma = 0.010, 0.100, 1.000, 0.001, 0.500 # extinction
else:
    sys.exit("Wrong simCase!")
if simCase != "Trial":
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

# folder of plots of 3d phase portrait
dirnameTraj3DPhasePortrait = dirname + "/" + str("trajectory3DPhasePortrait")
Path(dirnameTraj3DPhasePortrait).mkdir(parents=True, exist_ok=True)
folderTraj3DphasePortrait = "./" + str(dirnameTraj3DPhasePortrait) + "/"

if nsLoops: 
    # folder of plots of NSLoop only for Verticella
    dirnameNSLoop3D = dirname + "/" + str("NSLoop3D")
    Path(dirnameNSLoop3D).mkdir(parents=True, exist_ok=True)
    folderNSLoop3D = "./" + str(dirnameNSLoop3D) + "/"

# folder of plots of x and y phase portrait with ax.streamplot # after version V511
if plotStreamplot:# == True:
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

#import os
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
    #return xx, yy
    return np.array([xx, yy])

def func(variables,mu0):
    x, y = variables
    xx, yy = newAttractor(x, y, mu0)
    return np.array([xx,yy])

def jacobian(fs, xs, mu0, h=1e-4):
    """
    Reference: Gezerlis, Numerical Methods in Phyisics with Python, p.284
    """
    n = xs.size
    iden = np.identity(n)
    Jf = np.zeros((n,n))
    fs0 = fs(xs,mu0)
    for j in range(n):  # through columns to allow for vector addition
        fs1 = fs(xs+iden[:,j]*h,mu0)
        Jf[:,j] = ( fs1 - fs0 )/h
    return Jf  

def plotFigure(mu0, ptsX, ptsY,xmin,xmax,ymin,ymax):
    plt.figure(figsize=(16,9))#, constrained_layout=True)
    ax = plt.gca()
    plt.text(xmax * 0.9, ymax * 0.9, r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=24)
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
    plt.tight_layout()
    plt.savefig(str(folderTraj)+str(simCase)+"_"+"Trajectory"+"_{:.5f}".format(mu0)+".jpg")#, bbox_inches = 'tight',pad_inches = 0)
    plt.close()
   
def plotFigureAxStream(mu0, ptsX, ptsY,xmin,xmax,nx,ymin,ymax,ny):

    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)
    X, Y = np.meshgrid(x,y)
    # next value vector
    xNext, yNext = newAttractor(X, Y, mu0)   
    dx = xNext - X
    dy = yNext - Y
    speed = np.sqrt(dx**2+dy**2) # check if it is "+" or ","
    lw = 1 * speed / speed.max()
    
    plt.figure(figsize=(16,9))#, constrained_layout=True)
    plt.title("PhasePortrait")
    ax = plt.gca()
    plt.text(xmax * 0.9, ymax * 0.9, r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=16)    
   
    ax.plot(ptsX,ptsY,'k.',markersize=10)
    
    fPtE1, fPtE2, fPtE3 = fixedPoints(mu0)
    
    typeStability = sFPt(fPtE3,mu0)
    if typeStability == "sink":
        ax.scatter(fPtE3[0],fPtE3[1],s = 80, facecolors='r',edgecolors='r')
    elif typeStability == "source":
        ax.scatter(fPtE3[0],fPtE3[1],s = 80, facecolors='none',edgecolors='r')
    elif typeStability == "saddle":    
        ax.scatter(fPtE3[0],fPtE3[1],s = 80, marker=MarkerStyle("o", fillstyle="right"),facecolors='r',edgecolors='r')
        ax.scatter(fPtE3[0],fPtE3[1],s = 80, marker=MarkerStyle("o", fillstyle="left"),facecolors='none',edgecolors='r')
    elif typeStability == "nonhyperbole":
        ax.scatter(fPtE3[0],fPtE3[1],s = 80, marker=r'$?$',facecolors='none',edgecolors='r')
    else:
        print("({:.3f},{:.3f}): other cases. Check line {}".format(fPtE3[0],fPtE3[1],get_linenumber()) )
    
    ax.streamplot(X,Y,dx,dy, density=1,linewidth=1, color = 'b', arrowstyle='->')
    ax.quiver(X, Y, dx, dy,color='g',width=0.001)#,scale=500,scale_units='height')
    
    if plotNullclines:
        cs = ax.contour(X,Y,dx,levels=[0], linewidths=lw, colors='r')
        ax.clabel(cs,cs.levels,fmt=r'$\dot{x}=0$',font=32)
   
        cs = ax.contour(X,Y,dy,levels=[0], linewidths=1, colors='k')
        ax.clabel(cs,cs.levels,fmt=r'$\dot{y}=0$',font=32)
   
        ax.legend()
    
    plt.minorticks_on()
    minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
    minorLocatorY = AutoMinorLocator(5)
    ax.set_xlabel("x",size = 16)
    ax.set_ylabel("y",size = 16)
    ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    if streamPlotZoomIn:
        ax.set_xlim([xMinStreamPlotZoomIn,xMaxStreamPlotZoomIn])
        ax.set_ylim([yMinStreamPlotZoomIn,yMaxStreamPlotZoomIn])
    else:
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])        
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(folderPhasePortrait)+str(simCase)+"_"+"PhasePortrait"+"_{:.5f}".format(mu0)+".jpg")#, bbox_inches = 'tight',pad_inches = 0)
    plt.close()

# added to save csv of x vs. n and y vs. n
def outPutPoPVsIteration(nIterations,mu0, ptsX, ptsY):
    data = {'nIterations': nIterations,'prey': ptsX,'predator':ptsY}
    df = pd.DataFrame(data)
    df_file = open(str(folderPopVsN)+str(simCase)+"PopVsN"+"_{:.5f}".format(mu0)+".csv",'w',newline='')
    df.to_csv(df_file, sep=',', encoding='utf-8',index=False)
    df_file.close()

# added to plot x vs. n and y vs. n  
def plotPopVsIteration(nIterations,mu0, ptsX, ptsY,xmin,xmax,ymin,ymax):
    fig, ax = plt.subplots(2, 1, figsize=(16,11))#, constrained_layout=True)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12) 

    # the first subplot
    xAxis = 200
    yAxis = ymax * 0.8
    ax[0].text(xAxis*0.8,yAxis,r'$\mu_{0} = $' + "{:.3f}".format(mu0),fontsize=24)
    ax[0].scatter(nIterations,ptsX,color = 'k' , s = 16 , marker = 'o')
    
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
    ax[1].scatter(nIterations,ptsY,color = 'k', s = 16, marker = 'o')
    ax[1].set_xlabel("$n$",size = 32)
    ax[1].set_ylabel("$y_{n}$",size = 32)  
    ax[1].xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    ax[1].yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax[1].set_xlim([-10,xAxis]) 
    ax[1].set_ylim([ymin,ymax])
    ax[1].grid(True)   
    plt.tight_layout()       
    #plt.show()
    plt.savefig(str(folderPopVsN)+str(simCase)+"_"+"PopVsN"+"_{:.5f}".format(mu0)+".jpg")#, bbox_inches = 'tight',pad_inches = 0)
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
    plt.tight_layout()
    if XorY == "XPhaseSpace": 
        plt.savefig(str(folderXPhaseSpace)+str(simCase)+"_"+"XPhaseSpace"+\
                    "_{:.5f}".format(mu0)+".jpg")#, bbox_inches = 'tight',pad_inches = 0)      
    else:
        plt.savefig(str(folderYPhaseSpace)+str(simCase)+"_"+"YPhaseSpace"+\
                    "_{:.5f}".format(mu0)+".jpg")#, bbox_inches = 'tight',pad_inches = 0)
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
    w_ = LA.eigvals( jacobian(func, np.array([fPt[0], fPt[1]]), mu0 ) )
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

# find stability of a single fixed point
def sFPt(fPt,mu0):
    tolerance = 1e-8
    fPtX = fPt[0]
    fPtY = fPt[1]
    w= LA.eigvals( jacobian( func,np.array([fPtX,fPtY]), mu0 ) )
    
    ReW0, ImW0 = np.real(w[0]), np.imag(w[0])#, map(abs,w0E3)
    absW0 = np.sqrt( ReW0**2 + ImW0**2 )
    
    if abs(absW0-abs(w[0]))>1e-4: 
        print("error for w0, Line {}".format(get_linenumber()))
        sys.exit()
    
    ReW1, ImW1 = np.real(w[1]), np.imag(w[1])#, map(abs,w1E3)
    absW1 = np.sqrt( ReW1**2 + ImW1**2 )
    
    if abs(absW1-abs(w[1]))>1e-4:
        print("error for w1, Line {}".format(get_linenumber()))
        sys.exit()
    
    if abs(absW0-1)<tolerance or abs(absW1-1)<tolerance:
        writeLog("mu0 = {:.3f}, ({:.3f},{:.3f}): nonhyperbole, linearization could fail.\n".format(mu0,fPtX,fPtY)) 
        typeStability = "nonhyperbole"      
    elif absW0<1 and absW1<1:
        writeLog("mu0 = {:.3f},({:.3f},{:.3f}): sink.\n".format(mu0,fPtX,fPtY))
        typeStability = "sink"        
    elif absW0>1 and absW1>1:
        writeLog("mu0 = {:.3f},({:.3f},{:.3f}): source.\n".format(mu0,fPtX,fPtY))
        typeStability = "source"
    elif (absW0<1 and absW1>1) or (absW0>1 and absW1<1) :
        writeLog("mu0 = {:.3f},({:.3f},{:.3f}): saddle.\n".format(mu0,fPtX,fPtY))
        typeStability = "saddle"       
    else:
        writeLog("mu0 = {:.3f},({:.3f},{:.3f}): other cases. Check line {}\n".format(mu0, fPtX, fPtY, get_linenumber()))
        typeStability = "None"
        sys.exit()
    return typeStability

# Find suitable muList
nIterations = 2500

mu = np.arange(mu0_min, mu0_max, step)
#%%
muList = []
targetMaxMuList = 3

traceX = np.zeros([len(mu), nIterations])
traceY = np.zeros([len(mu), nIterations])

for ii, mu_ in enumerate(mu):
    traceX[ii,0] = initX
    traceY[ii,0] = initY

    i = 0 
    while ( all( pVal>=pVal_Min and pVal<=pVal_Max 
                 for pVal in newAttractor(traceX[ii,i], traceY[ii,i], mu_) )
            and i < nIterations-1):
            x_next, y_next = newAttractor(traceX[ii,i], traceY[ii,i], mu_)
            traceX[ii,i+1] = x_next
            traceY[ii,i+1] = y_next
            i += 1
    if i == nIterations-1: 
        muList += [mu_]
    else:
        break
if len(muList) == 0 or muList[-1] < targetMaxMuList:
   writeLog("Inappropriate parameters. Empty or short muList")
   print("--------------------------------------------------")
   sys.exit()
else:
    writeLog(r'mu0_max = '+ "{:.5f}\n".format( muList[-1] ) )
    print("--------------------------")
    mu = np.array(muList)     
    
if len(traceX[:,0])!=len(mu) or len(traceY[:,0])!=len(mu):
   writeLog("Warning Line {}! len(traceX) = {} and len(traceY) = {}, but len(mu) = {}".format(
           get_linenumber(),len(traceX[:,0]),len(traceY[:,0]),len(mu)))
   print("--------------------------")
   #sys.exit()

traceX = traceX[:len(mu),:]
traceY = traceY[:len(mu),:]

if len(traceX[:,0])!=len(mu) or len(traceY[:,0])!=len(mu):
   print("Warning Line {}! len(traceX) = {} and len(traceY) = {}, but len(mu) = {}".format(
           get_linenumber(),len(traceX[:,0]),len(traceY[:,0]),len(mu)))
   sys.exit()
# end of find suitable muList V508
#%% Find xlim, ylim for phase portrait, and phase spaces automatically.

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

maxMu = np.max(mu)
minMu = np.min(mu)

color_name = "Greys" #'viridis'
cmap = plt.cm.get_cmap(color_name)#mpl.colormaps.get_cmap(color_name)
#norm = mpl.colors.Normalize(vmin=mu.min(), vmax=mu.max())   # for continuous color map
norm = mpl.colors.LogNorm(vmin=mu.min(), vmax=mu.max())
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
    #w, area = fixedPointType(fPtE1, mu_)
    w = LA.eigvals( jacobian( func,np.array([fPtE1[0],fPtE1[1]]), mu_ ) )
    w0E1 += [w[0]]
    w1E1 += [w[1]]
    
    # type for fixpoint E2
    #w, area = fixedPointType(fPtE2, mu_)
    w = LA.eigvals( jacobian( func,np.array([fPtE2[0],fPtE2[1]]), mu_ ) )
    w0E2 += [w[0]]
    w1E2 += [w[1]]     

    # type for fixpoint E3
    #w, area = fixedPointType(fPtE3, mu_)
    w = LA.eigvals( jacobian( func,np.array([fPtE3[0],fPtE3[1]]), mu_ ) )
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
df_file = open(str(folderEigVals)+str(simCase)+"_ReImAbs.csv",'w',newline='')
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
    
    ax.plot( df.iloc[:,0].values, df.iloc[:,i+1].values, 'k-')# append .values
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
plt.savefig(str(folderEigVals)+str(simCase)+"_"+"ReImAbs.jpg")#, bbox_inches = 'tight',pad_inches = 0)
plt.close()
# end of plot Re(Omega0), Im(Omega0), |omega0|, Re(Omega1), Im(omega1), |omega1|
# vs mu0 for fixed points E1, E2, and E3
#%% plot 3 fixed points
fig = plt.figure(figsize=(16,9))#, constrained_layout=True)
ax = fig.add_subplot(111)
  
for ii in range(len(mu)):
    mu_ = mu[ii]
    color = cmap.to_rgba(mu_) 
    
    fPtE1, fPtE2, fPtE3 = fixedPoints(mu_)
    # type for fixpoint E1
    w, area = fixedPointType(fPtE1, mu_)
    if fPtE1[0]>=0.0 and fPtE1[0]<=1.0 and fPtE1[1]>=0.0 and fPtE1[1]<=1.0:
    #   scatter = ax.scatter(fPtE1[0],fPtE1[1],color = color, s = area, alpha=alpha_)
    #   handles1, labels1, i10, i50, i100, i200 =\
    #   handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       line1, = ax.plot(fPtE1[0],fPtE1[1],'r.',markersize=3)
       #line1.set_label(r'$\E_{1}$')
        
    # type for fixpoint E2
    w, area = fixedPointType(fPtE2, mu_)
    if fPtE2[0]>=0.0 and fPtE2[0]<=1.0 and fPtE2[1]>=0.0 and fPtE2[1]<=1.0:
    #   scatter = ax.scatter(fPtE2[0],fPtE2[1],color = color, s = area, alpha=alpha_)
    #   handles1, labels1, i10, i50, i100, i200 =\
    #   handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       line2, = ax.plot(fPtE2[0],fPtE2[1],'b.',markersize=3)
       #line2.set_label(r'$\E_{2}$')
        
    # type for fixpoint E3
    w, area = fixedPointType(fPtE3, mu_)
    if fPtE3[0]>=0.0 and fPtE3[0]<=1.0 and fPtE3[1]>=0.0 and fPtE3[1]<=1.0:
    #   scatter = ax.scatter(fPtE3[0],fPtE3[1],color = color, s = area, alpha=alpha_)
    #   handles1, labels1, i10, i50, i100, i200 =\
    #   handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       line3, = ax.plot(fPtE3[0],fPtE3[1],'g.',markersize=3)
       #line3.set_label(r'$\E_{3}$')
           
plt.minorticks_on()  # no fig.minorticks_on()
ax.set_xlabel("x",size = 16)
ax.set_ylabel("y",size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
#ax.legend(loc="upper left")
#plt.tight_layout()
ax.grid(True)
plt.tight_layout()
fig.savefig(str(folderTraj)+str(simCase)+"_"+"fixedPoints.png")#, bbox_inches = 'tight',pad_inches = 0)
plt.close()
# end of plot 3 fixed points
#%% plot coordinate space (phase portrait)
# type of fixed points are also included after V506
fig = plt.figure(figsize=(16,9))#, constrained_layout=True)
ax = fig.add_subplot(111)

nIterations = 2500 # nIterations=600 if the black area is large.    
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
    outPutPoPVsIteration(np.arange(nIterations),mu_, traceX[ii,0:nIterations], traceY[ii,0:nIterations]) # added to save csv of x vs. n and y vs. n
    
    fPtE1, fPtE2, fPtE3 = fixedPoints(mu_)
    
    if plotStreamplot:
       nx = 10
       ny = 10      
       plotFigureAxStream(mu_,traceX[ii,0:nIterations],traceY[ii,0:nIterations],
                          xMin, xMax, nx, yMin, yMax, ny)
       x = np.linspace(xMin, xMax, nx) # xMin, xMax, nx
       y = np.linspace(yMin, yMax, ny) # yMin, yMax, ny
       X, Y = np.meshgrid(x,y)
       # next value vector
       xNext, yNext = newAttractor(X, Y, mu_)   
       dx = xNext - X
       dy = yNext - Y
       speed = np.sqrt(dx**2+dy**2) # check if it is "+" or ","
       lw = 1*speed / speed.max()
       
       for ii in [462,474,511,540,570]:
       
           #ax.streamplot(X,Y,dx,dy, density=1,linewidth=1, color = 'b', arrowstyle='->')
           #ax.quiver(X, Y, dx, dy,color='r',width=0.001)#,scale=500,scale_units='height')
       
           if plotNullclines:
               cs = ax.contour(X,Y,dx,levels=[0], linewidths=1, colors='r') # be care of colors!
               ax.clabel(cs,cs.levels,fmt=r'$\dot{x}=0$',font=32)
   
               cs = ax.contour(X,Y,dy,levels=[0], linewidths=1, colors='k')
               ax.clabel(cs,cs.levels,fmt=r'$\dot{y}=0$',font=32)
   
               ax.legend()
       
    # type for fixpoint E1
    w, area = fixedPointType(fPtE1, mu_)
    if fPtE1[0]>=0.0 and fPtE1[0]<=1.0 and fPtE1[1]>=0.0 and fPtE1[1]<=1.0:
    #   scatter = ax.scatter(fPtE1[0],fPtE1[1],color = color, s = area, alpha=alpha_)
    #   handles1, labels1, i10, i50, i100, i200 =\
    #   handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       ax.plot(fPtE1[0],fPtE1[1],'k.',markersize=1)
        
    # type for fixpoint E2
    w, area = fixedPointType(fPtE2, mu_)
    if fPtE2[0]>=0.0 and fPtE2[0]<=1.0 and fPtE2[1]>=0.0 and fPtE2[1]<=1.0:
    #   scatter = ax.scatter(fPtE2[0],fPtE2[1],color = color, s = area, alpha=alpha_)
    #   handles1, labels1, i10, i50, i100, i200 =\
    #   handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       ax.plot(fPtE2[0],fPtE2[1],'k.',markersize=1)
        
    # type for fixpoint E3
    w, area = fixedPointType(fPtE3, mu_)
    if fPtE3[0]>=0.0 and fPtE3[0]<=1.0 and fPtE3[1]>=0.0 and fPtE3[1]<=1.0:
    #   scatter = ax.scatter(fPtE3[0],fPtE3[1],color = color, s = area, alpha=alpha_)
    #   handles1, labels1, i10, i50, i100, i200 =\
    #   handlesAndLabels(scatter, area, handles1, labels1, i10, i50, i100, i200)
       ax.plot(fPtE3[0],fPtE3[1],'k.',markersize=1)         
           
#cbar = fig.colorbar(cmap,ticks = np.arange(1,int(np.ceil(norm.vmax))+1),aspect=40)
#cbar.ax.get_xaxis().labelpad = 15
#cbar.ax.set_xlabel(r'$\mu_{0}$')
plt.minorticks_on()  # no fig.minorticks_on()
ax.set_xlabel("x",size = 16)
ax.set_ylabel("y",size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax.grid(True)
# remove the xlim and ylim if we want a whole-range plot
if streamPlotZoomIn:
    ax.set_xlim([xMinStreamPlotZoomIn,xMaxStreamPlotZoomIn])
    ax.set_ylim([yMinStreamPlotZoomIn,yMaxStreamPlotZoomIn])
else:
    ax.set_xlim([xMin, xMax])
    ax.set_ylim([yMin, yMax])  
plt.tight_layout()
fig.savefig(str(folderTraj)+str(simCase)+"_"+"Trajectory_all.png")#, bbox_inches = 'tight',pad_inches = 0)
plt.close()
# end of plot coordinate space (phase portrait) from V506
#%% plot phase spaces
## plot phase space of x
fig = plt.figure(figsize=(16,9))#, constrained_layout=True)
ax = fig.add_subplot(111)
alpha_ = 0.2

for ii in range(len(mu)):
    mu_ = mu[ii]
    plotFigurePhaseSpace("XPhaseSpace",mu_, traceX[ii, :-1], diffTraceX[ii,:],
                         xMin, xMax, dxMin, dxMax)
    ax.plot(traceX[ii, :-1], diffTraceX[ii,:],'.',color=cmap.to_rgba(mu_),alpha=alpha_,markersize=3)# for continuous color map
            
#cbar = fig.colorbar(cmap,ticks = np.arange(1,int(np.ceil(norm.vmax))+1), aspect = 40)
#cbar.ax.get_xaxis().labelpad = 15
#cbar.ax.set_xlabel(r'$\mu_{0}$')
plt.minorticks_on()
ax.set_xlabel("$x$",size = 16)
ax.set_ylabel(r'$\Delta x$',size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax.grid(True)
plt.xlim(0.300, 0.400)
plt.ylim(-0.045, 0.045)
#plt.show()
fig.savefig(str(folderXPhaseSpace)+str(simCase)+"_"+"phaseSpaceX_all.png")
plt.close()

## plot phase space of y
fig = plt.figure(figsize=(16,9))#, constrained_layout=True)
ax = fig.add_subplot(111)

for ii in range(len(mu)):
    mu_ = mu[ii]
    plotFigurePhaseSpace("YPhaseSpace",mu_, traceY[ii, :-1], diffTraceY[ii,:],
                         yMin, yMax, dyMin, dyMax)                   
    ax.plot(traceY[ii, :-1], diffTraceY[ii,:],'.',color=cmap.to_rgba(mu_),alpha=alpha_,markersize=3)  # for continuous color map
                             
#cbar = fig.colorbar(cmap,ticks = np.arange(1,int(np.ceil(norm.vmax))+1), aspect = 40)
#cbar.ax.get_xaxis().labelpad = 15
#cbar.ax.set_xlabel(r'$\mu_{0}$')
plt.minorticks_on()
ax.set_xlabel("$y$",size = 16)
ax.set_ylabel(r'$\Delta y$',size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax.grid(True)
plt.xlim(0.300, 0.400)
plt.ylim(-0.045, 0.045)
#plt.show()
plt.tight_layout()
fig.savefig(str(folderYPhaseSpace)+str(simCase)+"_"+"phaseSpaceY_all.png")#, bbox_inches = 'tight',pad_inches = 0)
plt.close()
# end of plot phase space
#%% plot 3d phase portrait vs mu (animation)
nIterations = 600 # nIterations=600 if the black area is large.
if simCase == "Vorticella" or simCase == "Trial_Vorticella": nIterations = 2500 
    
avoidTransient = 200
alpha_ = 0.8  

angleStep = 20
step3d = 1
step3dMu = 1

# for animation
for elev in range(0,360,angleStep):
    for azim in range(0,360,angleStep):
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(elev, azim) #elev, azim 
                                 # (0,0) -> YZ; (0, -90) -> XZ; (90, 270) -> XY 
        
        iiList = np.arange(0,len(mu),step3dMu)
        for ii in iiList: #range(0,len(mu),step3dMu):
            mu_ = mu[ii]
            color = cmap.to_rgba(mu_)
            if simCase == "Normal":
                ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                        np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                        traceX[ii,avoidTransient:nIterations:step3d],
                        '.',color=color,alpha=alpha_,markersize=1) 
            if simCase == "Extinction":
                if ii % 5 == 0:
                    ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                        np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                        traceX[ii,avoidTransient:nIterations:step3d],
                        '.',color='k',alpha=alpha_,markersize=1)
                else:
                    ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                            np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                            traceX[ii,avoidTransient:nIterations:step3d],
                            '.',color=color,alpha=alpha_,markersize=1)
            
            # the block is for Vorticella only
            if simCase == "Trial_Vorticella":             
                ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                        np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                        traceX[ii,avoidTransient:nIterations:step3d],
                        '.',color=color,alpha=alpha_,markersize=1)                
                             
            if simCase == "Vorticella":
                if (ii % 5 == 0) and (i <= 400):
                    ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                            np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                            traceX[ii,avoidTransient:nIterations:step3d],
                            '.',color=color,alpha=alpha_,markersize=1)                
                
                if (ii > 440 and ii % 10 == 0) or (ii == iiList[-1]):
                    points = np.c_[traceY[ii,avoidTransient:nIterations:step3d],
                                   traceX[ii,avoidTransient:nIterations:step3d]]
                    ## delimit the boundary 
                    ## reference:
                    ## https://stackoverflow.com/questions/44685052/boundary-points-from-a-set-of-coordinates
                    alpha = 0.95#alphashape.optimizealpha(points,silent = True)
                    hull = alphashape.alphashape(points, alpha)
                    hull_pts = hull.exterior.coords.xy
                    ax.plot(hull_pts[0], np.array( [mu_] * len( hull_pts[0] ) ), hull_pts[1],
                            '-',color='k',alpha=alpha_,markersize=1) 
            
        ax.text(yMax * 0.8, 3.5, xMax * 0.8,
                r'$\theta$ = {}$^\circ$, $\phi$ = {}$^\circ$'.format(elev,azim),
                color='red', fontsize = 12 )            

        #plt.minorticks_on()  # no fig.minorticks_on()
        ax.set_xlabel("y",size = 16)
        ax.set_ylabel(r'$\mu_{0}$',size = 16)
        ax.set_zlabel("x",size = 16)
        #ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
        #ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
        ax.set_ylim([3,4])
        
        # Get rid of the panes                          
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

        # Get rid of the spines                         
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        ## Get rid of the ticks                          
        #ax.set_xticks([])                               
        #ax.set_yticks([])                               
        #ax.set_zticks([])
        
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(str(folderTraj3DphasePortrait)+str(simCase)+"_"+"phasePortrait3D"+"_"+\
                    "{:03d}_{:03d}.jpg".format(elev,azim))
        plt.close()
# end of plot 3d phase portrait vs mu (animation)
#%% plot 3d phase portrait vs mu (rotate with mouth)
matplotlib.use('Qt5Agg') # show figure

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')

# determine elev and azim
elev = 20
azim = 60

ax.view_init(elev, azim) #elev, azim 
                         # (0,0) -> YZ; (0, -90) -> XZ; (90, 270) -> XY 

iiList = np.arange(0,len(mu),step3dMu)

for ii in iiList: #range(0,len(mu),step3dMu):
    mu_ = mu[ii]
    color = cmap.to_rgba(mu_)
    if simCase == "Normal":
        ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                traceX[ii,avoidTransient:nIterations:step3d],
                '.',color=color,alpha=alpha_,markersize=1) 
    if simCase == "Extinction":
        if ii % 5 == 0:
            ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                    np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                    traceX[ii,avoidTransient:nIterations:step3d],
                    '.',color='k',alpha=alpha_,markersize=1)
        else:
                    ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                            np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                            traceX[ii,avoidTransient:nIterations:step3d],
                            '.',color=color,alpha=alpha_,markersize=1)
            
    # the block is for Vorticella only
    if simCase == "Trial_Vorticella":             
        ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                traceX[ii,avoidTransient:nIterations:step3d],
                '.',color=color,alpha=alpha_,markersize=1)                
                             
    if simCase == "Vorticella":
        if (ii % 5 == 0) and (i <= 400):
            ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                    np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                    traceX[ii,avoidTransient:nIterations:step3d],
                    '.',color=color,alpha=alpha_,markersize=1)                
                
        if (ii > 440 and ii % 10 == 0) or (ii == iiList[-1]):
            points = np.c_[traceY[ii,avoidTransient:nIterations:step3d],
                           traceX[ii,avoidTransient:nIterations:step3d]]
            ## delimit the boundary 
            ## reference:
            ## https://stackoverflow.com/questions/44685052/boundary-points-from-a-set-of-coordinates
            alpha = 0.95#alphashape.optimizealpha(points,silent = True)
            hull = alphashape.alphashape(points, alpha)
            hull_pts = hull.exterior.coords.xy
            ax.plot(hull_pts[0], np.array( [mu_] * len( hull_pts[0] ) ), hull_pts[1],
                    '-',color='k',alpha=alpha_,markersize=1)         

ax.text(yMax * 0.9, 3.5, xMax * 0.9,
        r'$\gamma = {}, \theta$ = {}$^\circ$, $\phi$ = {}$^\circ$'.format(gamma,elev,azim),
        color='red', fontsize = 12 )  

#plt.minorticks_on()  # no fig.minorticks_on()
ax.set_xlabel("y",size = 16)
ax.set_ylabel(r'$\mu_{0}$',size = 16)
ax.set_zlabel("x",size = 16)
#ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
#ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
ax.set_ylim([3.0,4.0])
        
# Get rid of the panes                          
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

# Get rid of the spines                         
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
ax.grid(True)
#ax.set_title(r'$\gamma = {}$'.format(gamma))
plt.tight_layout()
plt.show()
fig.savefig(str(folderTraj3DphasePortrait)+str(simCase)+"_"+"phasePortrait3D"+"_"+\
            "{:.3f}_{:03d}_{:03d}.jpg".format(gamma,elev,azim))
# end of plot 3d phase portrait vs mu (rotate with mouth)
#%% one big plot for phase portrait 3D All
fig = plt.figure(figsize=(9, 16))#, constrained_layout=True)
angleStep = 84
step3d = 5
for i in range(0,int((360/angleStep)**2)):
    
    ax = fig.add_subplot(6, 3, i+1, projection='3d')

    ## debug information
    #k = 10
    #j = int(360/angleStep)
    #if i>= k * j and i < ((k+1) * j):
    #    elev = angleStep * (i//int(360/angleStep))
    #    azim = angleStep * (i%int(360/angleStep))
    #    print("--------------------------------------")
    #    print("i = {}, elev = {}, azim = {}".format(i, elev,azim))
    ## end of debug information
    elev = angleStep * (i//int(360/angleStep))
    azim = angleStep * (i%int(360/angleStep)) 
    
    ax.view_init(elev, azim) #elev, azim 
                             # (0,0) -> YZ; (0, -90) -> XZ; (90, 270) -> XY 
    
    iiList = np.arange(0,len(mu),step3dMu)
    for ii in iiList: #range(0,len(mu),step3dMu):
        mu_ = mu[ii]
        color = cmap.to_rgba(mu_)
        if simCase == "Normal":
            ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                    np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                    traceX[ii,avoidTransient:nIterations:step3d],
                    '.',color=color,alpha=alpha_,markersize=1) 
        if simCase == "Extinction":
            if ii % 5 == 0:
                ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                        np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                        traceX[ii,avoidTransient:nIterations:step3d],
                        '.',color='k',alpha=alpha_,markersize=1)
            else:
                ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                        np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                        traceX[ii,avoidTransient:nIterations:step3d],
                        '.',color=color,alpha=alpha_,markersize=1)
            
        # the block is for Vorticella only    
        if simCase == "Trial_Vorticella":
            ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                    np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                    traceX[ii,avoidTransient:nIterations:step3d],
                    '.',color=color,alpha=alpha_,markersize=1)                

        if simCase == "Vorticella":
            if (ii % 10 == 0) and (i <= 400):
                ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                        np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                        traceX[ii,avoidTransient:nIterations:step3d],
                        '.',color=color,alpha=alpha_,markersize=1)                
                
            if (ii > 440 and ii % 20 == 0) or (ii == iiList[-1]):
                points = np.c_[traceY[ii,avoidTransient:nIterations:step3d],
                               traceX[ii,avoidTransient:nIterations:step3d]]
            #    # delimit the boundary 
            #    # reference:
            #    # https://stackoverflow.com/questions/44685052/boundary-points-from-a-set-of-coordinates
                alpha = 0.95#alphashape.optimizealpha(points,silent = True)
                hull = alphashape.alphashape(points, alpha)
                hull_pts = hull.exterior.coords.xy
                ax.plot(hull_pts[0], np.array( [mu_] * len( hull_pts[0] ) ), hull_pts[1],
                        '-',color='k',alpha=alpha_,markersize=1)  
        
    ax.text(yMax * 0.5, 3.5, xMax * 0.5,
            r'$\theta$ = {}$^\circ$, $\phi$ = {}$^\circ$'.format(elev,azim),
            color='red',fontsize = 8 )            

    #ax.set_xlabel("y",size = 8)
    #ax.set_ylabel(r'$\mu_{0}$',size = 8)
    #ax.set_zlabel("x",size = 8)
    #ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
    #ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
    ax.set_ylim([3,4])
    #ax.grid(True)
    ax.axis('off')
    
    # Get rid of the panes                          
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

    # Get rid of the spines                         
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
    # Get rid of the ticks                          
    ax.set_xticks([])                               
    ax.set_yticks([])                               
    ax.set_zticks([])    
plt.tight_layout()    
fig.savefig(str(folderTraj3DphasePortrait)+str(simCase)+"_"+"phasePortrait3DAll.png")#, bbox_inches = 'tight',pad_inches = 0)
plt.close()
# end of one big plot for phase portrait 3D All
#%%       
# plot 3D Neimark Sackar loops (animation)
if nsLoops:
    nIterations = 2500
    avoidTransient = 200
    alpha_ = 0.6
    step3dMu = 1
    angleStep = 20
    step3d = 1
    iIndex = [411,430,462,474,511,540,570]
    for elev in range(0,360,angleStep):
        for azim in range(0,360,angleStep):
            fig = plt.figure(figsize=(16,9))#, constrained_layout=True)
            ax = fig.add_subplot(111, projection='3d')

            ax.view_init(elev, azim) #elev, azim 
                                     # (0,0) -> YZ; (0, -90) -> XZ; (90, 270) -> XY 
    
            for i in iIndex:
                mu_ = mu[i]
                color = cmap.to_rgba(mu_)
            
                ax.plot(traceY[i,avoidTransient:nIterations:step3d],
                        np.array( [mu_] * len( traceY[i,avoidTransient:nIterations:step3d] ) ),
                        traceX[i,avoidTransient:nIterations:step3d],
                        '.',color=color,alpha=alpha_,markersize=1)

                points = np.c_[traceY[i,avoidTransient:nIterations:step3d],
                               traceX[i,avoidTransient:nIterations:step3d]]
                #delimit the boundary 
                #reference:
                #https://stackoverflow.com/questions/44685052/boundary-points-from-a-set-of-coordinates
                if simCase == "Vorticella":    
                    if i != 511:
                        alpha = 0.95#alphashape.optimizealpha(points,silent = True)
                        hull = alphashape.alphashape(points, alpha)
                        hull_pts = hull.exterior.coords.xy
                        ax.plot(hull_pts[0], np.array( [mu_] * len( hull_pts[0] ) ), hull_pts[1],
                               '-',color='k',alpha=alpha_,markersize=1)        
        
            ax.text(yMax * 0.5, 3.5, xMax * 0.5,
                    r'$\theta$ = {}$^\circ$, $\phi$ = {}$^\circ$'.format(elev,azim),fontsize = 12 )            

            #plt.minorticks_on()  # no fig.minorticks_on()
            ax.set_xlabel("y",size = 16)
            ax.set_ylabel(r'$\mu_{0}$',size = 16)
            ax.set_zlabel("x",size = 16)
            #ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
            #ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
            ax.set_ylim([3,4])
            
            # Get rid of the panes                          
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
            
            # Get rid of the spines                         
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
            # Get rid of the ticks                          
            #ax.set_xticks([])                               
            #ax.set_yticks([])                               
            #ax.set_zticks([])  
            
            ax.grid(True)
            plt.tight_layout()
            fig.savefig(str(folderNSLoop3D)+str(simCase)+"_"+"NSLoops"+"_"+\
                        "{:03d}_{:03d}.jpg".format(elev,azim))#, bbox_inches = 'tight',pad_inches = 0)
            plt.close()
# end of plot 3D Neimark Sackar loops (animation)     
# one big plot for 3D Neimark Sackar loops
#if nsLoops:
    fig = plt.figure(figsize=(9,16))#, constrained_layout=True)
    angleStep = 84
    step3d = 5
    
    for i in range(0,int((360/angleStep)**2)):
        ax = fig.add_subplot(6, 3, i+1, projection='3d')

        elev = angleStep * (i//int(360/angleStep))
        azim = angleStep * (i%int(360/angleStep)) 
    
        ax.view_init(elev, azim) #elev, azim 
                                 # (0,0) -> YZ; (0, -90) -> XZ; (90, 270) -> XY 
    
        for ii in iIndex:
            mu_ = mu[ii]
            color = cmap.to_rgba(mu_)
            
            ax.plot(traceY[ii,avoidTransient:nIterations:step3d],
                    np.array( [mu_] * len( traceY[ii,avoidTransient:nIterations:step3d] ) ),
                    traceX[ii,avoidTransient:nIterations:step3d],
                    '.',color=color,alpha=alpha_,markersize=1)

            points = np.c_[traceY[ii,avoidTransient:nIterations:step3d],
                           traceX[ii,avoidTransient:nIterations:step3d]]
            # delimit the boundary 
            # reference:
            # https://stackoverflow.com/questions/44685052/boundary-points-from-a-set-of-coordinates
            if simCase == "Vorticella":    
                if ii != 511:
                    alpha = 0.95#alphashape.optimizealpha(points,silent = True)
                    hull = alphashape.alphashape(points, alpha)
                    hull_pts = hull.exterior.coords.xy
                    ax.plot(hull_pts[0], np.array( [mu_] * len( hull_pts[0] ) ), hull_pts[1],
                            '-',color='k',alpha=alpha_,markersize=1)        
        
        ax.text(yMax * 0.5, 3.5, xMax * 0.5,
                r'$\theta$ = {}$^\circ$, $\phi$ = {}$^\circ$'.format(elev,azim),fontsize = 8 )            

        #plt.minorticks_on()  # no fig.minorticks_on()
        #ax.set_xlabel("y",size = 10)
        #ax.set_ylabel(r'$\mu_{0}$',size = 10)
        #ax.set_zlabel("x",size = 10)
        #ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
        #ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
        ax.set_ylim([3,4])
        #ax.grid(True)
        ax.axis('off')
        
        # Get rid of the panes                          
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        
        # Get rid of the spines                         
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Get rid of the ticks                          
        ax.set_xticks([])                               
        ax.set_yticks([])                               
        ax.set_zticks([])      
    plt.tight_layout()    
    fig.savefig(str(folderNSLoop3D)+str(simCase)+"_"+"NSLoops.png")#, bbox_inches = 'tight',pad_inches = 0)
    plt.close()
# end of one big plot for 3D Neimark Sackar loops  
#%%                 
#import moviepy.video.io.ImageSequenceClip
if animations:
    warnings.filterwarnings("ignore", category=DeprecationWarning)    
    fps=10
    image_files = [os.path.join(folderTraj,img)
                   for img in sorted(os.listdir(folderTraj))
                   if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
    clip.write_videofile(str(folderTraj)+str(simCase)+"_"+'trajectory.mp4')
    print("------------------------------------------\n")
    image_files = [os.path.join(folderTraj3DphasePortrait,img)
                   for img in sorted(os.listdir(folderTraj3DphasePortrait))
                   if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=1)
    clip.write_videofile(str(folderTraj3DphasePortrait)+str(simCase)+"_"+'phasePortrait3D.mp4')
    print("------------------------------------------\n")
    if nsLoops:
       image_files = [os.path.join(folderNSLoop3D,img)
                        for img in sorted(os.listdir(folderNSLoop3D))
                        if img.endswith(".jpg")]
       clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=1)
       clip.write_videofile(str(folderNSLoop3D)+str(simCase)+"_"+'NSLoop3D.mp4')
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
    if plotStreamplot:
       image_files = [os.path.join(folderPhasePortrait,img)
                      for img in sorted(os.listdir(folderPhasePortrait))
                      if img.endswith(".jpg")]
       clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files[0:-1], fps=fps)
       clip.write_videofile(str(folderPhasePortrait)+str(simCase)+"_"+'PhasePortrait.mp4')