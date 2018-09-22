import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils import getFullVar, calcReleases
import math

M = 7 # number of policy inputs (sSL, sHB, sTQ, sTB, zHN_fcst, sin(2pi*t/365 - phi1), cos(2pit/365 - phi2))
K = 4 # number of policy output (uSL, uHB, uTQ, uTB)
N = M + K + 1 # number of RBFs

Input_ranges = np.loadtxt('Input_ranges.txt', usecols=[1,2])
Output_ranges = np.loadtxt('Output_ranges.txt', usecols=[1,2])
columns = [r'$s_t^{SL}$', r'$s_t^{HB}$', r'$s_t^{TQ}$', r'$s_t^{TB}$', r'$\tilde{z}_{t+2}^{HN}$']
dates = pd.read_csv('dates.csv',delimiter=',',header=None)

def showNegVarianceCases():
    # find days of minimum and maximum correlation of inputs across select policies
    solns = [33, 52, 37, 35] # Best Flood, Deficit, Hydro, Compromise
    solnNames = ['Best Flood Policy','Best Deficit Policy','Best Hydro Policy','Compromise Policy']
    days = 365
    minCorr = 1
    maxCorr = -1
    for i, soln in enumerate(solns):
        for day in range(days):
            simulations = np.loadtxt('../HydroInfo_100/simulations/Soln' + str(soln) + '/HydroInfo_100_thinned_proc' + \
                                   str(soln) + '_day' + str(day+1) + '.txt',usecols=[0,1,2,3,4])
            corr = np.corrcoef(np.transpose(simulations))
            if np.min(corr) < minCorr:
                minCorr = np.min(corr)
                argminCorr = np.unravel_index(np.argmin(corr, axis=None), corr.shape)
                minCorrDay = day + 1
                minCorrSoln = soln
                minCorrSolnName = solnNames[i]
                
            # replace all near perfect correlations with -1s to find next largest correlation 
            # and only look in the flood season
            if day > 150 and day < 270:
                corr[corr>0.999999] = -1
                if np.max(corr) > maxCorr:
                    maxCorr = np.max(corr)
                    argmaxCorr = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
                    maxCorrDay = day + 1
                    maxCorrSoln = soln
                    maxCorrSolnName = solnNames[i]
                
    # load simulation with minCorrSoln on minCorrDay
    simulations = np.loadtxt('../HydroInfo_100/simulations/Soln' + str(minCorrSoln) + '/HydroInfo_100_thinned_proc' + \
                                   str(minCorrSoln) + '_day' + str(minCorrDay) + '.txt',usecols=[0,1,2,3,4])
    # normalize simulation inputs
    for i in range(M-2):
        simulations[:,i] = (simulations[:,i] - Input_ranges[i,0]) / (Input_ranges[i,1] - Input_ranges[i,0])
    
    # find sample years
    greenYr1 = np.argmin((simulations[:,0]-0.7)**2 + (simulations[:,1]-0.85)**2)
    blueYr1 = np.argmin((simulations[:,0]-0.2)**2 + (simulations[:,1]-0.2)**2)
    
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(2,3,1)
    # make scatter plot of minimum correlation
    makeScatterPlot(ax, simulations, argminCorr, minCorrDay, greenYr1, blueYr1, 1)
    
    # load policy with above correlation structure and get its decision variables and inputs on each day
    policy = np.loadtxt('../HydroInfo_100/HydroInfo_100_thinned.resultfile')[minCorrSoln-1,0:-4] # don't read in objs and constr
    var = getFullVar(policy, M, K, N)
    ax = fig.add_subplot(2,3,2)
    makeReleasePlot(ax, var, simulations, argminCorr, minCorrDay, policy, greenYr1, blueYr1, 1, minCorrSolnName)
    ax = fig.add_subplot(2,3,3)
    makeReleasePlot(ax, var, simulations, argminCorr, minCorrDay, policy, greenYr1, blueYr1, 0, '')
    
    # load simulation with maxCorrSoln on maxCorrDay
    simulations = np.loadtxt('../HydroInfo_100/simulations/Soln' + str(maxCorrSoln) + '/HydroInfo_100_thinned_proc' + \
                                   str(maxCorrSoln) + '_day' + str(maxCorrDay) + '.txt',usecols=[0,1,2,3,4])
    # normalize simulation inputs
    for i in range(M-2):
        simulations[:,i] = (simulations[:,i] - Input_ranges[i,0]) / (Input_ranges[i,1] - Input_ranges[i,0])
    
    # find sample years
    greenYr2 = np.argmin((simulations[:,1]-0.1)**2 + (simulations[:,0]-0.7)**2)
    blueYr2 = np.argmin((simulations[:,1]-0.6)**2 + (simulations[:,0]-0.4)**2)
    
    ax = fig.add_subplot(2,3,4)
    # make scatter plot of maximum correlation
    makeScatterPlot(ax, simulations, argmaxCorr, maxCorrDay, greenYr2, blueYr2, 1)
    
    # load policy with above correlation structure and get its decision variables and inputs on each day
    policy = np.loadtxt('../HydroInfo_100/HydroInfo_100_thinned.resultfile')[maxCorrSoln-1,0:-4] # don't read in objs and constr
    var = getFullVar(policy, M, K, N)
    ax = fig.add_subplot(2,3,5)
    makeReleasePlot(ax, var, simulations, argmaxCorr, maxCorrDay, policy, greenYr2, blueYr2, 1, maxCorrSolnName)
    ax = fig.add_subplot(2,3,6)
    makeReleasePlot(ax, var, simulations, argmaxCorr, maxCorrDay, policy, greenYr2, blueYr2, 0, '')
    
    fig.set_size_inches([17,9.5])
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    fig.savefig('NegVarianceCases.pdf')
    fig.clf()
    
    return None

def makeScatterPlot(ax, simulations, argCorr, corrDay, greenYr, blueYr, x):
    # x = 0 means make the first input the x axis and the second input the y axis
    # x = 1 means make the first input the y axis and the second input the x axis
    ax.scatter(simulations[:,argCorr[x]], simulations[:,argCorr[1-x]], c=np.array([192,0,0])/256.0)
    ax.set_xlabel(columns[argCorr[x]],fontsize=16)
    ax.set_ylabel(columns[argCorr[1-x]],fontsize=16)
    ax.set_title(dates[0][corrDay-1],fontsize=18)
    ax.tick_params(axis='both',labelsize=14)
    
    ax.scatter(simulations[greenYr,argCorr[x]], simulations[greenYr,argCorr[1-x]], c='#4daf4a', zorder=500)
    ax.scatter(simulations[blueYr,argCorr[x]], simulations[blueYr,argCorr[1-x]], c='#377eb8', zorder=500)
    
    return None

def makeReleasePlot(ax, var, simulations, argCorr, corrDay, policy, highYr, lowYr, x, title):
    # x = 0 means make the first input the x axis and the second input the y axis
    # x = 1 means make the first input the y axis and the second input the x axis
    
    normInputs = np.zeros([102,M]) # 100 values along first input; all others constant
    # find value of sin() and cos() inputs on minCorrDay
    normInputs[:,5] = np.sin(2*math.pi*corrDay/365 - policy[-2])
    normInputs[:,6] = np.sin(2*math.pi*corrDay/365 - policy[-1])
    
    # plot prescribed Hoa Binh release vs. first input on Julian day of minimum correlation
    # of highYr and lowYr while holding all other values constant at values from that day and year
    years = [highYr, lowYr]
    colors=['#4daf4a','#377eb8']
    for k in range(len(years)):
        for i in range(M-2): # all inputs but sin() and cos() functions
            # find upper and lower quartiles to use as inputs
            normInputs[:,i] = (simulations[years[k],i])
                
        # vary input 1 from min to max
        normInputs[0:-1,argCorr[x]] = np.arange(0,1.001,0.01)
        # normalized input 1 in highYr or lowYr
        normInputs[-1,argCorr[x]] = (simulations[years[k],argCorr[x]])
        u = calcReleases(M, K, N, var, normInputs)
            
        ax.plot(normInputs[0:-1,argCorr[x]],u[0:-1,1], c='k')
        for j in range(2):
            ax.scatter(normInputs[-1,argCorr[x]], u[-1,1], c=colors[k], zorder=500)
            
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel(columns[argCorr[x]],fontsize=16)
    ax.set_ylabel(r'$u_t^{HB}$',fontsize=16)
    ax.set_title(title,fontsize=16)
    
    return None

showNegVarianceCases()