import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils import getFormulations

sns.set_style("dark")

class Soln():
    def __init__(self):
        self.name = None
        self.solnNo = None
        self.FloodYr = None
        self.DeficitYr = None
        self.HydroYr = None
        self.allInputs = None
        self.allOutputs = None
        self.allObjs = None
        self.xHNfcst = None
        
M = 5 # number of inputs: storage at the 4 reservoirs, forecasted water level at Hanoi, \
# ignore sin(2*pi*t/365 - phi1) and cos(2*pi*t/365 - phi2) b/c their centers and radii are fixed
N = 12 # number of RBFs
K = 4 # number of outputs: release at the 4 reservoirs

inputs = [r'$s_t^{SL}$',r'$s_t^{HB}$',r'$s_t^{TQ}$',r'$s_t^{TB}$',r'$\tilde{z}_{t+2}^{HN}$']
inputNames = ['sSL','sHB','sTQ','sTB','HNfcst']
outputNames = ['uSL','uHB','uTQ','uTB']
inputRanges = np.array([[2223600000, 3215000000, 402300000, 402300000, 0], \
                        [12457000000, 10890000000, 2481000000, 3643000000, 20]])
outputRanges = np.array([[0, 0, 0, 0],[40002, 35784, 13551, 3650]])
objNames = ['Flood','DefSq','Hydro']

colors = ['#fb9a99','#e31a1c','#33a02c','#6a3d9a','#1f78b4','#ff7f00']
pSL = plt.Rectangle((0,0), 1, 1, fc=colors[0], edgecolor='none')
pHB = plt.Rectangle((0,0), 1, 1, fc=colors[1], edgecolor='none')
pTQ = plt.Rectangle((0,0), 1, 1, fc=colors[2], edgecolor='none')
pTB = plt.Rectangle((0,0), 1, 1, fc=colors[3], edgecolor='none')
pHNfcst = plt.Rectangle((0,0), 1, 1, fc=colors[4], edgecolor='none')
        
def makeFigureS3_S6():
    guidelines_100 = getFormulations('guidelines_100')
    HydroInfo_100 = getFormulations('HydroInfo_100')
    
    diffs = [False, True]
    figNames = ['FigureS3.pdf','FigureS6.pdf']
    for i, diff in enumerate(diffs):
        BestFloodSoln = getSolns(HydroInfo_100.bestFlood, 'Flood', '0.005', inputNames, inputRanges, \
                                 outputNames, outputRanges, objNames, diff)
        BestDefSoln = getSolns(HydroInfo_100.bestDeficit, 'Deficit', '0.005', inputNames, inputRanges, \
                                 outputNames, outputRanges, objNames, diff)
        BestHydroSoln = getSolns(HydroInfo_100.bestHydro, 'Hydro', '0.005', inputNames, inputRanges, \
                                 outputNames, outputRanges, objNames, diff)
        CompromiseSoln = getSolns(HydroInfo_100.compromise, 'Compromise', '0.005', inputNames, inputRanges, \
                                 outputNames, outputRanges, objNames, diff)
        Guidelines = getSolns(guidelines_100.bestHydro,'Guidelines', '0.005', inputNames, inputRanges, \
                                 outputNames, outputRanges, objNames, diff)
        solns = [BestFloodSoln, BestHydroSoln, BestDefSoln, CompromiseSoln, Guidelines]
        makePlot(solns, figNames[i])
    
    return None

def makePlot(solns, figname):
    # plot state and prescribed release trajectories across solutions (rows) and events (columns)
    fig = plt.figure()
    if figname == 'FigureS3.pdf':
        ymaxs = np.ones([5])
    else:
        ymaxs = np.zeros([5])
    ymins = np.zeros([5])
    axes = []
    titles = ['Year of WP1 Flood','Year of WP1 Hydro','Year of WP1 Deficit']
    ylabels = ['EMODPS Best\nFlood Policy','EMODPS Best\nHydro Policy','EMODPS Best\nDeficit Policy',\
               'EMODPS\nCompromise\nPolicy','Guidelines\nPolicy']
    for i in range(len(solns)):
        years = [solns[i].FloodYr, solns[i].HydroYr, solns[i].DeficitYr]
        
        for j in range(len(years)):
            ax = fig.add_subplot(len(solns),len(years),i*len(years)+j+1)
            if i != len(solns)-1: # not the guidelines
                if figname == 'FigureS3.pdf': # state trajectories
                    for k in range(len(inputs)):
                        ax.plot(range(0,365),solns[i].allInputs[years[j],:,k],c=colors[k],linewidth=2)
                else: # difference between true and prescribed releases
                    for k in range(len(outputNames)):
                        # find non-zero differences to plot on log-scale
                        ax.plot(range(0,365),solns[i].allOutputs[years[j],:,k],c=colors[k])
                        ymins[i] = min(ymins[i],np.min(solns[i].allOutputs[years[j],:,k]))
                        ymaxs[i] = max(ymaxs[i],np.max(solns[i].allOutputs[years[j],:,k]))
                
            else:
                if figname == 'FigureS3.pdf':
                    for k in range(len(inputs)):
                        ax.plot(range(0,10),solns[i].allInputs[years[j],0:10,k],c=colors[k],linewidth=2)
                        ax.plot(range(208,365),solns[i].allInputs[years[j],208:365,k],c=colors[k],linewidth=2)
                        ax.plot(range(44,138),solns[i].allInputs[years[j],44:138,k],c=colors[k],linewidth=2)
                else:
                    for k in range(len(outputNames)):
                        ax.plot(range(0,10),solns[i].allOutputs[years[j],0:10,k],c=colors[k])
                        ax.plot(range(208,365),solns[i].allOutputs[years[j],208:365,k],c=colors[k])
                        ax.plot(range(44,138),solns[i].allOutputs[years[j],44:138,k],c=colors[k])
                        ymins[i] = min(ymins[i],np.min(solns[i].allOutputs[years[j],0:10,k]))
                        ymins[i] = min(ymins[i],np.min(solns[i].allOutputs[years[j],208:365,k]))
                        ymins[i] = min(ymins[i],np.min(solns[i].allOutputs[years[j],44:138,k]))
                        ymaxs[i] = max(ymaxs[i],np.max(solns[i].allOutputs[years[j],0:10,k]))
                        ymaxs[i] = max(ymaxs[i],np.max(solns[i].allOutputs[years[j],208:365,k]))
                        ymaxs[i] = max(ymaxs[i],np.max(solns[i].allOutputs[years[j],44:138,k]))
                
            # Box for reservoirs between seasons (unregulated by guidelines)
            if i == len(solns)-1:
                if figname == 'FigureS3.pdf':
                    # Box for reservoirs between seasons (unregulated by guidelines)
                    ax.plot([10,10], [0,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([10,44], [0,0], c='k', linewidth=2, linestyle='--')
                    ax.plot([10,44], [1,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([44,44], [0,1], c='k', linewidth=2, linestyle='--')
                    
                    ax.plot([138,138], [0,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([138,208], [0,0], c='k', linewidth=2, linestyle='--')
                    ax.plot([138,208], [1,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([208,208], [0,1], c='k', linewidth=2, linestyle='--')
            
            if i != len(solns)-1:
                ax.tick_params(axis='x',labelbottom='off')
            else:
                ax.set_xticks([15,45,75,106,137,167,198,229,259,289,319,350])
                ax.set_xticklabels(['M','J','J','A','S','O','N','D','J','F','M','A'],fontsize=14)
                
            if j != 0:
                ax.tick_params(axis='y', labelleft='off')
            else:
                ax.set_ylabel(ylabels[i], fontsize=16)
                ax.tick_params(axis='y', labelsize=14)
                
            if i == 0:
                ax.set_title(titles[j], fontsize=16)
                
            ax.set_xlim([0,364])
            
            #if figname == 'FigureS6.pdf':
            #    ax.set_yscale('log')
                
            axes.append(ax)
            
    for i in range(len(axes)):
        axes[i].set_ylim([ymins[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]])
        if figname == 'FigureS6.pdf' and i>= 12: # create box for guidelines in Figure S6
            axes[i].plot([10,10], [ymins[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
            axes[i].plot([10,44], [ymins[int(np.floor(i/3.0))],ymins[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
            axes[i].plot([10,44], [ymaxs[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
            axes[i].plot([44,44], [ymins[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
            
            axes[i].plot([138,138], [ymins[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
            axes[i].plot([138,208], [ymins[int(np.floor(i/3.0))],ymins[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
            axes[i].plot([138,208], [ymaxs[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
            axes[i].plot([208,208], [ymins[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]], \
                    c='k', linewidth=2, linestyle='--')
                
    fig.text(0.02, 0.5, 'Normalized Value', va='center', rotation='vertical', fontsize=18)
    fig.subplots_adjust(bottom=0.15,hspace=0.3)
    if figname == 'FigureS3.pdf':
        fig.suptitle('State Trajectories', fontsize=18)
        plt.figlegend([pSL,pHB,pTQ,pTB,pHNfcst],inputs,\
                  loc='lower center', ncol=3, fontsize=16, frameon=True)
    else:
        fig.suptitle('True Release - Prescribed Release', fontsize=18)
        plt.figlegend([pSL,pHB,pTQ,pTB],\
                  [r'$s_t^{SL}$',r'$s_t^{HB}$',r'$s_t^{TQ}$',r'$s_t^{TB}$'],\
                  loc='lower center', ncol=2, fontsize=16, frameon=True)
    fig.set_size_inches([10,12.5])
    fig.savefig(figname)
    fig.clf()
    
    return None

def getSolns(solnNo, name, perturbation, inputNames, inputRanges, outputNames, outputRanges, objNames, diff = False):
    soln = Soln()
    soln.name = name
    soln.solnNo = solnNo
    soln.allInputs = np.zeros([1000,365,5])
    soln.allOutputs = np.zeros([1000,365,4])
    soln.allObjs = np.zeros([1000,365,3])
    doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
    for day in range(365):
        if name != 'Guidelines':
            soln.allInputs[:,day,:] = pd.read_csv('../HydroInfo_100/simulations/Soln' + \
                          str(soln.solnNo) + '/HydroInfo_100_thinned_proc' \
                          + str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', \
                          sep=' ', names=inputNames, usecols=[0,1,2,3,4])
            if diff == False:
                soln.allOutputs[:,day,:] = pd.read_csv('../HydroInfo_100/simulations/Soln' + \
                           str(soln.solnNo) + '/HydroInfo_100_thinned_proc' \
                          + str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', \
                          sep=' ', names=outputNames, usecols=[5,6,7,8])
            else:
                soln.allOutputs[:,day,:] = pd.read_csv('../HydroInfo_100/perturbations/Soln' + \
                           str(soln.solnNo) + '/r/Delta' + perturbation + '/HydroInfo_100_thinned_proc' \
                          + str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', \
                          sep=' ', names=outputNames, usecols=[5,6,7,8]) - \
                          pd.read_csv('../HydroInfo_100/perturbations/Soln' + \
                           str(soln.solnNo) + '/u/Delta' + perturbation + '/HydroInfo_100_thinned_proc' \
                          + str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', \
                          sep=' ', names=outputNames, usecols=[5,6,7,8])
            soln.allObjs[:,day,:] = pd.read_csv('../HydroInfo_100/simulations/Soln' + \
                        str(soln.solnNo) + '/HydroInfo_100_thinned_proc' \
                          + str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', \
                          sep=' ', names=objNames, usecols=[9,10,11])
        else:
            soln.allInputs[:,day,:] = pd.read_csv('../guidelines_100/perturbations/Soln' + \
                          str(soln.solnNo) + '/u/Delta' + perturbation + '/guidelines_100_day' + \
                          str(doy[day]) + '.txt', sep=' ', names=inputNames, usecols=[0,1,2,3,4])
            if diff == False:
                soln.allOutputs[:,day,:] = pd.read_csv('../guidelines_100/perturbations/Soln' + \
                           str(soln.solnNo) + '/u/Delta' + perturbation + '/guidelines_100_day' + \
                           str(doy[day]) + '.txt', sep=' ', names=outputNames, usecols=[5,6,7,8])
            else:
                soln.allOutputs[:,day,:] = pd.read_csv('../guidelines_100/perturbations/Soln' + \
                           str(soln.solnNo) + '/r/Delta' + perturbation + '/guidelines_100_day' + \
                           str(doy[day]) + '.txt', sep=' ', names=outputNames, usecols=[5,6,7,8]) - \
                           pd.read_csv('../guidelines_100/perturbations/Soln' + \
                           str(soln.solnNo) + '/u/Delta' + perturbation + '/guidelines_100_day' + \
                           str(doy[day]) + '.txt', sep=' ', names=outputNames, usecols=[5,6,7,8])
            soln.allObjs[:,day,:] = pd.read_csv('../guidelines_100/perturbations/Soln' +\
                            str(soln.solnNo) + '/u/Delta' + perturbation + '/guidelines_100_day' + \
                            str(doy[day]) + '.txt', sep=' ', names=objNames, usecols=[9,10,11])
            
    for i in range(np.shape(inputRanges)[1]):
        soln.allInputs[:,:,i] = (soln.allInputs[:,:,i] - inputRanges[0,i]) / \
            (inputRanges[1,i] - inputRanges[0,i])
            
    for i in range(np.shape(outputRanges)[1]):
        soln.allOutputs[:,:,i] = (soln.allOutputs[:,:,i] - outputRanges[0,i]) / \
            (outputRanges[1,i] - outputRanges[0,i])
            
    soln.FloodYr = np.argsort(np.max(soln.allObjs[:,:,0],1))[990]
    soln.DeficitYr = np.argsort(np.sum(soln.allObjs[:,:,1],1))[990]
    soln.HydroYr = np.argsort(np.sum(soln.allObjs[:,:,2],1))[990]
    
    return soln

makeFigureS3_S6()