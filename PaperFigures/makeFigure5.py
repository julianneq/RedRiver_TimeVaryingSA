import numpy as np
from netCDF4 import Dataset
import seaborn.apionly as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

class Formulation:
    def __init__(self):
        self.name = None
        self.reference = None
        self.bestFlood = None
        self.bestHydro = None
        self.bestDeficit = None
        self.compromise = None

def makeFigure5():
        
    EMODPS = getFormulations('HydroInfo_100', 'Compromise')
    GL = getFormulations('guidelines_100', 'Hydro')
    
    makeTimeFig(EMODPS.compromise, GL.bestHydro, 'Compromise', 'Figure5ab.pdf')
    makeStorageFig(EMODPS.compromise, GL.bestHydro, 'Compromise', 'Figure5cd.pdf')

    return None

def makeTimeFig(EMODPS, GL, preference, figName):
    EMODPSprobs = np.log10(getTimeProbs(EMODPS[1]))
    GLprobs = np.log10(getTimeProbs(GL[1]))
    a = EMODPSprobs[EMODPSprobs > -np.inf]
    b = GLprobs[GLprobs > -np.inf]
    
    tickMin = min(np.min(a), np.min(b))
    tickMax = max(np.max(a), np.max(b))
    
    sns.set_style("dark")
    ymax = 15
    ymin = 0
    
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    sm = ax.imshow(EMODPSprobs, cmap='RdYlBu_r',origin="upper",norm=mpl.colors.Normalize(vmin=tickMin, vmax=tickMax))
    ax.set_xticks([45,137,229,319])
    ax.set_xticklabels(['Jun','Sep','Dec','Mar'],fontsize=18)
    ax.set_yticks(np.arange(0,366+366/3,366/3))
    ax.set_yticklabels(np.arange(ymax,ymin-5,-5),fontsize=18)
    ax.set_ylabel(r'$z^{HN} (m)$',fontsize=18)
    ax.set_ylim([365,0])
    ax.set_xlim([0,365])
    ax.set_title('EMODPS ' + preference + ' Policy',fontsize=18)
    alarm, = ax.plot([0,365],[(1-11.25/15.0)*365.0,(1-11.25/15.0)*365.0],linestyle='--',c='k') # second alarm
    dikeHeight, = ax.plot([0,365],[(1-13.4/15.0)*365.0,(1-13.4/15.0)*365.0],linewidth=2,c='k') # dike height
    
    ax = fig.add_subplot(1,2,2)
    sm = ax.imshow(GLprobs, cmap='RdYlBu_r',origin="upper",norm=mpl.colors.Normalize(vmin=tickMin, vmax=tickMax))
    ax.set_xticks([45,137,229,319])
    ax.set_xticklabels(['Jun','Sep','Dec','Mar'],fontsize=18)
    ax.set_yticks(np.arange(0,366+366/3,366/3))
    ax.tick_params(axis='y',which='both',labelleft='off')
    ax.set_ylim([365,0])
    ax.set_xlim([0,365])
    ax.set_title('Guidelines Policy',fontsize=18)
    alarm, = ax.plot([0,365],[(1-11.25/15.0)*365.0,(1-11.25/15.0)*365.0],linestyle='--',c='k') # second alarm
    dikeHeight, = ax.plot([0,365],[(1-13.4/15.0)*365.0,(1-13.4/15.0)*365.0],linewidth=2,c='k') # dike height
    
    fig.subplots_adjust(right=0.8, bottom=0.2)
    fig.legend([alarm, dikeHeight],['Alarm Level', 'Dike Height'], \
        loc='lower center', ncol=2, fontsize=18, frameon=True)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(-4,0,1))
    cbar.ax.set_ylabel('Probability Density',fontsize=18)
    cbar.ax.set_yticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'],fontsize=16)
    fig.set_size_inches([11.7125, 5.975])
    fig.savefig(figName)
    fig.clf()
    
    return None

def makeStorageFig(EMODPS, GL, preference, figName):
    EMODPSprobs = np.log10(getStorageProbs(EMODPS[0], EMODPS[1]))
    GLprobs = np.log10(getStorageProbs(GL[0], GL[1]))
    a = EMODPSprobs[EMODPSprobs > -np.inf]
    b = GLprobs[GLprobs > -np.inf]
    
    tickMin = min(np.min(a), np.min(b))
    tickMax = max(np.max(a), np.max(b))
    
    sns.set_style("dark")
    ymax = 15
    ymin = 0
    xmin = 5
    xmax = 30
    
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    sm = ax.imshow(EMODPSprobs, cmap='RdYlBu_r',origin="upper",norm=mpl.colors.Normalize(vmin=tickMin, vmax=tickMax))
    ax.set_xticks(np.arange(0,100+100/5,100/5))
    ax.set_xticklabels(np.arange(xmin, xmax+5, 5),fontsize=18)
    ax.set_xlabel(r'$s^{TOT} (km^3\!)$',fontsize=18)
    ax.set_yticks(np.arange(0,100+100/3,100/3))
    ax.set_yticklabels(np.arange(ymax,ymin-5,-5),fontsize=18)
    ax.set_ylabel(r'$z^{HN} (m)$',fontsize=18)
    ax.set_ylim([100,0])
    ax.set_xlim([0,100])
    ax.set_title('EMODPS ' + preference + ' Policy',fontsize=18)
    alarm, = ax.plot([0,100],[(1-11.25/15.0)*100.0,(1-11.25/15.0)*100.0],linestyle='--',c='k') # second alarm
    dikeHeight, = ax.plot([0,100],[(1-13.4/15.0)*100.0,(1-13.4/15.0)*100.0],linewidth=2,c='k') # dike height

    
    ax = fig.add_subplot(1,2,2)
    sm = ax.imshow(GLprobs, cmap='RdYlBu_r',origin="upper",norm=mpl.colors.Normalize(vmin=tickMin, vmax=tickMax))
    ax.set_xticks(np.arange(0,100+100/5,100/5))
    ax.set_xticklabels(np.arange(xmin, xmax+5, 5),fontsize=18)
    ax.set_xlabel(r'$s^{TOT} (km^3\!)$',fontsize=18)
    ax.set_yticks(np.arange(0,100+100/3,100/3))
    ax.tick_params(axis='y',which='both',labelleft='off')
    ax.set_ylim([100,0])
    ax.set_xlim([0,100])
    ax.set_title('Guidelines Policy',fontsize=18)
    alarm, = ax.plot([0,100],[(1-11.25/15.0)*100.0,(1-11.25/15.0)*100.0],linestyle='--',c='k') # second alarm
    dikeHeight, = ax.plot([0,100],[(1-13.4/15.0)*100.0,(1-13.4/15.0)*100.0],linewidth=2,c='k') # dike height
    
    fig.subplots_adjust(right=0.8, bottom=0.2)
    fig.legend([alarm, dikeHeight],['Alarm Level', 'Dike Height'], \
        loc='lower center', ncol=2, fontsize=18, frameon=True)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(-7,-2,1))
    cbar.ax.set_ylabel('Probability Density',fontsize=18)
    cbar.ax.set_yticklabels([r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$'],fontsize=16)
    fig.set_size_inches([11.7125, 5.975])
    fig.savefig(figName)
    fig.clf()
    
    return None

def getFormulations(name, soln):
    formulation = Formulation()
    formulation.name = name
    
    formulation.reference = np.loadtxt('./../' + name + '/' + name + '_thinned.reference')
    
    if soln == 'Hydro':
        formulation.bestHydro = loadData('./../' + name + '/' + name + '_thinned_proc' + \
            str(np.argmin(formulation.reference[:,0])+1) + '_re-eval_1x100000.nc')
    elif soln == 'Deficit':
        formulation.bestDeficit = loadData('./../' + name + '/' + name + '_thinned_proc' + \
            str(np.argmin(formulation.reference[:,1])+1) + '_re-eval_1x100000.nc')
    elif soln == 'Flood':
        formulation.bestFlood = loadData('./../' + name + '/' + name + '_thinned_proc' + \
            str(np.argmin(formulation.reference[:,2])+1) + '_re-eval_1x100000.nc')
    elif soln == 'Compromise':
        compIndex = findCompromise(formulation.reference[:,0:3],1)
        formulation.compromise = loadData('./../' + name + '/' + name + '_thinned_proc' + \
            str(compIndex+1) + '_re-eval_1x100000.nc')
        
    return formulation

def findCompromise(refSet, deficitIndex):
    # normalize objectives for calculation of compromise solution
    nobjs = np.shape(refSet)[1]
    normObjs = np.zeros([np.shape(refSet)[0],nobjs])
    for i in range(np.shape(refSet)[0]):
        for j in range(nobjs):
            # take the square root of the deficit so it's less skewed
            if j == deficitIndex:
                normObjs[i,j] = (np.sqrt(refSet[i,j])-np.mean(np.sqrt(refSet[:,j])))/np.std(np.sqrt(refSet[:,j]))
            else:
                normObjs[i,j] = (refSet[i,j]-np.mean(refSet[:,j]))/np.std(refSet[:,j])
    
    # find comprommise solution (solution closest to ideal point)
    dists = np.zeros(np.shape(refSet)[0])
    for i in range(len(dists)):
        for j in range(nobjs):
            dists[i] = dists[i] + (normObjs[i,j]-np.min(normObjs[:,j]))**2
            
    compromise = np.argmin(dists)
    
    return compromise
    
def loadData(file):
    dataset = Dataset(file)
    sTOT = dataset.variables['sTOT'][:]
    hLev = dataset.variables['hLev'][:]
        
    return [sTOT, hLev]
    
def getTimeProbs(data):
    probMatrix = np.zeros([366,365])
    ymax = 15.0
    ymin = 0.0
    step = (ymax-ymin)/366.0
    for i in range(np.shape(probMatrix)[0]):
        for j in range(np.shape(probMatrix)[1]):
            probMatrix[i,j] = ((data[:,j] < ymax-step*i) & (data[:,j] >= ymax-step*(i+1))).sum()/100000.0
    
    return probMatrix
    
def getStorageProbs(s, h):
    probMatrix = np.zeros([100,100])
    ymax = 15.0
    ymin = 0.0
    xmax = 3E10
    xmin = 0.5E10
    yStep = (ymax-ymin)/np.shape(probMatrix)[0]
    xStep = (xmax-xmin)/np.shape(probMatrix)[1]
    for i in range(np.shape(s)[0]):
        for j in range(np.shape(s)[1]):
            # figure out which "box" the simulated s and h are in
            row = int(np.floor((ymax-h[i,j])/yStep))
            col = int(np.ceil((s[i,j]-xmin)/xStep))
            if row < np.shape(probMatrix)[0] and col < np.shape(probMatrix)[1]:
                probMatrix[row,col] = probMatrix[row,col] + 1
                            
    probMatrix = probMatrix/(np.shape(s)[0]*np.shape(s)[1])
    
    return probMatrix
    
makeFigure5()