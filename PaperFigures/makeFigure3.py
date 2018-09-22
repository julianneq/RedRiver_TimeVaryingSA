import numpy as np
import scipy.stats as ss
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

cmap = matplotlib.cm.get_cmap('plasma')

def makeFigure3():
    # generate positively correlated values for (x1,x2) from a Gaussian copula
    x1, x2 = generateXs(0.75)
    
    # pick unlikely combos (one high, one low)
    x1_indices = np.where(x1<0.2)[0]
    blueYr = x1_indices[np.argmax(x2[x1_indices])]
    x2_indices = np.where(x2<0.2)[0]
    redYr = x2_indices[np.argmax(x1[x2_indices])]
    
    sns.set_style("dark")
    fig = plt.figure()
    ax = fig.add_subplot(2,3,1)
    makeScatterPlot(ax, x1, x2, r'$(x_t)_a$', r'$(x_t)_b$', blueYr, redYr)
    
    ax = fig.add_subplot(2,3,2)
    makeReleasePlot(ax, r'$(x_t)_a$', r'$u_t$', [x1[blueYr],x2[blueYr]], [x1[redYr],x2[redYr]], 1)
    
    ax = fig.add_subplot(2,3,3)
    makeReleasePlot(ax, r'$(x_t)_b$', r'$u_t$', [x1[blueYr],x2[blueYr]], [x1[redYr],x2[redYr]], 0)
    
    # generate negatively correlated values for (x1,x2) from a Gaussian copula
    x1, x2 = generateXs(-0.75)
    
    # pick unlikely combos (both high, both low)
    x1_indices = np.where(x1>0.8)[0]
    redYr = x1_indices[np.argmax(x2[x1_indices])]
    x2_indices = np.where(x2<0.2)[0]
    blueYr = x2_indices[np.argmin(x1[x2_indices])]
    
    ax = fig.add_subplot(2,3,4)
    makeScatterPlot(ax, x1, x2, r'$(x_t)_a$', r'$(x_t)_b$', blueYr, redYr)
    
    ax = fig.add_subplot(2,3,5)
    makeReleasePlot(ax, r'$(x_t)_a$', r'$u_t$', [x1[blueYr],x2[blueYr]], [x1[redYr],x2[redYr]], 1)
    
    ax = fig.add_subplot(2,3,6)
    makeReleasePlot(ax, r'$(x_t)_b$', r'$u_t$', [x1[blueYr],x2[blueYr]], [x1[redYr],x2[redYr]], 0)
    
    sm = matplotlib.cm.ScalarMappable(cmap=cmap)
    sm.set_array([0,1])
    fig.subplots_adjust(hspace=0.3,wspace=0.3,right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(axis='both',labelsize=14)
    cbar.ax.set_ylabel(r'$(x_t)_a$' + ' or ' + r'$(x_t)_b$',fontsize=16)
    fig.set_size_inches([14.25,7.25])
    fig.savefig('Figure3.pdf')
    fig.clf()
    
    return None

def generateXs(corr):
    mu = np.array([0,0])
    cov = np.array([[1,corr],[corr,1]])
    dist = ss.multivariate_normal(mu, cov)
    rvs = dist.rvs(1000)
    x1 = ss.norm.cdf(rvs[:,0])
    x2 = ss.norm.cdf(rvs[:,1])
    
    return x1, x2

def makeScatterPlot(ax, x1, x2, xlabel, ylabel, blueYr, redYr):
    ax.scatter(x1, x2, c='#fc8d62')
    ax.scatter(x1[blueYr], x2[blueYr], c='#2e75b6', zorder=500)
    ax.scatter(x1[redYr], x2[redYr], c='#c00000', zorder=500)
               
    ax.set_xlabel(xlabel,fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    
    return None

def makeReleasePlot(ax, xlabel, ylabel, blueYr, redYr, index):
    x1 = np.arange(0.0,1.001,0.01)
    discrete_x2 = np.arange(0.0,1.001,0.05)
    
    # example centers for each input (assume radius = 1 for both inputs and n=1 RBF)
    if index == 1:
        c1 = 0.5
        c2 = 0.6
    else:
        c1 = 0.6
        c2 = 0.5
    
    for j in range(len(discrete_x2)):
        x2 = np.array([discrete_x2[j]]*len(x1))
        u = np.exp(-(x1-c1)**2 - (x2-c2)**2)
        ax.plot(x1,u,c=cmap(discrete_x2[j]))
        
    x2 = np.array([redYr[index]]*len(x1))
    u = np.exp(-(x1-c1)**2 -(x2-c2)**2)
    ax.plot(x1, u, c=cmap(redYr[index]))
    ax.scatter(redYr[1-index], np.exp(-(redYr[1-index]-c1)**2 -(redYr[index]-c2)**2), c='#c00000', zorder=500)
    
    x2 = np.array([blueYr[index]]*len(x1))
    u = np.exp(-(x1-c1)**2 -(x2-c2)**2)
    ax.plot(x1, u, c=cmap(blueYr[index]))
    ax.scatter(blueYr[1-index], np.exp(-(blueYr[1-index]-c1)**2 -(blueYr[index]-c2)**2), c='#2e75b6', zorder=500)
        
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlim([0,1])
    
    return None

makeFigure3()