import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class Formulation:
    def __init__(self):
        self.name = None
        self.reference = None
        self.bestFlood = None
        self.bestHydro = None
        self.bestDeficit = None
        self.compromise = None

def getFormulations(name, indices):
    formulation = Formulation()
    formulation.name = name
    formulation.reference = np.loadtxt('./../' + name + '/' + name +  \
                                           '_thinned.reference')
    formulation.reeval_1000 = np.loadtxt('./../' + name + '/' + name +  \
                                           '_thinned_re-eval_1x1000.obj')
    
    formulation.reeval_1000[:,[0,1,3]] = formulation.reference
    
    formulation.reference = formulation.reeval_1000[:,indices]
    
    formulation.bestHydro = np.argmin(formulation.reference[:,0])
    formulation.bestDeficit = np.argmin(formulation.reference[:,1])
    formulation.bestFlood = np.argmin(formulation.reference[:,2])
    
    compIndex = findCompromise(formulation.reference,1)
    formulation.compromise = compIndex
        
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
    
    # find compromise solution (solution closest to ideal point)
    dists = np.zeros(np.shape(refSet)[0])
    for i in range(len(dists)):
        for j in range(nobjs):
            dists[i] = dists[i] + (normObjs[i,j]-np.min(normObjs[:,j]))**2
            
    compromise = np.argmin(dists)
    
    return compromise

def makeFigureS3():
    # indices of re-evaluation objectives to plot
    # order is Jhydro_WP1, Jdef2_WP1, JmaxDef_WP1, Jflood_100, Jflood_500, Jhydro_EV, Jdef2_EV, JmaxDef_EV
    indices = [[0,1,4], [5,6,3], [0,2,3], [5,7,3]]
    xlims = [[1.7,4.8],[-0.1,3.3],[-0.1,3.3],[-0.1,3.3]]
    ylims = [[34,47],[47,57],[34,47],[47,57]]
    xlabels = [r'$J_{Flood,500}$ (m above 11.25 m)', r'$J_{Flood,100}$ (m above 11.25 m)',\
               r'$J_{Flood,100}$ (m above 11.25 m)', r'$J_{Flood,100}$ (m above 11.25 m)']
    ylabels = [r'$J_{Hydro,WP1}$ (Gwh/day)',r'$J_{Hydro,EV}$ (Gwh/day)',r'$J_{Hydro,WP1}$ (Gwh/day)',r'$J_{Hydro,EV}$ (Gwh/day)']
    titles = [r'$J_{Deficit^2, WP1}$ ' + r'$(m^3/s)^2$', r'$J_{Deficit^2, EV}$ ' + r'$(m^3/s)^2$', \
              r'$J_{Max Deficit, WP1}$ ' + r'$(m^3/s)^2$', r'$J_{Max Deficit, EV}$ ' + r'$(m^3/s)^2$']
    figOrder = [4,3,1,2]
    
    fig = plt.figure()
    for i in range(len(indices)):
        guidelines = getFormulations('guidelines_100',indices[i])
        optimized = getFormulations('HydroInfo_100',indices[i])
        
        optimized.reference[:,0] = -optimized.reference[:,0]
        guidelines.reference[:,0] = -guidelines.reference[:,0]
        
        idealPoint = np.zeros(3)
        idealPoint[0] = np.max([np.max(optimized.reference[:,0]),np.max(guidelines.reference[:,0])])
        idealPoint[1] = np.min([np.min(optimized.reference[:,1]),np.min(guidelines.reference[:,1])])
        idealPoint[2] = np.min([np.min(optimized.reference[:,2]),np.min(guidelines.reference[:,2])])
        
        worstPoint = np.zeros(3)
        worstPoint[0] = np.min([np.min(optimized.reference[:,0]),np.min(guidelines.reference[:,0])])
        worstPoint[1] = np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])])
        worstPoint[2] = np.max([np.max(optimized.reference[:,2]),np.max(guidelines.reference[:,2])])
        
        sns.set()
        
        #2-D Figure multi color
        ax = fig.add_subplot(2,2,figOrder[i])
        pts1 = ax.scatter(optimized.reference[:,2], optimized.reference[:,0], \
            s=200*optimized.reference[:,1]/np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]), \
            color = '#377eb8')
        pts1_hydro = ax.scatter(optimized.reference[optimized.bestHydro,2], optimized.reference[optimized.bestHydro,0],\
            s=200*optimized.reference[optimized.bestHydro,1]/np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]), \
            facecolor = '#377eb8', edgecolor = 'k', linewidth=2)
        pts1_flood = ax.scatter(optimized.reference[optimized.bestFlood,2], optimized.reference[optimized.bestFlood,0],\
            s=200*optimized.reference[optimized.bestFlood,1]/np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]), \
            facecolor = '#377eb8', edgecolor = 'k', linewidth=2)
        pts1_def = ax.scatter(optimized.reference[optimized.bestDeficit,2], optimized.reference[optimized.bestDeficit,0],\
            s=200*optimized.reference[optimized.bestDeficit,1]/np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]), \
            facecolor = '#377eb8', edgecolor = 'k', linewidth=2)
        pts1_comp = ax.scatter(optimized.reference[optimized.compromise,2], optimized.reference[optimized.compromise,0],\
            s=200*optimized.reference[optimized.compromise,1]/np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]), \
            facecolor = '#377eb8', edgecolor = 'k', linewidth=2)
            
        pts2 = ax.scatter(guidelines.reference[:,2], guidelines.reference[:,0], \
            s=200*guidelines.reference[:,1]/np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]), \
            color = '#e41a1c')
        pts2_hydro = ax.scatter(guidelines.reference[guidelines.bestHydro,2], guidelines.reference[guidelines.bestHydro,0], \
            s=200*guidelines.reference[guidelines.bestHydro,1]/np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]), \
            facecolor = '#e41a1c', edgecolor = 'k', linewidth=2)
              
        l1 = ax.scatter([],[], s=200*np.min([np.min(optimized.reference[:,1]),np.min(guidelines.reference[:,1])])/ \
            np.max([np.max(optimized.reference[:,1]),np.max(guidelines.reference[:,1])]),  color='k')
        l2 = ax.scatter([],[],s=200, color='k')
        
        ax.set_xlabel(xlabels[i], fontsize=16)
        ax.set_xlim(xlims[i])
        ax.set_ylabel(ylabels[i], fontsize=16)
        ax.tick_params(axis='both',labelsize=14)
        dikeLine, = ax.plot([2.15,2.15],ylims[i],c='#4daf4a',linewidth=2)
        ax.set_ylim(ylims[i])
        legend1 = ax.legend([l1, l2], [str(round(np.min([np.min(optimized.reference[:,1]),np.min(guidelines.reference[:,1])]),1)), \
            str(round(np.max([np.max(optimized.reference[:,1]),np.min(guidelines.reference[:,1])]),1))], \
            scatterpoints=1, title=titles[i], fontsize=14, loc='upper right', frameon=False)
        plt.setp(legend1.get_title(),fontsize=14)
        
    # Put a legend below current axis
    fig.subplots_adjust(bottom=0.15)
    legend2 = fig.legend([pts1, pts2, dikeLine], ['Optimized Policies', 'Guidelines', 'Dike Height'], \
                            scatterpoints=1, loc= 'lower center', ncol=3, frameon=True, fontsize=14)
    fig.set_size_inches([19.2,9.5])
    plt.savefig('FigureS3.pdf')
    plt.clf()

    return None

makeFigureS3()