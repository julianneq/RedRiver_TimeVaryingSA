import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from utils import getFormulations, getSolns, getSolns_r #, getSolns_u, 

sns.set_style("dark")

# plotting features
inputs = ['sSL','sHB','sTQ','sTB','HNfcst']
reservoir = inputs[1] # HoaBinh
colors = ['#fb9a99','#e31a1c','#33a02c','#6a3d9a','#1f78b4','#ff7f00']
pSL = plt.Rectangle((0,0), 1, 1, fc=colors[0], edgecolor='none')
pHB = plt.Rectangle((0,0), 1, 1, fc=colors[1], edgecolor='none')
pTQ = plt.Rectangle((0,0), 1, 1, fc=colors[2], edgecolor='none')
pTB = plt.Rectangle((0,0), 1, 1, fc=colors[3], edgecolor='none')
pHNfcst = plt.Rectangle((0,0), 1, 1, fc=colors[4], edgecolor='none')
pInteract = plt.Rectangle((0,0), 1, 1, fc=colors[5], edgecolor='none')

def makeFigure7_S8():
    # get data for HydroInfo_100 formulation, including:
    # FloodYr = year of 100-yr flood
    # DeficitYr = year of 100-yr squared deficit
    # HydroYr = year of 100-yr hydropower production
    HydroInfo_100 = getFormulations('HydroInfo_100')
    guidelines_100 = getFormulations('guidelines_100')
    
    # compare release sensitivities across solutions (rows) and events (columns)
    # using analytical and numerical estimates of prescribed and true releases
    # with a perturbation of 0.005 (Figure 7)
    CompAnalytical = getSolns(HydroInfo_100, HydroInfo_100.compromise, '0')
    #CompNumericalU = getSolns_u(HydroInfo_100, HydroInfo_100.compromise, '0.005')
    CompNumericalR = getSolns_r(HydroInfo_100, HydroInfo_100.compromise, '0.005')
    GL = getSolns_r(guidelines_100, guidelines_100.bestHydro, '0.005')
    #solns = [CompAnalytical, CompNumericalU, CompNumericalR, GL]
    solns = [CompAnalytical, CompNumericalR, GL]
    ylabels = ['Analytical\nSensitivity of\nCompromise ' + r'$u_t^{HB}$',\
               #'Numerical\nSensitivity of\nCompromise ' + r'$u_t^{HB}$',\
               'Numerical\nSensitivity of\nCompromise ' + r'$r_{t+1}^{HB}$',\
               'Numerical\nSensitivity of\nGuidelines ' + r'$r_{t+1}^{HB}$']
    #makePlot(solns, reservoir, ylabels, 'Figure7.pdf', 12.5)
    makePlot(solns, reservoir, ylabels, 'Figure7.pdf', 9.5)
    
    # with a perturbation of 0.05 (Figure S8)
    CompAnalytical = getSolns(HydroInfo_100, HydroInfo_100.compromise, '0')
    #CompNumericalU = getSolns_u(HydroInfo_100, HydroInfo_100.compromise, '0.05')
    CompNumericalR = getSolns_r(HydroInfo_100, HydroInfo_100.compromise, '0.05')
    GL = getSolns_r(guidelines_100, guidelines_100.bestHydro, '0.05')
    #solns = [CompAnalytical, CompNumericalU, CompNumericalR, GL]
    solns = [CompAnalytical, CompNumericalR, GL]
    #makePlot(solns, reservoir, ylabels, 'FigureS8.pdf', 12.5)
    makePlot(solns, reservoir, ylabels, 'FigureS8.pdf', 9.5)
    
    return None

def makePlot(solns, reservoir, ylabels, figureName, height):
        
    fig = plt.figure()
    ymaxs = np.ones([len(solns)])
    ymins = np.zeros([len(solns)])
    axes = []
    titles = ['Year of WP1 Flood','Year of WP1 Hydro','Year of WP1 Deficit']
    for i in range(len(solns)):
        years = [solns[i].FloodYr, solns[i].HydroYr, solns[i].DeficitYr]
        SIs = solns[i].uHB_SI
            
        for j in range(len(years)):
            ax = fig.add_subplot(len(solns),len(years),i*len(years)+j+1)
            y1 = np.zeros([365])
            for k in range(len(inputs)): # five 1st order SIs
                y2 = np.zeros([365])
                posIndices = np.where(np.sum(SIs[years[j]*365:(years[j]+1)*365,:],1)>0)
                y2[posIndices] = np.sum(SIs[years[j]*365:(years[j]+1)*365,0:(k+1)],1)[posIndices]/ \
                        np.sum(SIs[years[j]*365:(years[j]+1)*365,:],1)[posIndices]
                if i != len(solns)-1: # not guidelines policy
                    ax.plot(range(0,365),y2,c='None')
                    ax.fill_between(range(0,365), y1, y2, where=y2>y1, color=colors[k])
                    ymaxs[i] = max(ymaxs[i],np.max(y2))
                else:
                    ax.plot(range(0,10),y2[0:10],c='None')
                    ax.fill_between(range(0,10),y1[0:10],y2[0:10],where=y2[0:10]>y1[0:10],color=colors[k])
                    if reservoir != 'sTB':
                        ax.plot(range(45,138),y2[45:138],c='None')
                        ax.fill_between(range(45,138),y1[45:138],y2[45:138],where=y2[45:138]>y1[45:138],color=colors[k])
                        ymaxs[i] = max(ymaxs[i],np.max([np.max(y2[0:10]),np.max(y2[45:138]),np.max(y2[209:365])]))
                    else:
                        ymaxs[i] = max(ymaxs[i],np.max([np.max(y2[0:10]),np.max(y2[209:365])]))
                        
                    ax.plot(range(208,365),y2[208:365],c='None')
                    ax.fill_between(range(208,365),y1[208:365],y2[208:365],where=y2[208:365]>y1[208:365],color=colors[k])
                
                y1 = y2
                
            y2 = np.ones([365])
            ZeroIndices = np.where(y1==0)
            y2[ZeroIndices] = 0
            negIndices = np.where(np.sum(SIs[years[j]*365:(years[j]+1)*365,len(inputs)::],1)<0)
            y2[negIndices] = np.sum(SIs[years[j]*365:(years[j]+1)*365,len(inputs)::],1)[negIndices]/ \
                np.sum(SIs[years[j]*365:(years[j]+1)*365,:],1)[negIndices]
            if i != len(solns)-1:
                ax.fill_between(range(0,365), y1, y2, where=y1<y2, color=colors[-1])
                ax.fill_between(range(0,365), y2, 0, where=y1>y2, color=colors[-1])
                ymaxs[i] = max(ymaxs[i], np.max(y2))
                ymins[i] = min(ymins[i], np.min(y2))
            else:
                ax.fill_between(range(0,10), y1[0:10], y2[0:10], where=y1[0:10]<y2[0:10], color=colors[-1]) # interactions
                ax.fill_between(range(0,10), y2[0:10], 0, where=y1[0:10]>y2[0:10], color=colors[-1]) # interactions
                if reservoir != 'sTB':
                    ax.fill_between(range(45,138), y1[45:138], y2[45:138], where=y1[45:138]<y2[45:138], color=colors[-1]) # interactions
                    ax.fill_between(range(45,138), y2[45:138], 0, where=y1[45:138]>y2[45:138], color=colors[-1]) # interactions
                    ymaxs[i] = max(ymaxs[i],np.max([np.max(y2[0:10]),np.max(y2[45:138]),np.max(y2[209:365])]))
                    ymins[i] = min(ymins[i],np.min([np.min(y2[0:10]),np.min(y2[45:138]),np.min(y2[209:365])]))
                else:
                    ymaxs[i] = max(ymaxs[i],np.max([np.max(y2[0:10]),np.max(y2[209:365])]))
                    ymins[i] = max(ymins[i],np.min([np.min(y2[0:10]),np.min(y2[209:365])]))
                    
                ax.fill_between(range(208,365), y1[208:365], y2[208:365], where=y1[208:365]<y2[208:365], color=colors[-1]) # interactions
                ax.fill_between(range(208,365), y2[208:365], 0, where=y1[208:365]>y2[208:365], color=colors[-1]) # interactions
                
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
            axes.append(ax)
            
    for i in range(len(axes)):
        axes[i].set_ylim([ymins[int(np.floor(i/3.0))],ymaxs[int(np.floor(i/3.0))]])
                
    fig.text(0.95, 0.5, 'Portion of Variance', va='center', rotation='vertical', fontsize=18)
    fig.subplots_adjust(bottom=0.15,hspace=0.3)
    plt.figlegend([pSL,pHB,pTQ,pTB,pHNfcst,pInteract],\
                  [r'$s_t^{SL}$',r'$s_t^{HB}$',r'$s_t^{TQ}$',r'$s_t^{TB}$',r'$\tilde{z}_{t+2}^{HN}$','Interactions'],\
                  loc='lower center', ncol=3, fontsize=16, frameon=True)
    fig.suptitle('Sensitivity of ' + r'$u_t^{HB}$' + ' vs. ' + r'$r_{t+1}^{HB}$', fontsize=18)
    fig.set_size_inches([10,height])
    fig.savefig(figureName)
    fig.clf()
    
    return None

makeFigure7_S8()
