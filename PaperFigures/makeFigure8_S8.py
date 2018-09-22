import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from utils import getFormulations, getSolns, getSolns_r#, getSolns_u, 

sns.set_style("dark")

# plotting features
inputs = ['sSL','sHB','sTQ','sTB','HNfcst']
colors = ['#fb9a99','#e31a1c','#33a02c','#6a3d9a','#1f78b4','#ff7f00']
pSL = plt.Rectangle((0,0), 1, 1, fc=colors[0], edgecolor='none')
pHB = plt.Rectangle((0,0), 1, 1, fc=colors[1], edgecolor='none')
pTQ = plt.Rectangle((0,0), 1, 1, fc=colors[2], edgecolor='none')
pTB = plt.Rectangle((0,0), 1, 1, fc=colors[3], edgecolor='none')
pHNfcst = plt.Rectangle((0,0), 1, 1, fc=colors[4], edgecolor='none')
pInteract = plt.Rectangle((0,0), 1, 1, fc=colors[5], edgecolor='none')

def makeFigures8_S8():
    # get data for HydroInfo_100 formulation, including:
    # FloodYr = year of 100-yr flood
    # DeficitYr = year of 100-yr squared deficit
    # HydroYr = year of 100-yr hydropower production
    HydroInfo_100 = getFormulations('HydroInfo_100')
    guidelines_100 = getFormulations('guidelines_100')
    
    # compare sensitivities across reservoirs during year of 100-yr flood
    # using analytical and numerical estimates of prescribed and true releases
    # with a perturbation of 0.005 (Figure 8)
    CompAnalytical = getSolns(HydroInfo_100, HydroInfo_100.compromise, '0')
    #CompNumericalU = getSolns_u(HydroInfo_100, HydroInfo_100.compromise, '0.005')
    CompNumericalR = getSolns_r(HydroInfo_100, HydroInfo_100.compromise, '0.005')
    GL = getSolns_r(guidelines_100, guidelines_100.bestHydro, '0.005')
    #solns = [CompAnalytical, CompNumericalU, CompNumericalR, GL]
    solns = [CompAnalytical, CompNumericalR, GL]
    titles = ['Analytical Sensitivity\nof Compromise\nPrescribed Releases',
               #'Numerical Sensitivity\nof Compromise\nPrescribed Releases',\
               'Numerical Sensitivity\nof Compromise\nTrue Releases',\
               'Numerical Sensitivity\nof Guidelines\nTrue Releases']
    ylabels = [r'$u_t^{SL}$' + ' or ' r'$r_{t+1}^{SL}$',\
               r'$u_t^{HB}$' + ' or ' r'$r_{t+1}^{HB}$',\
               r'$u_t^{TQ}$' + ' or ' r'$r_{t+1}^{TQ}$',\
               r'$u_t^{TB}$' + ' or ' r'$r_{t+1}^{TB}$']
    #makePlot(solns, titles, ylabels, 'Figure8.pdf', 13)
    makePlot(solns, titles, ylabels, 'Figure8.pdf', 10)
    
    # with a perturbation of 0.05 (Figure S8)
    CompAnalytical = getSolns(HydroInfo_100, HydroInfo_100.compromise, '0')
    #CompNumericalU = getSolns_u(HydroInfo_100, HydroInfo_100.compromise, '0.05')
    CompNumericalR = getSolns_r(HydroInfo_100, HydroInfo_100.compromise, '0.05')
    GL = getSolns_r(guidelines_100, guidelines_100.bestHydro, '0.05')
    #solns = [CompAnalytical, CompNumericalU, CompNumericalR, GL]
    solns = [CompAnalytical, CompNumericalR, GL]
    #makePlot(solns, titles, ylabels, 'FigureS8.pdf', 13)
    makePlot(solns, titles, ylabels, 'FigureS8.pdf', 10)

    return None
               
def makePlot(solns, titles, ylabels, figname, width):
    # first compare release sensitivities across reservoirs (rows) and events (columns)
    # for each solution (figure)
    
    fig = plt.figure()
    ymaxs = np.ones([len(ylabels)])
    ymins = np.zeros([len(ylabels)])
    axes = []
    for i in range(len(solns)):
        SIs = [solns[i].uSL_SI, solns[i].uHB_SI, solns[i].uTQ_SI, solns[i].uTB_SI]
        year = solns[i].FloodYr
        for j in range(len(SIs)):
            ax = fig.add_subplot(len(SIs),len(solns),j*len(solns)+i+1)
            y1 = np.zeros([365])
            for k in range(len(inputs)): # five 1st order SIs
                y2 = np.zeros([365])
                # find and plot where portion of variance is positive
                posIndices = np.where(np.sum(SIs[j][year*365:(year+1)*365,:],1)>0)
                y2[posIndices] = np.sum(SIs[j][year*365:(year+1)*365,0:(k+1)],1)[posIndices]/ \
                        np.sum(SIs[j][year*365:(year+1)*365,:],1)[posIndices]
                
                if i == len(solns) - 1: # guidelines
                    if j != len(SIs)-1: # not Thacba
                        ymaxs[j] = max(ymaxs[j],np.max([np.max(y2[0:10]),np.max(y2[45:138]),np.max(y2[209:365])]))
                    else: # Thacba
                        ymaxs[j] = max(ymaxs[j],np.max([np.max(y2[0:10]),np.max(y2[209:365])]))
                    ax.plot(range(0,10),y2[0:10],c='None')
                    ax.fill_between(range(0,10),y1[0:10],y2[0:10],where=y2[0:10]>y1[0:10],color=colors[k])
                    if j != len(SIs)-1: # not Thacba
                        ax.plot(range(45,138),y2[45:138],c='None')
                        ax.fill_between(range(45,138),y1[45:138],y2[45:138],where=y2[45:138]>y1[45:138],color=colors[k])
                    ax.plot(range(209,365),y2[209:365],c='None')
                    ax.fill_between(range(209,365),y1[209:365],y2[209:365],where=y2[209:365]>y1[209:365],color=colors[k])
                else: # not guidelines
                    ymaxs[j] = max(ymaxs[j],np.max(y2))
                    ax.plot(range(0,365),y2,c='None')
                    ax.fill_between(range(0,365), y1, y2, where=y2>y1, color=colors[k])

                y1 = y2
            
            # find and plot 0 or negative sensitivities
            y2 = np.ones([365])
            ZeroIndices = np.where(y1==0)
            y2[ZeroIndices] = 0
            negIndices = np.where(np.sum(SIs[j][year*365:(year+1)*365,len(inputs)::],1)<0)
            y2[negIndices] = np.sum(SIs[j][year*365:(year+1)*365,len(inputs)::],1)[negIndices]/ \
                np.sum(SIs[j][year*365:(year+1)*365,:],1)[negIndices]
            if i != len(solns)-1: # plot interactions for EMODPS policies
                ax.fill_between(range(0,365), y1, y2, where=y1<y2, color=colors[-1])
                ax.fill_between(range(0,365), y2, 0, where=y1>y2, color=colors[-1])
                ymaxs[j] = max(ymaxs[j], np.max(y2))
                ymins[j] = min(ymins[j], np.min(y2))
            else: # plot interactions for guidelines
                ax.fill_between(range(0,10), y1[0:10], y2[0:10], where=y1[0:10]<y2[0:10], color=colors[-1]) # interactions
                ax.fill_between(range(0,10), y2[0:10], 0, where=y1[0:10]>y2[0:10], color=colors[-1]) # interactions
                if j != len(SIs)-1:
                    ax.fill_between(range(45,138), y1[45:138], y2[45:138], where=y1[45:138]<y2[45:138], color=colors[-1]) # interactions
                    ax.fill_between(range(45,138), y2[45:138], 0, where=y1[45:138]>y2[45:138], color=colors[-1]) # interactions
                ax.fill_between(range(209,365), y1[209:365], y2[209:365], where=y1[209:365]<y2[209:365], color=colors[-1]) # interactions
                ax.fill_between(range(209,365), y2[209:365], 0, where=y1[209:365]>y2[209:365], color=colors[-1]) # interactions
                
                if j != len(SIs)-1:
                    # Box for reservoirs between seasons (unregulated)
                    ax.plot([10,10], [0,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([10,44], [0,0], c='k', linewidth=2, linestyle='--')
                    ax.plot([10,44], [1,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([44,44], [0,1], c='k', linewidth=2, linestyle='--')
                    
                    ax.plot([138,138], [0,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([138,208], [0,0], c='k', linewidth=2, linestyle='--')
                    ax.plot([138,208], [1,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([208,208], [0,1], c='k', linewidth=2, linestyle='--')
                    ymaxs[j] = max(ymaxs[j],np.max([np.max(y2[0:10]),np.max(y2[45:138]),np.max(y2[209:365])]))
                    ymins[j] = min(ymins[j],np.min([np.min(y2[0:10]),np.min(y2[45:138]),np.min(y2[209:365])]))
                else:
                    # Box for ThacBa in flood season (unregulated)
                    ax.plot([10,10], [0,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([10,208], [0,0], c='k', linewidth=2, linestyle='--')
                    ax.plot([10,208], [1,1], c='k', linewidth=2, linestyle='--')
                    ax.plot([208,208], [0,1], c='k', linewidth=2, linestyle='--')
                    ymaxs[j] = max(ymaxs[j],np.max([np.max(y2[0:10]),np.max(y2[209:365])]))
                    ymins[j] = min(ymins[j],np.min([np.min(y2[0:10]),np.min(y2[209:365])]))
                        
            if i != 0: # turn off tick labels if not in first column
                ax.tick_params(axis='y',labelleft='off')
            else: # label y axes if in first column
                ax.set_ylabel(ylabels[j],fontsize=16)
                ax.tick_params(axis='y',labelsize=14)
                
            if j == 0: # put titles on top row
                ax.set_title(titles[i],fontsize=16)
                
            if j == len(SIs)-1: # put xlabels on last row
                ax.set_xticks([15,45,75,106,137,167,198,229,259,289,319,350])
                ax.set_xticklabels(['M','J','J','A','S','O','N','D','J','F','M','A'],fontsize=14)
            else:
                ax.tick_params(axis='x',labelbottom='off')
                
            ax.set_xlim([0,364])
            axes.append(ax)
            
    for i in range(len(axes)):
        axes[i].set_ylim([ymins[np.mod(i,len(SIs))],ymaxs[np.mod(i,len(SIs))]])
                
    fig.text(0.02, 0.5, 'Portion of Variance', va='center', rotation='vertical',fontsize=18)
    fig.subplots_adjust(bottom=0.15,hspace=0.3)
    plt.figlegend([pSL,pHB,pTQ,pTB,pHNfcst,pInteract],\
                  [r'$s_t^{SL}$',r'$s_t^{HB}$',r'$s_t^{TQ}$',r'$s_t^{TB}$',r'$\tilde{z}_{t+2}^{HN}$','Interactions'],\
                  loc='lower center', ncol=3, fontsize=16, frameon=True)
    fig.set_size_inches([width,12.5])
    fig.savefig(figname)
    fig.clf()

    return None

makeFigures8_S8()