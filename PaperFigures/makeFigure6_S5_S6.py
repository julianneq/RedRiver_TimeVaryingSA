import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from utils import getFormulations, getSolns, getSolns_u, getSolns_r

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

def makeFigures6_S5_S6():
    # get data for HydroInfo_100 formulation, including:
    # FloodYr = year of 100-yr flood
    # DeficitYr = year of 100-yr squared deficit
    # HydroYr = year of 100-yr hydropower production
    HydroInfo_100 = getFormulations('HydroInfo_100')
    
    # compare release sensitivities across solutions (rows) and events (columns)
    # using analytical estimates of prescribed releases (Figure 6)
    BestFloodSoln = getSolns(HydroInfo_100, HydroInfo_100.bestFlood, '0')
    BestHydroSoln = getSolns(HydroInfo_100, HydroInfo_100.bestHydro, '0')
    BestDefSoln = getSolns(HydroInfo_100, HydroInfo_100.bestDeficit, '0')
    CompromiseSoln = getSolns(HydroInfo_100, HydroInfo_100.compromise, '0')
    solns = [BestFloodSoln, BestHydroSoln, BestDefSoln, CompromiseSoln]
    ylabels = ['EMODPS Best\nFlood Policy','EMODPS Best\nHydro Policy','EMODPS Best\nDeficit Policy',\
               'EMODPS\nCompromise\nPolicy']
    outputs = [r'$u_t^{SL}$',r'$u_t^{HB}$',r'$u_t^{TQ}$',r'$u_t^{TB}$']
    makePlot(solns, reservoir, outputs, ylabels, 12.5, 'Figure6.pdf')
    
    # using numerical estimates of prescribed releases (Figure S4)
    BestFloodSoln = getSolns_u(HydroInfo_100, HydroInfo_100.bestFlood, '0.005')
    BestHydroSoln = getSolns_u(HydroInfo_100, HydroInfo_100.bestHydro, '0.005')
    BestDefSoln = getSolns_u(HydroInfo_100, HydroInfo_100.bestDeficit, '0.005')
    CompromiseSoln = getSolns_u(HydroInfo_100, HydroInfo_100.compromise, '0.005')
    solns = [BestFloodSoln, BestHydroSoln, BestDefSoln, CompromiseSoln]
    ylabels = ['EMODPS Best\nFlood Policy','EMODPS Best\nHydro Policy','EMODPS Best\nDeficit Policy',\
               'EMODPS\nCompromise\nPolicy']
    outputs = [r'$u_t^{SL}$',r'$u_t^{HB}$',r'$u_t^{TQ}$',r'$u_t^{TB}$']
    makePlot(solns, reservoir, outputs, ylabels, 12.5, 'FigureS5.pdf')
    
    # using numerical estimates of true releases (Figure S5)
    BestFloodSoln = getSolns_r(HydroInfo_100, HydroInfo_100.bestFlood, '0.005')
    BestHydroSoln = getSolns_r(HydroInfo_100, HydroInfo_100.bestHydro, '0.005')
    BestDefSoln = getSolns_r(HydroInfo_100, HydroInfo_100.bestDeficit, '0.005')
    CompromiseSoln = getSolns_r(HydroInfo_100, HydroInfo_100.compromise, '0.005')
    solns = [BestFloodSoln, BestHydroSoln, BestDefSoln, CompromiseSoln]
    ylabels = ['EMODPS Best\nFlood Policy','EMODPS Best\nHydro Policy','EMODPS Best\nDeficit Policy',\
               'EMODPS\nCompromise\nPolicy']
    outputs = [r'$r_{t+1}^{SL}$',r'$r_{t+1}^{HB}$',r'$r_{t+1}^{TQ}$',r'$r_{t+1}^{TB}$']
    makePlot(solns, reservoir, outputs, ylabels, 12.5, 'FigureS6.pdf')
    
    return None

def makePlot(solns, reservoir, outputs, ylabels, height, figureName):
        
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
                ax.plot(range(0,365),y2,c='None')
                ax.fill_between(range(0,365), y1, y2, where=y2>y1, color=colors[k])
                ymaxs[i] = max(ymaxs[i],np.max(y2))
                y1 = y2
                
            y2 = np.ones([365])
            ZeroIndices = np.where(y1==0)
            y2[ZeroIndices] = 0
            negIndices = np.where(np.sum(SIs[years[j]*365:(years[j]+1)*365,len(inputs)::],1)<0)
            y2[negIndices] = np.sum(SIs[years[j]*365:(years[j]+1)*365,len(inputs)::],1)[negIndices]/ \
                np.sum(SIs[years[j]*365:(years[j]+1)*365,:],1)[negIndices]
            ax.fill_between(range(0,365), y1, y2, where=y1<y2, color=colors[-1])
            ax.fill_between(range(0,365), y2, 0, where=y1>y2, color=colors[-1])
            ymaxs[i] = max(ymaxs[i], np.max(y2))
            ymins[i] = min(ymins[i], np.min(y2))
            
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
                
    fig.text(0.01, 0.5, 'Portion of Variance', va='center', rotation='vertical', fontsize=18)
    fig.subplots_adjust(bottom=0.15,hspace=0.3)
    plt.figlegend([pSL,pHB,pTQ,pTB,pHNfcst,pInteract],\
                  [r'$s_t^{SL}$',r'$s_t^{HB}$',r'$s_t^{TQ}$',r'$s_t^{TB}$',r'$\tilde{z}_{t+2}^{HN}$','Interactions'],\
                  loc='lower center', ncol=3, fontsize=16, frameon=True)
    fig.suptitle('Sensitivity of ' + outputs[1], fontsize=18)
    fig.set_size_inches([10,height])
    fig.savefig(figureName)
    fig.clf()
    
    return None

makeFigures6_S5_S6()
