import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class Reservoir:
    def __init__(self):
        self.min_rel = None
        self.Ltarget = None
        self.Utarget = None    

def makeFigureS1():

    SonLa = getReservoirs('SL')
    HoaBinh = getReservoirs('HB')
    TachBa = getReservoirs('TB')
    TuyenQuang = getReservoirs('TQ')
    
    sns.set_style("dark")
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.plot(np.arange(0,365),SonLa.Utarget,c='#fb9a99',linestyle='-',linewidth=2)
    ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='x',labelbottom='off')
    ax.set_ylabel('Reservoir Water Level',fontsize=16)
    ax.set_title('Son La',fontsize=18)
    
    ax = fig.add_subplot(2,2,2)
    ax.plot(np.arange(0,365),TuyenQuang.Utarget,c='#33a02c',linestyle='-',linewidth=2)
    ax.plot(np.arange(0,365),TuyenQuang.Ltarget,c='#33a02c',linestyle='--',linewidth=2)
    ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='x',labelbottom='off')
    ax.set_ylabel('Reservoir Water Level',fontsize=16)
    ax.set_title('Tuyen Quang',fontsize=18)
    ax2 = ax.twinx()
    ax2.plot(np.arange(0,365),TuyenQuang.min_rel,c='#33a02c',linestyle=':',linewidth=2)
    ax2.tick_params(axis='both',labelsize=14)
    ax2.tick_params(axis='x',labelbottom='off')
    ax2.set_ylabel('Reservoir Release',fontsize=16)
    
    ax = fig.add_subplot(2,2,3)
    ax.plot(np.arange(0,365),HoaBinh.Utarget,c='#e31a1c',linestyle='-',linewidth=2)
    ax.plot(np.arange(0,365),HoaBinh.Ltarget,c='#e31a1c',linestyle='--',linewidth=2) 
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xticks([15,45,75,106,137,167,198,229,259,289,319,350])
    ax.set_xticklabels(['M','J','J','A','S','O','N','D','J','F','M','A'],fontsize=16)
    ax.set_ylabel('Reservoir Water Level',fontsize=16)
    ax.set_title('Hoa Binh',fontsize=18)
    ax2 = ax.twinx()  
    ax2.plot(np.arange(0,365),HoaBinh.min_rel,c='#e31a1c',linestyle=':',linewidth=2)
    ax2.tick_params(axis='both',labelsize=14)
    ax2.set_ylabel('Reservoir Release',fontsize=16)
    
    ax = fig.add_subplot(2,2,4)
    ax.plot(np.arange(0,365),TachBa.Ltarget,c='#6a3d9a',linestyle='--',linewidth=2)
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xticks([15,45,75,106,137,167,198,229,259,289,319,350])
    ax.set_xticklabels(['M','J','J','A','S','O','N','D','J','F','M','A'],fontsize=16)
    ax.set_ylabel('Reservoir Water Level',fontsize=16)
    ax.set_title('Tach Ba',fontsize=18)
    ax2 = ax.twinx()  
    ax2.plot(np.arange(0,365),TachBa.min_rel,c='#6a3d9a',linestyle=':',linewidth=2)
    ax2.tick_params(axis='both',labelsize=14)
    ax2.set_ylabel('Reservoir Release',fontsize=16)
    
    upperLine, = ax.plot([],[],c='k',linestyle='-',linewidth=2)
    lowerLine, = ax.plot([],[],c='k',linestyle='--',linewidth=2)
    releaseLine, = ax.plot([],[],c='k',linestyle=':',linewidth=2)
    
    fig.set_size_inches([14.5,9.5])
    fig.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)
    fig.legend([upperLine,lowerLine,releaseLine],['Upper Target','Lower Target','Minimum Release'],loc='lower center',
               ncol=3,fontsize=16,frameon=True)
    
    fig.savefig('FigureS1.pdf')
    fig.clf()

    return None

def getReservoirs(name):

    reservoir = Reservoir()
    if name == 'SL' or name == 'TQ' or name == 'HB':
        reservoir.Utarget = np.array(np.concatenate((np.loadtxt('../guidelines_100/targets/' + \
                                                                name + '_lev_Utarget.txt')[120::],
                                                     np.loadtxt('../guidelines_100/targets/' + \
                                                                name + '_lev_Utarget.txt')[0:120]),0))
    if name == 'TB' or name == 'TQ' or name == 'HB':
        reservoir.Ltarget = np.array(np.concatenate((np.loadtxt('../guidelines_100/targets/' + \
                                                                name + '_lev_Ltarget.txt')[120::],
                                                    np.loadtxt('../guidelines_100/targets/' + \
                                                               name + '_lev_Ltarget.txt')[0:120]),0)) 
        reservoir.min_rel = np.array(np.concatenate((np.loadtxt('../guidelines_100/targets/' + \
                                                                name + '_min_rel.txt')[120::],
                                                    np.loadtxt('../guidelines_100/targets/' + \
                                                               name + '_min_rel.txt')[0:120]),0))
    
    return reservoir

makeFigureS1()
