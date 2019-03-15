import numpy as np

class Delta:
    def __init__(self):
        self.name = None
        self.pos = None
        self.neg = None

def calcNumericalSIs_u():
    days = 365
    years = 1000
    solns = [33, 35, 37, 52]
    perturbNames = ['sl', 'hb', 'tq', 'tb', 'HNfcst']
    inputNames = ['sSL','sHB','sTQ','sTB','HNfcst']
    nInputs = len(inputNames)
    outputNames = ['uSL','uHB','uTQ','uTB']
    nOutputs = len(outputNames)
    IO_ranges = np.array([[2223600000, 3215000000, 402300000, 402300000, 0, \
                           0, 0, 0, 0], \
                            [12457000000, 10890000000, 2481000000, 3643000000, 20, \
                             40002, 35784, 13551, 3650]])
    
    header = ''
    for input in inputNames:
        header = header + ',' + input + '_1' # first order indices
        
    for i in range(len(inputNames)-1):
        for j in range(i+1,len(inputNames)):
            header = header + ',' + inputNames[i] + '+' + inputNames[j]
            
    header = header[1:] # remove beginning comma

    doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
    
    perturbations = ['0.05','0.005']
    
    # find covariances at each day
    for perturbation in perturbations:
        for soln in solns:
            cov = np.zeros([days, nInputs, nInputs])
            allData = np.zeros([years, days, nInputs + nOutputs])
            dsl = getDeltas('dsl', years, days, nInputs, nOutputs)
            dhb = getDeltas('dhb', years, days, nInputs, nOutputs)
            dtq = getDeltas('dtq', years, days, nInputs, nOutputs)
            dtb = getDeltas('dtb', years, days, nInputs, nOutputs)
            dHNfcst = getDeltas('dHNfcst', years, days, nInputs, nOutputs)
            deltas = [dsl, dhb, dtq, dtb, dHNfcst]
            for day in range(days):
                dailyData = np.loadtxt('./perturbations/Soln' + str(soln) + '/u/Delta' + perturbation + \
                                       '/HydroInfo_100_thinned_proc' + str(soln) + '_day' + str(doy[day]) + '.txt')
                dailyData = normalizeInputs(dailyData[:,0:(nInputs+nOutputs)], IO_ranges)
                if np.shape(dailyData)[0] > years:
                    dailyData = dailyData[0:years,:]
                allData[:,day,:] = dailyData
                cov[day,:,:] = np.cov(np.transpose(dailyData[:,0:nInputs]))
                
                for i in range(len(perturbNames)):
                    posData = np.loadtxt('./perturbations/Soln' + str(soln) + '/u/Delta' + perturbation + \
                                         '/HydroInfo_100_pos_d' + perturbNames[i] + '_day' + str(doy[day]) + '.txt')
                    posData = normalizeInputs(posData[:,0:(nInputs+nOutputs)], IO_ranges)
                    if np.shape(posData)[0] > years:
                        posData = posData[0:years,:]
                    negData = np.loadtxt('./perturbations/Soln' + str(soln) + '/u/Delta' + perturbation + \
                                         '/HydroInfo_100_neg_d' + perturbNames[i] + '_day' + str(doy[day]) + '.txt')
                    negData = normalizeInputs(negData[:,0:(nInputs+nOutputs)], IO_ranges)
                    if np.shape(negData)[0] > years:
                        negData = negData[0:years,:]
                    deltas[i].pos[:,day,:] = posData
                    deltas[i].neg[:,day,:] = negData
                
            # find sensitivity indices at each time step
            for output in range(nOutputs):
                allSI = np.zeros([days*years, int(nInputs + nInputs*(nInputs-1)/2)])
                for year in range(years):
                    for day in range(days):
                        for col in range(nInputs): # first order indices
                            # calculate empirical first order partial derivative of output with respect to input
                            D = (deltas[col].pos[year,day,output+nInputs] - deltas[col].neg[year,day,output+nInputs])/ \
                                (deltas[col].pos[year,day,col] - deltas[col].neg[year,day,col])                        
                            allSI[year*365+day,col] = D**2 * cov[day,col,col]
                            
                        count = 0
                        for col1 in range(nInputs-1): # second order indices
                            for col2 in range(col1+1,nInputs):
                                D1 = (deltas[col1].pos[year,day,output+nInputs] - deltas[col1].neg[year,day,output+nInputs])/ \
                                    (deltas[col1].pos[year,day,col1] - deltas[col1].neg[year,day,col1])
                                    
                                D2 = (deltas[col2].pos[year,day,output+nInputs] - deltas[col2].neg[year,day,output+nInputs])/ \
                                    (deltas[col2].pos[year,day,col2] - deltas[col2].neg[year,day,col2])
                                allSI[year*365+day,nInputs+count] = 2*D1*D2*cov[day,col1,col2]
                                count = count + 1
                                
                np.savetxt('./sensitivities/Soln' + str(soln) + '/u/Delta' + perturbation + '/HydroInfo_100_' + \
                           outputNames[output] + '.txt', allSI, header=header, comments='', delimiter=',')
                    
    return None

def getDeltas(name, years, days, nInputs, nOutputs):
    delta = Delta()
    delta.name = name
    delta.pos = np.zeros([years, days, nInputs + nOutputs])
    delta.neg = np.zeros([years, days, nInputs + nOutputs])
    
    return delta

def normalizeInputs(inputs, input_ranges):
    normInputs = np.zeros(np.shape(inputs))
    for i in range(np.shape(input_ranges)[1]):
        normInputs[:,i] = (inputs[:,i] - input_ranges[0,i]) / (input_ranges[1,i] - input_ranges[0,i])
    
    return normInputs

calcNumericalSIs_u()
