import numpy as np
import math

M = 5 # number of inputs: storage at the 4 reservoirs, forecasted water level at Hanoi, \
# ignore sin(2*pi*t/365 - phi1) and cos(2*pi*t/365 - phi2) b/c their centers and radii are fixed
N = 12 # number of RBFs
K = 4 # number of outputs: release at the 4 reservoirs

def calcAnalyticalSIs():
    policyVars = np.loadtxt('HydroInfo_100_thinned.resultfile')
    solns = [35, 37, 52, 33]
    days = 365
    years = 1000
    inputNames = ['sSL','sHB','sTQ','sTB','HNfcst']
    outputNames = ['uSL','uHB','uTQ','uTB']
    # first row inputs lower bounds, second row outputs lower bounds (including bounds on sin() and cos())
    # third row inputs upper bounds, fourth row output upper bounds
    IO_ranges = np.array([[2223600000, 3215000000, 402300000, 402300000, 0, -1, -1, \
                           0, 0, 0, 0], \
                            [12457000000, 10890000000, 2481000000, 3643000000, 20, 1, 1, \
                             40002, 35784, 13551, 3650]])
    
    header = ''
    for input in inputNames:
        header = header + ',' + input + '_1' # first order indices
        
    for i in range(len(inputNames)-1):
        for j in range(i+1,len(inputNames)):
            header = header + ',' + inputNames[i] + '+' + inputNames[j]
            
    header = header[1:] # remove beginning comma
    
    doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
    
    for soln in solns:
        # load decision variables of this solution
        policy = policyVars[soln-1,0:168]
        phi1 = policyVars[soln-1,168]
        phi2 = policyVars[soln-1,169]
        C, B, W = reorganizeVars(M, N, K, policy)
        
        # find covariances at each day
        cov = np.zeros([days, M, M])
        allData = np.zeros([years, days, M+2+K])
        for day in range(days):
            dailyData = np.loadtxt('./simulations/Soln' + str(soln) + '/HydroInfo_100_thinned_proc' + str(soln) + '_day' + str(doy[day]) + '.txt')
            # append sin() and cos() functions to inputs
            sinArray = np.ones([years,1])*np.sin(2*math.pi*doy[day]/365.0 - phi1)
            cosArray = np.ones([years,1])*np.cos(2*math.pi*doy[day]/365.0 - phi2)
            dailyData = np.concatenate((np.concatenate((dailyData[:,0:M],sinArray),1),dailyData[:,M:(M+K)]),1)
            dailyData = np.concatenate((np.concatenate((dailyData[:,0:(M+1)],cosArray),1),dailyData[:,(M+1):(M+1+K)]),1)
            dailyData = normalizeInputs(dailyData[:,0:(M+2+K)], IO_ranges)
            allData[:,day,:] = dailyData
            cov[day,:,:] = np.cov(np.transpose(dailyData[:,0:M]))
            
        # find sensitivity indices at each time step
        for output in range(K):
            allSI = np.zeros([days*years, int(M + M*(M-1)/2)])
            for year in range(years):
                for day in range(days):
                    inputValues = allData[year,day,0:(M+2)]
                    for col in range(M): # first order indices
                        D = calcD(C, B, W, col, inputValues, output)
                        allSI[year*365+day,col] = D**2 * cov[day,col,col]
                        
                    count = 0
                    for col1 in range(M-1): # second order indices
                        for col2 in range(col1+1,M):
                            D1 = calcD(C, B, W, col1, inputValues, output)
                            D2 = calcD(C, B, W, col2, inputValues, output)
                            allSI[year*365+day,M+count] = 2*D1*D2*cov[day,col1,col2]
                            count = count + 1
                            
            np.savetxt('./sensitivities/Soln' + str(soln) + '/u/' + outputNames[output] + '.txt', \
                       allSI, header=header, comments='', delimiter=',')
                    
    return None
                    
def calcD(C, B, W, inputNumber, inputValues, outputNumber):
    # calculate analytical first order partial derivative of RBF outputNumber with respect to inputNumber 
    # located at inputValues
    D = 0
    for n in range(N):
        innerSum = 0
        for m in range(M):
            innerSum = innerSum - (inputValues[m] - C[m,n])**2 / B[m,n]**2
        
        D = D - 2 * ((inputValues[inputNumber]-C[inputNumber,n])/(B[inputNumber, n])**2) * W[outputNumber,n] * np.exp(innerSum)
    
    return D

def reorganizeVars(M, N, K, policy):    
    # Re-organize decision variables into weight (W), center (C) and raddi (B) matrices
    # C and R are M x N, W is K X N where M = # of inputs, K = # of outputs and N = # of RBFs
    C = np.zeros([M+2,N])
    B = np.zeros([M+2,N])
    W = np.zeros([K,N])
    for n in range(N):
        for m in range(M):
            C[m,n] = policy[(2*M+K)*n + 2*m]
            if policy[(2*M+K)*n + 2*m + 1] < 10**-6:
                B[m,n] = 10**-6
            else:
                B[m,n] = policy[(2*M+K)*n + 2*m + 1]
        
        C[M,n] = 0.0 # center for sin(2*pi*t/365 - phi1)
        C[M+1,n] = 0.0 # center for cos(2*pi*t/365 - phi1)
        B[M,n] = 1.0 # radius for sin(2*pi*t/365 - phi1)
        B[M+1,n] = 1.0 # radius for cos(2*pi*t/365 - phi1)
        for k in range(K):
            W[k,n] = policy[(2*M+K)*n + 2*M + k]
            
    # Normalize weights to sum to 1 across N RBFs (so each row of W should sum to 1)
    totals = np.sum(W,1)
    for k in range(K):
        if totals[k] > 10**-6:
            W[k,:] = W[k,:]/totals[k]
        
    return C, B, W

def normalizeInputs(inputs, input_ranges):
    normInputs = np.zeros(np.shape(inputs))
    for i in range(np.shape(input_ranges)[1]):
        normInputs[:,i] = (inputs[:,i] - input_ranges[0,i]) / (input_ranges[1,i] - input_ranges[0,i])
    
    return normInputs

calcAnalyticalSIs()
