import numpy as np

class Formulation():
    def __init__(self):
        self.name = None
        self.reference = None
        self.bestFlood = None
        self.bestHydro = None
        self.bestDeficit = None
        self.compromise = None

class Soln():
    def __init__(self):
        self.solnNo = None
        self.FloodYr = None
        self.DeficitYr = None
        self.HydroYr = None
        self.uSL_SI = None
        self.uHB_SI = None
        self.uTQ_SI = None
        self.uTB_SI = None

def getFormulations(name):
    formulation = Formulation()
    formulation.name = name
    
    formulation.reference = np.loadtxt('./../' + name + '/' + name + '_thinned.reference')
    
    formulation.bestHydro = np.argmin(formulation.reference[:,0])+1
    formulation.bestDeficit = np.argmin(formulation.reference[:,1])+1
    formulation.bestFlood = np.argmin(formulation.reference[:,2])+1
    formulation.compromise = findCompromise(formulation.reference[:,0:3],1)+1
        
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

def getSolns(formulation, solnNo, perturbation):
    soln = Soln()
    soln.solnNo = solnNo
        
    if formulation.name == 'guidelines_100':
        soln = getGL(soln.solnNo, perturbation)
    else:
        allObjs = np.zeros([1000,365,3])
        doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
        for day in range(365):
            allObjs[:,day,:] = np.loadtxt('../HydroInfo_100/simulations/Soln' + str(soln.solnNo) + '/HydroInfo_100_thinned_proc' + \
                str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', usecols=[9,10,11])
            
        soln.FloodYr = np.argsort(np.max(allObjs[:,:,0],1))[990]
        soln.DeficitYr = np.argsort(np.sum(allObjs[:,:,1],1))[990]
        soln.HydroYr = np.argsort(np.sum(allObjs[:,:,2],1))[990]
        
        soln.uSL_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/uSL.txt',skiprows=1,delimiter=',')
        soln.uHB_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/uHB.txt',skiprows=1,delimiter=',')
        soln.uTQ_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/uTQ.txt',skiprows=1,delimiter=',')
        soln.uTB_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/uTB.txt',skiprows=1,delimiter=',')
    
    return soln

def getSolns_r(formulation, solnNo, perturbation):
    soln = Soln()
    soln.solnNo = solnNo
        
    if formulation.name == 'guidelines_100':
        soln = getGL_r(soln.solnNo, perturbation)
    else:
        allObjs = np.zeros([1000,365,3])
        doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
        for day in range(365):
            allObjs[:,day,:] = np.loadtxt('../HydroInfo_100/perturbations/Soln' + str(soln.solnNo) + '/r/Delta' + 
                   perturbation + '/HydroInfo_100_thinned_proc' + str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', \
                   usecols=[9,10,11])
            
        soln.FloodYr = np.argsort(np.max(allObjs[:,:,0],1))[990]
        soln.DeficitYr = np.argsort(np.sum(allObjs[:,:,1],1))[990]
        soln.HydroYr = np.argsort(np.sum(allObjs[:,:,2],1))[990]
        
        soln.uSL_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/r/Delta' + \
                                 perturbation + '/HydroInfo_100_rSL.txt',skiprows=1,delimiter=',')
        soln.uHB_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/r/Delta' + \
                                 perturbation + '/HydroInfo_100_rHB.txt',skiprows=1,delimiter=',')
        soln.uTQ_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/r/Delta' + \
                                 perturbation + '/HydroInfo_100_rTQ.txt',skiprows=1,delimiter=',')
        soln.uTB_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/r/Delta' + \
                                 perturbation + '/HydroInfo_100_rTB.txt',skiprows=1,delimiter=',')
    
    return soln

def getSolns_u(formulation, solnNo, perturbation):
    soln = Soln()
    soln.solnNo = solnNo
        
    allObjs = np.zeros([1000,365,3])
    doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
    for day in range(365):
        allObjs[:,day,:] = np.loadtxt('../HydroInfo_100/perturbations/Soln' + str(soln.solnNo) + '/u/Delta' + 
               perturbation + '/HydroInfo_100_thinned_proc' + str(soln.solnNo) + '_day' + str(doy[day]) + '.txt', \
               usecols=[9,10,11])
        
    soln.FloodYr = np.argsort(np.max(allObjs[:,:,0],1))[990]
    soln.DeficitYr = np.argsort(np.sum(allObjs[:,:,1],1))[990]
    soln.HydroYr = np.argsort(np.sum(allObjs[:,:,2],1))[990]
    
    soln.uSL_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/Delta' + \
                             perturbation + '/HydroInfo_100_uSL.txt',skiprows=1,delimiter=',')
    soln.uHB_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/Delta' + \
                             perturbation + '/HydroInfo_100_uHB.txt',skiprows=1,delimiter=',')
    soln.uTQ_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/Delta' + \
                             perturbation + '/HydroInfo_100_uTQ.txt',skiprows=1,delimiter=',')
    soln.uTB_SI = np.loadtxt('../HydroInfo_100/sensitivities/Soln' + str(soln.solnNo) + '/u/Delta' + \
                             perturbation + '/HydroInfo_100_uTB.txt',skiprows=1,delimiter=',')
    
    return soln

def getGL(solnNo, perturbation):
    soln = Soln()
    soln.solnNo = solnNo
    allObjs = np.zeros([1000,365,3])
    doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
    for day in range(365):
        allObjs[:,day,:] = np.loadtxt('../guidelines_100/perturbations/Soln' + str(soln.solnNo) + \
               '/u/Delta' + perturbation + '/guidelines_100_day' + str(doy[day]) + '.txt', usecols=[9,10,11])
        
    soln.FloodYr = np.argsort(np.max(allObjs[:,:,0],1))[990]
    soln.DeficitYr = np.argsort(np.sum(allObjs[:,:,1],1))[990]
    soln.HydroYr = np.argsort(np.sum(allObjs[:,:,2],1))[990]
    
    soln.uSL_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/u/Delta' + perturbation + '/guidelines_100_uSL.txt',skiprows=1,delimiter=',')
    soln.uHB_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/u/Delta' + perturbation + '/guidelines_100_uHB.txt',skiprows=1,delimiter=',')
    soln.uTQ_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/u/Delta' + perturbation + '/guidelines_100_uTQ.txt',skiprows=1,delimiter=',')
    soln.uTB_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/u/Delta' + perturbation + '/guidelines_100_uTB.txt',skiprows=1,delimiter=',')
    
    return soln

def getGL_r(solnNo, perturbation):
    soln = Soln()
    soln.solnNo = solnNo
    allObjs = np.zeros([1000,365,3])
    doy = np.array(np.concatenate((np.arange(121,366,1),np.arange(1,121,1)),0))
    for day in range(365):
        allObjs[:,day,:] = np.loadtxt('../guidelines_100/perturbations/Soln' + str(soln.solnNo) + \
               '/r/Delta' + perturbation + '/guidelines_100_day' + str(doy[day]) + '.txt', usecols=[9,10,11])
        
    soln.FloodYr = np.argsort(np.max(allObjs[:,:,0],1))[990]
    soln.DeficitYr = np.argsort(np.sum(allObjs[:,:,1],1))[990]
    soln.HydroYr = np.argsort(np.sum(allObjs[:,:,2],1))[990]
    
    soln.uSL_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/r/Delta' + perturbation + '/guidelines_100_rSL.txt',skiprows=1,delimiter=',')
    soln.uHB_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/r/Delta' + perturbation + '/guidelines_100_rHB.txt',skiprows=1,delimiter=',')
    soln.uTQ_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/r/Delta' + perturbation + '/guidelines_100_rTQ.txt',skiprows=1,delimiter=',')
    soln.uTB_SI = np.loadtxt('../guidelines_100/sensitivities/Soln' + str(soln.solnNo) + \
                             '/r/Delta' + perturbation + '/guidelines_100_rTB.txt',skiprows=1,delimiter=',')
    
    return soln

def getFullVar(var, M, K, N):
    fullVar = np.zeros([N*(2*M+K)])
    count = 0
    for i in range(N):
        for j in range(M-2):
            fullVar[count + 4*i] = var[count]
            fullVar[count + 4*i + 1] = var[count + 1]
            count = count + 2
            
        for j in range(2):
            fullVar[count + 4*i + 2*j] = 0.0
            fullVar[count + 4*i + 2*j + 1] = 1.0
            
        for k in range(K):
            fullVar[count + 4*(i+1)] = var[count]
            count = count + 1
    
    return fullVar

def calcReleases(M, K, N, policy, normInputs):    
    # Re-organize decision variables into weight (W), center (C) and raddi (R) matrices
    # C and R are M x N, W is K X N where
    C = np.zeros([M,N])
    R = np.zeros([M,N])
    W = np.zeros([K,N])
    for n in range(N):
        for m in range(M):
            C[m,n] = policy[(2*M+K)*n + 2*m]
            R[m,n] = policy[(2*M+K)*n + 2*m + 1]
        for k in range(K):
            W[k,n] = policy[(2*M+K)*n + 2*M + k]
            
    # Normalize weights to sum to 1 across N RBFs (so each row of W should sum to 1)
    totals = np.sum(W,1)
    for k in range(K):
        if totals[k] > 10**-6:
            W[k,:] = W[k,:]/totals[k]
            
    # Calculate normalized releases (u) corresponding to each sample of normalized inputs
    # u is a 2-D matrix with K columns (1 for each output)
    # and as many rows as there are samples of inputs
    u = np.zeros([np.shape(normInputs)[0],K])
    for i in range(np.shape(normInputs)[0]):
        for k in range(K):
            for n in range(N):
                BF = 0
                for m in range(M):
                    if R[m,n] > 10**-6:
                        BF = BF + ((normInputs[i,m]-C[m,n])/R[m,n])**2
                    else:
                        BF = BF + ((normInputs[i,m]-C[m,n])/(10**-6))**2
                    
                u[i,k] = u[i,k] + W[k,n]*np.exp(-BF)
        
    return u