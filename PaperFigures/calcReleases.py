import numpy as np

def calcReleases(M, N, K, policy, normInputs, Output_ranges):    
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
                
    # Convert normalized releases to actual releases
    for i in range(np.shape(u)[1]):
        u[:,i] = Output_ranges[i,0] + u[:,i]*(Output_ranges[i,1]-Output_ranges[i,0])
        
    return u