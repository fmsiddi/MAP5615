import numpy as np
from numpy import random as rnd
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')
#%%

paths = np.array([[1,1.09,1.08,1.34],
                  [1,1.16,1.26,1.54],
                  [1,1.22,1.07,1.03],
                  [1,.93,.97,.92],
                  [1,1.11,1.56,1.52],
                  [1,.76,.77,.9],
                  [1,.92,.84,1.01],
                  [1,.88,1.22,1.34]])
# paths = np.array([[1,1.09,1.08,1.34],
#                   [1,1.16,1.26,1.54],
#                   [1,1.02,1.07,1.03],
#                   [1,.93,.97,.92],
#                   [1,1.11,1.56,1.52],
#                   [1,.76,.77,.9],
#                   [1,.92,.84,1.01],
#                   [1,.88,1.22,1.34]])
T = 3
N = 8
K=1.1
r=.06
Δt = 1
M = int(T/Δt + 1)
t = np.linspace(0,T,M)
cf = np.zeros((N,M))
cf[:,-1] = np.where(K-paths[:,-1]<0,0,K-paths[:,-1])
# for i in range(M-1,0,-1):
for i in range(M-1,0,-1):
    cf_index = np.where(K-paths[:,i-1]>0)[0]
    X = paths[np.where(K-paths[:,i-1]>0),i-1][0]
    Y = np.zeros(X.shape[0])
    for m in range(i,M):
        Y += cf[np.where(K-paths[:,i-1]>0),m][0]*np.exp(-r*Δt*(m-i+1))
    
    model = np.poly1d(np.polyfit(X, Y, 2))
    exercise = K - X
    continuation = model(X)
    index = np.where(exercise-continuation>0)[0]
    cf[cf_index[index],i-1] = exercise[index]
    cf[:,i] = np.where(cf[:,i]-cf[:,i-1]==cf[:,i],cf[:,i],0)
    for n in range(N):
        cf[n,:] = np.where(cf[n,:].max()-cf[n,:]>0,0,cf[n,:])
px = 0
for i in range(1,M):
    px += sum(cf[:,i]*np.exp(-r*t[i]))
px = px/N
print(px)

#%%
# black-scholes price
def BS(S,K,T,t,r,σ):
    d1 = (np.log(S/K) + (r + .5*σ**2)*(T-t)) / (σ*np.sqrt(T-t))
    d2 = d1 - (σ*np.sqrt(T-t))
    return S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2) 

def LSM_put(S,K,T,r,σ,M,N):
    Δt = T/M
    Z = rnd.normal(size=(N,M))
    S_t = np.zeros((N,M+1)) # create matrix for all paths
    S_t[:,0] = S # intitialize S_0 for all paths
    a1 = (r - .5*σ**2)*Δt # constant calculated outside of loop to reduce flops required
    a2 = σ*np.sqrt(Δt) # constant calculated outside of loop to reduce flops required
    for k in range(M):
        S_t[:,k+1] = S_t[:,k] * np.exp(a1 + a2*Z[:,k])
    
    cf = np.zeros(S_t.shape)
    cf[:,-1] = np.where(K-S_t[:,-1]<0,0,K-S_t[:,-1])
    for i in range(M,0,-1):
        cf_index = np.where(K-S_t[:,i-1]>0)[0]
        X = S_t[np.where(K-S_t[:,i-1]>0),i-1][0]
        Y = np.zeros(X.shape[0])
        for m in range(i,M+1):
            Y += cf[np.where(K-S_t[:,i-1]>0),m][0]*np.exp(-r*Δt*(m-i+1))
        # Y = cf[np.where(K-S_t[:,i-1]>0),i][0]*np.exp(-r*Δt)
        
        model = np.poly1d(np.polyfit(X, Y, 2))
        exercise = K - X
        continuation = model(X)
        index = np.where(exercise-continuation>0)[0]
        cf[cf_index[index],i-1] = exercise[index]
        cf[:,i] = np.where(cf[:,i]-cf[:,i-1]==cf[:,i],cf[:,i],0)
        for n in range(N):
            cf[n,i:] = np.where(cf[n,i:].max()-cf[n,i:]>0,0,cf[n,i:])
    
    t = np.linspace(0,T,M+1)
    px = 0
    
    for i in range(1,M+1):
        px += sum(cf[:,i]*np.exp(-r*t[i]))
    px = px/N
    return px

S = 36
K = 40
T = 1
r = .06
σ = .2
m = 50
M = int(m*T)
N = 100000

Δt = T/M
Z = rnd.normal(size=(N,M))
S_t = np.zeros((N,M+1)) # create matrix for all paths
S_t[:,0] = S # intitialize S_0 for all paths
a1 = (r - .5*σ**2)*Δt # constant calculated outside of loop to reduce flops required
a2 = σ*np.sqrt(Δt) # constant calculated outside of loop to reduce flops required
for k in range(M):
    S_t[:,k+1] = S_t[:,k] * np.exp(a1 + a2*Z[:,k])

cf = np.zeros(S_t.shape)
cf[:,-1] = np.where(K-S_t[:,-1]<0,0,K-S_t[:,-1])
for i in tqdm(range(M,0,-1)):
    cf_index = np.where(K-S_t[:,i-1]>0)[0]
    X = S_t[np.where(K-S_t[:,i-1]>0),i-1][0]
    Y = np.zeros(X.shape[0])
    for m in range(i,M+1):
        Y += cf[np.where(K-S_t[:,i-1]>0),m][0]*np.exp(-r*Δt*(m-i+1))
    # Y = cf[np.where(K-S_t[:,i-1]>0),i][0]*np.exp(-r*Δt)
    
    model = np.poly1d(np.polyfit(X, Y, 2))
    exercise = K - X
    continuation = model(X)
    index = np.where(exercise-continuation>0)[0]
    cf[cf_index[index],i-1] = exercise[index]
    cf[:,i] = np.where(cf[:,i]-cf[:,i-1]==cf[:,i],cf[:,i],0)
    for n in range(N):
        cf[n,i:] = np.where(cf[n,i:].max()-cf[n,i:]>0,0,cf[n,i:])

t = np.linspace(0,T,M+1)
px = 0

for i in range(1,M+1):
    px += sum(cf[:,i]*np.exp(-r*t[i]))
px = px/N
print(px)

# S_t = LSM_put(S,K,T,r,σ,M,N)
