import numpy as np
from numpy import random as rnd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%%
# black-scholes price
def BS(S,K,T,t,r,σ):
    d1 = (np.log(S/K) + (r + .5*σ**2)*(T-t)) / (σ*np.sqrt(T-t))
    d2 = d1 - (σ*np.sqrt(T-t))
    return S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2) 

#%%
S = 100
K = 110
T = 1
r = .1
σ = .25
m = 10
M = int(m*T)
N = 500000
poly_degree = 6

def LS_put(S,K,T,r,σ,M,N,poly_degree):
    Δt = T/M
    discount_factor = np.exp(-r * Δt)
    Z = rnd.normal(size=(int(N/2),M))
    Z = np.concatenate((Z,-Z))
    S_t = np.zeros((N,M+1)) # create matrix for all paths
    S_t[:,0] = S # intitialize S_0 for all paths
    a1 = (r - .5*σ**2)*Δt # constant calculated outside of loop to reduce flops required
    a2 = σ*np.sqrt(Δt) # constant calculated outside of loop to reduce flops required
    for k in range(M):
        S_t[:,k+1] = S_t[:,k] * np.exp(a1 + a2*Z[:,k])
        
    payoff = np.maximum(K - S_t, np.zeros_like(S_t))
    value = np.zeros_like(payoff)
    value[:, -1] = payoff[:, -1]
    
    for t in tqdm(range(M-1 , 0 , -1)):
        regression = np.polyfit(S_t[:, t], value[:, t + 1]*discount_factor,poly_degree)
        continuation_value = np.polyval(regression, S_t[:, t])
        value[:, t] = np.where(
            payoff[:, t] > continuation_value,
            payoff[:, t],
            value[:, t + 1] * discount_factor
        )
    option_premium = np.mean(value[:, 1] * discount_factor)
    return option_premium

px = LS_put(S,K,T,r,σ,M,N,poly_degree)
print('\n',px)

#%%

def RF_put(S,K,T,r,σ,M,N,antithetic,tree_num,depth,min_samples,samples_per_tree,):
    Δt = T/M
    discount_factor = np.exp(-r * Δt)
    if antithetic == False:
        Z = rnd.normal(size=(N,M))
    else:
        Z = rnd.normal(size=(int(N/2),M))
        Z = np.concatenate((Z,-Z))
    S_t = np.zeros((N,M+1)) # create matrix for all paths
    S_t[:,0] = S # intitialize S_0 for all paths
    a1 = (r - .5*σ**2)*Δt # constant calculated outside of loop to reduce flops required
    a2 = σ*np.sqrt(Δt) # constant calculated outside of loop to reduce flops required
    for k in range(M):
        S_t[:,k+1] = S_t[:,k] * np.exp(a1 + a2*Z[:,k])
        
    payoff = np.maximum(K - S_t, np.zeros_like(S_t))
    value = np.zeros_like(payoff)
    value[:, -1] = payoff[:, -1]
    
    rf = RandomForestRegressor(n_estimators=tree_num, max_depth=depth, min_samples_leaf=min_samples, bootstrap=True, n_jobs=-1, max_samples=samples_per_tree)

    for t in tqdm(range(M-1 , 0 , -1)):
        X = S_t[:,t].reshape(-1,1)
        Y = value[:,t+1]
        rf.fit(X,Y)
        continuation_value = rf.predict(X)
        value[:, t] = np.where(
            payoff[:, t] > continuation_value,
            payoff[:, t],
            value[:, t + 1] * discount_factor
        )
    option_premium = np.mean(value[:, 1] * discount_factor)
    return option_premium
#%%
    
# tree_num = 10
# depth = 5
# min_samples = 100
# samples_per_tree = None

# French paper (bermudan) px = 11.987
# S = 100
# K = 110
# T = 1
# r = .1
# σ = .25
# m = 10


# Dubrov example (Approx American) px = 6.7389

tree_num = 10
depth = 5
min_samples = 100
samples_per_tree = 1/tree_num

S = 100
K = 100
T = 1
r = .03
σ = .2
m = 252

# tree_num = 10
# depth = 5
# min_samples = 100
# samples_per_tree = 1/tree_num

# Longstaff Schwartz
# S = 40
# K = 40
# T = 1
# r = .06
# σ = .2
# m = 50


M = int(m*T)
N = 100000
antithetic = True

px = RF_put(S,K,T,r,σ,M,N,antithetic,tree_num,depth,min_samples,samples_per_tree)
print('\n',px)

#%%

# 168089005022003074079122026078102108046069006038058066023088087111117002118099001123122025060038021096117067102122102114006090014008073002058066064077123099084073035075071078082126078094096098127004004118091108100089082006108031025121065089118001024&EXT=pdf&INDEX=TRUE
# https://towardsdatascience.com/example-of-random-forest-application-in-finance-option-pricing-d6ee06356c6e
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# https://builtin.com/data-science/random-forest-python

# we need 1 forest (15 trees) per time point
# need to simulate first for training data to train model, then resimulate new paths
# on going through new paths, use the forest associated with that timepoint to compute continuation value
# proceed as usual