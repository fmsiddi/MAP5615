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
N = 100000
poly_degree = 6

def put(S,K,T,r,σ,M,N,method):
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
        if method == 'LS':
            regression = np.polyfit(S_t[:, t], value[:, t + 1]*discount_factor,poly_degree)
            continuation_value = np.polyval(regression, S_t[:, t])
        elif method == 'RF':
            regression = RandomForestRegressor(n_estimators=10, max_depth=5, min_samples_leaf=100)
            continuation_value = regression.fit(S_t[:, t], value[:, t + 1]*discount_factor)
        value[:, t] = np.where(
            payoff[:, t] > continuation_value,
            payoff[:, t],
            value[:, t + 1] * discount_factor
        )
    option_premium = np.mean(value[:, 1] * discount_factor)
    return option_premium

#%%

S = 100
K = 110
T = 1
r = .1
σ = .25
m = 10
M = int(m*T)
N = 100000
poly_degree = 6
method = 'RF'

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
X = S_t[:,1].reshape(1,-1)
Y = payoff[:, -1]
rf = RandomForestRegressor(n_estimators=10, max_depth=5, min_samples_leaf=100)
rf_reg = rf.fit(X,Y)

for t in tqdm(range(M-1 , 0 , -1)):
    if method == 'LS':
        regression = np.polyfit(S_t[:, t], value[:, t + 1]*discount_factor,poly_degree)
        continuation_value = np.polyval(regression, S_t[:, t])
    elif method == 'RF':
        continuation_value = rf.predict(S_t[:, t].reshape(1,-1))
    value[:, t] = np.where(
        payoff[:, t] > continuation_value,
        payoff[:, t],
        value[:, t + 1] * discount_factor
    )
option_premium = np.mean(value[:, 1] * discount_factor)
print(option_premium)

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