# Libraries used:
import numpy as np
from scipy.stats import norm
import random as rnd
import time
import sympy
import re
from math import factorial
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def C(S,K,T,r,σ):
    d1 = (np.log(S/K) + (r + .5*σ**2)*T) / (σ*np.sqrt(T))
    d2 = d1 - (σ*np.sqrt(T))
    c = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) 
    return c

S = 50
T = 1
r = .1
σ = .3
K = 50

c = C(S,K,T,r,σ)
print(c)

#%%

def next_prime(x):
    if x > 100000:
        print('ERROR: Must pick number less than or equal to {}'.format(x))
        return
    primes = np.array(list(sympy.primerange(2,100000)))
    return primes[primes <= x][-1]

def log_b(b,x):
    return np.log(x)/np.log(b)

def binomial(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))

def gens_rperms(s,b):
    maxi = 10000000
    k = int(log_b(b,maxi)) + 1
    gens = np.zeros((s,k,k))
    for i in range(k):
        gens[0][i][i] = 1
    p = lambda a, i, j: np.mod(a**(j-i) * binomial(j-1, i-1), b)
    for a in range(1,s):
        for i in range(k):
            for j in range(i,k):
                gens[a][i][j] = p(a,i+1,j+1)
    rperms = np.zeros((s,k,s))
    for i in range(s):
        for j in range(k):
            rperms[i][j] = np.insert(np.random.permutation(np.arange(1,b)),0,0)
    return gens, rperms

def dig_scrambled_faure(n,s,b,gens,rperms):
    array = np.zeros(s)
    digs = np.flip(np.array(list(np.base_repr(n,b)),'int'))
    size = len(digs)
    base_powers = np.array([1/b**i for i in range(1,size+1)])
    real_digs = np.zeros(size)
    perm_digs = np.zeros(size)
    for i in range(s):
        real_digs = np.array(list(map(lambda x: np.mod(x,b), np.matmul(gens[i][:size,:size],digs))))
        for j in range(size):
            perm_digs[j] = rperms[i][j][int(real_digs[j])]
        array[i] = sum(perm_digs*base_powers)
    return array

n = 100
s = 5
b = next_prime(s)

gens, rperms = gens_rperms(s,b)

dig_scram_faure_array = np.zeros((n,b))
for i in range(n):
    dig_scram_faure_array[i] = dig_scrambled_faure(i+1,s,b,gens,rperms)



#%%

def box_muller(u1, u2):
    R = np.sqrt(-2 * np.log(u1))
    θ = 2 * np.pi * u2
    X = R * np.cos(θ)
    Y = R * np.sin(θ)
    return X, Y

def beas_spri_moro(u):
    a0 =   2.50662823884
    a1 = -18.61500062529
    a2 =  41.39119773534
    a3 = -25.44106049637
    b0 =  -8.47351093090
    b1 =  23.08336743743
    b2 = -21.06224101826
    b3 =   3.13082909833
    c0 =    .3374754822726147
    c1 =    .9761690190917186
    c2 =    .1607979714918209
    c3 =    .0276438810333863
    c4 =    .0038405729373609
    c5 =    .0003951896511919
    c6 =    .0000321767881768
    c7 =    .0000002888167364
    c8 =    .0000003960315187
    
    y = u - .5
    if abs(y) < .42:
        r = y**2
        x = y * (((a3*r + a2)*r + a1)*r + a0)/((((b3*r + b2)*r + b1)*r + b0)*r + 1)
    else:
        r = u
        if y > 0:
            r = 1 - u
        r = np.log(-np.log(r))
        x = c0 + r*(c1 + r*(c2 + r*(c3+ r*(c4 + r*(c5 + r*(c6 + r*(c7 + r*c8)))))))
        if y < 0:
            x = -x
    return x
        
#%%

# def GBM(S,T,n,r,σ):
#     Δt = T/n
#     S_t = np.zeros(n)
#     S_t[0] = S
#     for i in range(1,n):
#         u1 = rnd.random()
#         u2 = rnd.random()
#         Z1, Z2 = box_muller(u1, u2)
#         S_t[i] = S_t[i-1] * np.exp((r - .5*σ**2)*Δt + σ*np.sqrt(Δt)*Z1)
#     return S_t

# n = 10
# sims = 10000
# S = 50
# T = 1
# r = .1
# σ = .3
# K = 50
# paths = np.zeros((sims,n))
# for i in range(sims):
#     paths[i] = GBM(S,T,n,r,σ)
    
# plt.plot(paths.T)

# payoffs = paths[:,-1]-K
# payoffs = np.maximum(payoffs,0)
# approx_px = payoffs.mean()
# print(approx_px)


#%%

def GBM(S,T,n,r,σ,method):
    Δt = T/n
    if method == 'box-muller':
        S_t = np.zeros((2,n+1))
        S_t[0] = S
        S_t[1] = S
        for i in range(1,n+1):
            u1 = rnd.random()
            u2 = rnd.random()
            Z1, Z2 = box_muller(u1, u2)
            S_t[0][i] = S_t[0][i-1] * np.exp((r - .5*σ**2)*Δt + σ*np.sqrt(Δt)*Z1)
            S_t[1][i] = S_t[1][i-1] * np.exp((r - .5*σ**2)*Δt + σ*np.sqrt(Δt)*Z2)
    elif method == 'beasley-springer-moro':
        S_t = np.zeros(n+1)
        S_t[0] = S
        for i in range(1,n+1):
            u = rnd.random()
            Z = beas_spri_moro(u)
            S_t[i] = S_t[i-1] * np.exp((r - .5*σ**2)*Δt + σ*np.sqrt(Δt)*Z)
    return S_t

#%%
n = 10
sims = 10000
S = 50
T = 1
r = .1
σ = .3
K = 50

paths = np.zeros((sims,n+1))
for i in np.arange(0,sims,2):
    paths[i:i+2] = GBM(S,T,n,r,σ,method='box-muller')
    
payoffs = paths[:,-1]-K
payoffs = np.maximum(paths[:,-1]-K,0)
avg_payoff = payoffs.mean()
approx_px = np.exp(-r*T)*avg_payoff
print(approx_px)

#%%

paths = np.zeros((sims,n+1))
for i in range(sims):
    paths[i] = GBM(S,T,n,r,σ,method='beasley-springer-moro')
    
payoffs = paths[:,-1]-K
payoffs = np.maximum(paths[:,-1]-K,0)
avg_payoff = payoffs.mean()
approx_px = np.exp(-r*T)*avg_payoff
print(approx_px)