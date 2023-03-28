# Libraries used:
import numpy as np
from scipy.stats import norm
import random as rnd
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

primes = np.array(list(sympy.primerange(2,1000000)))

#%%

def next_prime(x):
    return primes[primes <= x][-1]

def log_b(b,x):
    return np.log(x)/np.log(b)

def binomial(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))

s = 5
b = next_prime(s)
maxi = 10000000
k = int(log_b(b,maxi)) + 1
print(k)

#%%

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

#%%

# def box_muller():
#     R = np.sqrt(-2 * np.log(np.random.uniform()))
#     θ = 2 * np.pi * np.random.uniform()
#     X = R * np.cos(θ)
#     Y = R * np.sin(θ)
#     return X, Y

def box_muller():
    u1 = rnd.random()
    u2 = rnd.random()
    R = np.sqrt(-2 * np.log(u1))
    θ = 2 * np.pi * u2
    X = R * np.cos(θ)
    Y = R * np.sin(θ)
    return X, Y

#%%

def GBM(S,T,n,r,σ):
    Δt = T/n
    S_t = np.zeros((2,n))
    S_t[0] = S
    S_t[1] = S
    for i in range(1,n):
        Z1, Z2 = box_muller()
        S_t[0][i] = S_t[0][i-1] * np.exp((r - .5*σ**2)*Δt + σ*np.sqrt(Δt)*Z1)
        S_t[1][i] = S_t[1][i-1] * np.exp((r - .5*σ**2)*Δt + σ*np.sqrt(Δt)*Z2)
    return S_t

n = 10
num = 10000
S = 50
T = 1
r = .1
σ = .3
K = 50
paths = np.zeros((num,n))
for i in np.arange(0,num,2):
    paths[i:i+2] = GBM(S,T,n,r,σ)
    
# plt.plot(paths.T)
# plt.plot(GBM(S,T,n,r,σ))
# GBM(S,T,n,r,σ)

avg_S = paths[:,-1].mean()
approx_px = max(avg_S - K, 0)
        
#%%

def GBM(S,T,n,r,σ):
    Δt = T/n
    S_t = np.zeros(n)
    S_t[0] = S
    for i in range(1,n):
        Z1, Z2 = box_muller()
        S_t[i] = S_t[i-1] * np.exp((r - .5*σ**2)*Δt + σ*np.sqrt(Δt)*Z1)
    return S_t

n = 10
num = 10000
S = 50
T = 1
r = .1
σ = .3
K = 50
paths = np.zeros((num,n))
for i in range(num):
    paths[i] = GBM(S,T,n,r,σ)
    
# plt.plot(paths.T)
# plt.plot(GBM(S,T,n,r,σ))
# GBM(S,T,n,r,σ)

payoffs = paths[:,-1]-K
payoffs = np.maximum(payoffs,0)
approx_px = payoffs.mean()
print(approx_px)

#%%

# n = 50000
# test = np.zeros(2*n)
# for i in np.arange(0,n,2):
#     Z1, Z2 = box_muller()
#     test[i] = Z1
#     test[i+1] = Z2
    
n = 100000
test = np.zeros(n)
for i in range(n):
    Z1, Z2 = box_muller()
    test[i] = Z2

plt.hist(test,bins=50)



#%%

def vdc(n,b):
    digs = list(np.base_repr(n,b))
    len_digs = len(digs)
    sum = 0
    for i in range(len_digs):
        sum += int(digs[i])/b**(len_digs-i)
    return sum

def halton(n,s):
    return [vdc(n,i) for i in primes[:s]]



