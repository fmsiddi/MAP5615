# Libraries used:
import numpy as np
from scipy.stats import norm
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

def vdc(n,b):
    digs = list(np.base_repr(n,b))
    len_digs = len(digs)
    sum = 0
    for i in range(len_digs):
        sum += int(digs[i])/b**(len_digs-i)
    return sum

def halton(n,s):
    return [vdc(n,i) for i in primes[:s]]




