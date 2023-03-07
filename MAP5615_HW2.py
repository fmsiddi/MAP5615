import numpy as np
import sympy
import matplotlib.pyplot as plt
plt.style.use('seaborn')

primes = list(sympy.primerange(2,200))

def vdc(n,b):
    digs = list(np.base_repr(n,b))
    len_digs = len(digs)
    sum = 0
    for i in range(len_digs):
        sum += int(digs[i])/b**(len_digs-i)
    return sum

vdc_4000 = np.array([vdc(n,2) for n in range(1,4001)])

def halton(n,s):
    return [vdc(n,i) for i in primes[:s]]

# plt.scatter(hal_2d_1000[:,0],hal_2d_1000[:,1])

#The following function assumes F(x) is the cdf for the uniform distribution with a = 0 and b = 1
def A_2_max(X,t):
    N = int(len(X)/t)
    # Z = np.sort(X) # Normally we would compute Z = F(X_i), which in this case is simply X_i
    V = np.array([max(X[(i*t):((i+1)*t)]) for i in range(N)])
    Z = np.sort(V)
    var = [(2*(i+1)-1)*(np.log(Z[i]**t)+np.log(1-Z[N-i-1]**t)) for i in range(N)]
    return - N - (1/N)*np.sum(var)

test = A_2_max(vdc_4000,4) # .008233


# This is similar to the previous function but takes in a halton sequence
def A_2_max_halton(X):
    t = X.shape[1]
    N = X.shape[0]
    V = np.array([max(X[i]) for i in range(N)])
    Z = np.sort(V)
    var = [(2*(i+1)-1)*(np.log(Z[i]**t)+np.log(1-Z[N-i-1]**t)) for i in range(N)]
    return - N - (1/N)*np.sum(var)

# PRODUCING 1000 2-DIMENSIONAL HALTON VECTORS
hal_4d_1000 = np.array([halton(n,4) for n in range(1,1001)])

test = A_2_max_halton(hal_4d_1000) # .008233

#%%
