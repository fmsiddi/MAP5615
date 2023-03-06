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

vdc_test = np.array([vdc(n,2) for n in range(10)])

def halton(n,s):
    return [vdc(n,i) for i in primes[:s]]


# PRODUCING 1000 2-DIMENSIONAL HALTON VECTORS
hal_2d_1000 = np.array([halton(n,2) for n in range(1,1001)])

plt.scatter(hal_2d_1000[:,0],hal_2d_1000[:,1])
