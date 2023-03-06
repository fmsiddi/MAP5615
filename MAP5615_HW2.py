import numpy as np
import sympy
import matplotlib.pyplot as plt
plt.style.use('seaborn')

primes = list(sympy.primerange(2,200))

def vdc(n,b):
    digs = list(np.base_repr(n,b))
    sum = 0
    for i in range(len(digs)):
        sum += int(digs[i])/b**i
    return sum

def halton(n,s):
    return [vdc(n,i) for i in primes[:s]]

