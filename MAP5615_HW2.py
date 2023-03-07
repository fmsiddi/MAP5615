import numpy as np
import sympy
import re
from tqdm import tqdm
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

def RANDU(n,x_0):
    x = x_0
    array = np.zeros(n)
    for i in range(n):
        x = 65539*x % (2**31)
        array[i] = x/(2**31)
    return array

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
# samples = RANDU(10000,1)
# num = len(samples)
# ax.scatter(samples[:num-2],samples[1:num-1],samples[2:num])
# plt.show()

def crit_vals(N,K,sig_values):
    p = np.zeros((N,N))
    p[0][0] = 1
    for n in range(1,N):
        p[0][n] = K**(-n)
    for n in range(1,N):
        p[n][n] = p[n-1][n-1]*(K-n)/K
    for j in tqdm(range(1,N)):
        for n in range(j+1,N):
            p[j,n] = p[j,n-1]*(j+1)/K + p[j-1][n-1]*(K-j)/K
    
    pr = np.zeros(N)
    for i in range(N):
        pr[i] = p[(N-1)-i][N-1]
        
    cvals = np.zeros((len(sig_values),2))
    s = 0
    j = 0
    for i in range(len(sig_values)):
        while s < sig_values[i]:
            s += pr[j]
            j += 1
        cvals[i] = [s,j-1]
        if s > .9999:
            break
    return cvals

# c_vals = crit_vals(20,100,[0.4,0.9,0.99])
# print(c_vals)


def digit_list(n):
    return list(map(int, re.findall('\d', str(n))))

def collision_test(X):
    urns = np.zeros((10,10,10,10,10))
    occup = np.zeros((10,10,10,10,10))
    for i in range(len(X)):
        w = digit_list(X[i])[1]
        x = digit_list(X[i])[2]
        y = digit_list(X[i])[3]
        z = digit_list(X[i])[4]
        a = digit_list(X[i])[5]
        if urns[w][x][y][z][a] != 0:
            occup[w][x][y][z][a] += 1
        urns[w][x][y][z][a] += 1 
    return np.sum(occup)

K = 100000
n = 20000
x_0 = np.random.randint(2**31)
collisions = collision_test(RANDU(n,1))
print('number of collisions after {0} throws into {1} urns and initial seed {2}: {3}'.format(n,K,x_0,collisions))
c_vals = crit_vals(n,K,[0.5,0.75,0.95,0.99])
print(c_vals)

# def p_jn(M,n):
#     p = np.zeros(M,n)
#     p[0,0] = 1
#     p[1:,:] = 0
#     for i in range(1:n):
#         p[0][i] = M**(-(i+1)+1)