from utils import *
import numpy as np

w = 10
d = 5
t = 3

'''
alpha0 = 1
beta0 = 0.1
alpha = np.ones((1,t)).ravel() * alpha0
beta = np.ones((1,w)).ravel() * beta0

phi, theta, prod0, nd, prod = generate_all(w,d,t,alpha,beta,seed=31)
print nd
'''
A = np.matrix([[1,1],[3,3]])
B = np.matrix([[1,1],[3,3]])
print dist(A,B)
