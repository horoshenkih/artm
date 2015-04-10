from utils import *
from algorithms import nmf
import numpy as np

w = 500
d = 200
t = 10

beta0 = 0.1  # const
for alpha0 in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2.]:
    alpha = np.ones((1,t)).ravel() * alpha0
    beta = np.ones((1,w)).ravel() * beta0

    phi0, theta0, prod0, nd, collection = generate_all(w,d,t,alpha,beta,seed=30)

    seed=40
    phi1 = generate_phi(w,t,beta,seed=seed)
    theta1 = generate_theta(d,t,alpha,seed=seed)

    n_iter = 200
    params = {
        'alpha': alpha,
        'beta': beta,
    }

    print "Alpha0:", alpha0
    for algorithm in ['adaptive_lda', 'lda']:
        phi, theta = nmf(collection, t, phi1, theta1, algorithm=algorithm, n_iter=n_iter, params=params, verbose=False)

        print "Algorithm:", algorithm
        print "D(phi, phi0):", dist(phi0, phi)
        print "D(theta, theta0)", dist(theta0, theta)
        print "D(prod, prod0)", dist (prod0, phi * theta)
