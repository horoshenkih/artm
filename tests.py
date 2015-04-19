from utils import *
from algorithms import nmf
import numpy as np

import sys

w = 200
d = 100
t = 6

beta0 = 0.01  # const
n_iter = 300

results = open(sys.argv[1], "w")
for run in range(1):
    seeds = [30+run,40+run]
    for alpha0 in [0.01, 0.02, 0.05]:
    #for alpha0 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2,1.4, 1.6, 1.8, 2.]:
        alpha = np.ones((1,t)).ravel() * alpha0
        beta = np.ones((1,w)).ravel() * beta0

        phi0, theta0, prod0, nd, collection = generate_all(w,d,t,alpha,beta,seed=seeds[0])

        seed=seeds[1]
        phi1 = generate_phi(w,t,beta,seed=seed)
        theta1 = generate_theta(d,t,alpha,seed=seed)

        params = {
            'alpha': alpha,
            'beta': beta,
            'use_early_stopping': False,
            'verbose': False,
            'gamma': .5,  # adaptive LDA, 0.5 <=> 1 / n_regularizers
        }

        print "Alpha0:", alpha0
        for algorithm in ['em','lda']:
            phi, theta = nmf(collection, t, phi1, theta1, algorithm=algorithm, n_iter=n_iter, params=params)

            print "Algorithm:", algorithm
            print "D(phi, phi0):", dist(phi0, phi)
            print "D(theta, theta0)", dist(theta0, theta)
            print "D(prod, prod0)", dist (prod0, phi * theta)
            results.write("\t".join(map(str,(run, alpha0, algorithm, 'phi', dist(phi0,phi)))))
            results.write("\n")
            results.write("\t".join(map(str,(run, alpha0, algorithm, 'theta', dist(theta0,theta)))))
            results.write("\n")
            results.write("\t".join(map(str,(run, alpha0, algorithm, 'prod', dist(prod0,phi*theta)))))
            results.write("\n")

        # several versions of adaptive LDA
        for gamma in [0.1, 0.5, 1.]:
            params['gamma'] = gamma
            phi, theta = nmf(collection, t, phi1, theta1, algorithm='adaptive_lda', n_iter=n_iter, params=params)

            algorithm = 'adaptive_lda_'+str(gamma)
            print "Algorithm: adaptive lda, gamma =", gamma
            print "D(phi, phi0):", dist(phi0, phi)
            print "D(theta, theta0)", dist(theta0, theta)
            print "D(prod, prod0)", dist (prod0, phi * theta)
            results.write("\t".join(map(str,(run, alpha0, algorithm, 'phi', dist(phi0,phi)))))
            results.write("\n")
            results.write("\t".join(map(str,(run, alpha0, algorithm, 'theta', dist(theta0,theta)))))
            results.write("\n")
            results.write("\t".join(map(str,(run, alpha0, algorithm, 'prod', dist(prod0,phi*theta)))))
            results.write("\n")

        print
results.close()
