import numpy as np

def nmf(collection, t, phi, theta, n_iter=10, algorithm='em',params=None,verbose=False):
    w, d = collection.shape
    for iteration in range(n_iter):
        if verbose:
            print "Iteration", iteration
        n_wt = np.zeros((w,t))
        n_dt = np.zeros((d,t))
        product = phi * theta
        if verbose:
            print "All triples"
        for d_i in range(d):
            for w_i in range(w):
                Z = product[w_i,d_i]
                for t_i in range(t):
                    delta = collection[w_i,d_i]*phi[w_i,t_i]*theta[t_i,d_i]/Z
                    n_wt[w_i,t_i] += delta
                    n_dt[d_i,t_i] += delta

        if verbose:
            print "Fill Phi"
        for t_i in range(t):
            n_t = 0.
            if algorithm == 'em':
                for w_i in range(w):
                    phi[w_i,t_i] = n_wt[w_i,t_i]
                    n_t += n_wt[w_i,t_i]
            elif algorithm == 'lda':
                beta = params['beta']
                for w_i in range(w):
                    increment = n_wt[w_i,t_i] + beta[w_i]
                    phi[w_i,t_i] = increment
                    n_t += increment
            else:
                print "not implemented"
                return
            # normalize
            for w_i in range(w):
                phi[w_i,t_i] /= n_t
        if verbose:
            print "Fill Theta"
        for d_i in range(d):
            n_d = 0.
            if algorithm == 'em':
                for t_i in range(t):
                    theta[t_i,d_i] = n_dt[d_i,t_i]
                    n_d += n_dt[d_i,t_i]
            elif algorithm == 'lda':
                alpha = params['alpha']
                for t_i in range(t):
                    increment = n_dt[d_i,t_i] + alpha[t_i]
                    theta[t_i,d_i] = increment
                    n_d += increment
            else:
                print "not implemented"
                return
            # normalize
            for t_i in range(t):
                theta[t_i,d_i] /= n_d
    return phi, theta
