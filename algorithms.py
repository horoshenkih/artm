import numpy as np
from math import log

def nmf(collection, t, phi, theta, n_iter=10, algorithm='em',params={}):
    w, d = collection.shape
    losses_sliding_window = []
    sliding_window_size = 10
    adaptive_lda_params = {
        'alpha0': 10.,
        'beta0': 10.,
    }
    for iteration in range(n_iter):
        if params['verbose']:
            print "Iteration", iteration
        n_wt = np.zeros((w,t))
        n_dt = np.zeros((d,t))
        product = phi * theta
        if params['verbose']:
            print "All triples"
        for d_i in range(d):
            for w_i in range(w):
                Z = product[w_i,d_i]
                c_wd = collection[w_i,d_i]
                for t_i in range(t):
                    delta = c_wd*phi[w_i,t_i]*theta[t_i,d_i]/Z
                    n_wt[w_i,t_i] += delta
                    n_dt[d_i,t_i] += delta

        if params['verbose']:
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
            elif algorithm == 'adaptive_lda':
                beta = params['beta']
                for w_i in range(w):
                    increment = n_wt[w_i,t_i] + adaptive_lda_params['beta0'] * beta[w_i]
                    phi[w_i,t_i] = increment
                    n_t += increment
            else:
                print "not implemented"
                return
            # normalize
            for w_i in range(w):
                phi[w_i,t_i] /= n_t
        if params['verbose']:
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
            elif algorithm == 'adaptive_lda':
                alpha = params['alpha']
                for t_i in range(t):
                    increment = n_dt[d_i,t_i] + adaptive_lda_params['alpha0'] * alpha[t_i]
                    theta[t_i,d_i] = increment
                    n_d += increment
            else:
                print "not implemented"
                return
            # normalize
            for t_i in range(t):
                theta[t_i,d_i] /= n_d

        # find losses
        L = 0.
        for d_i in range(d):
            for w_i in range(w):
                s = 0.
                for t_i in range(t):
                    s += phi[w_i, t_i]*theta[t_i, d_i]
                L += collection[w_i, d_i] * log(s)
        loss_phi = 0.
        if algorithm in ['lda', 'adaptive_lda']:
            beta = params['beta']
            for w_i in range(w):
                for t_i in range(t):
                    loss_phi += beta[w_i]*log(phi[w_i, t_i])
        loss_theta = 0.
        if algorithm in ['lda', 'adaptive_lda']:
            alpha = params['alpha']
            for d_i in range(d):
                for t_i in range(t):
                    loss_theta += alpha[t_i]*log(theta[t_i, d_i])
        if params['verbose']:
            print "Likelihood:", L
            if algorithm in ['lda', 'adaptive_lda']:
                print "Loss phi:", loss_phi
                print "Loss theta:", loss_theta

        if algorithm == 'adaptive_lda':
            adaptive_lda_params['alpha0'] = params['gamma'] * L / loss_theta
            adaptive_lda_params['beta0'] = params['gamma'] * L / loss_phi

        if params['use_early_stopping']:
            current_loss = 0.
            if algorithm == 'em':
                current_loss = L
            elif algorithm == 'lda':
                current_loss = L + loss_theta + loss_phi
            elif algorithm == 'adaptive_lda':
                current_loss = L + adaptive_lda_params['alpha0'] * loss_theta + adaptive_lda_params['beta0'] * loss_phi

            if iteration < 2 * sliding_window_size:
                losses_sliding_window.append(current_loss)
            else:
                # split window and calc errors
                early, later = losses_sliding_window[:5], losses_sliding_window[5:]
                early_loss = sum(early) / len(early)
                later_loss = sum(later) / len(later)
                if params['verbose']:
                    print "early stopping data"
                    print "early loss:", early_loss
                    print "later loss:", later_loss
                    print "/ early stopping data"
                if later_loss < early_loss: # and iteration >= 20:
                    print "Took iterations:", iteration
                    if algorithm == 'adaptive_lda':
                        print adaptive_lda_params
                    return phi, theta
                # add new loss
                losses_sliding_window.pop(0)
                losses_sliding_window.append(current_loss)
    print "Took iterations:", iteration
    if algorithm == 'adaptive_lda':
        print adaptive_lda_params
    return phi, theta
