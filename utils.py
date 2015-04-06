import numpy as np

def generate_phi(w, t, beta, seed=42):
    np.random.seed(seed)
    return np.matrix(np.random.dirichlet(beta, t).transpose())

def generate_theta(d, t, alpha, seed=42):
    np.random.seed(seed)
    return np.matrix(np.random.dirichlet(alpha, d).transpose())

def generate_nd(d,seed=42):
    np.random.seed(seed)
    return np.random.randint(600, 1000, d)

def generate_all(w,d,t,alpha,beta,seed=42):
    theta = generate_theta(d,t,alpha,seed)
    phi = generate_phi(w,t,beta,seed)
    prod0 = phi * theta
    nd = generate_nd(d,seed)
    prod = np.multiply(prod0, nd)

    return (phi, theta, prod0, nd, prod)

def dist(A,B):
    # Hellinger's distance between matrices
    X = np.power(np.sqrt(A) - np.sqrt(B),2)
    X = np.sum(X, axis=0)
    X = np.sqrt(0.5 * X)
    m = X.shape[1]
    X = np.sum(X, axis=1)
    return X[0,0]
