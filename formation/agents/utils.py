import numpy as np
import quadprog
from scipy import optimize

def proj2hull(z, equations):
    G = np.eye(len(z), dtype=float)
    a = np.array(z, dtype=float)
    C = np.array(equations[:, :-1], dtype=float)
    b = np.array(-equations[:, -1], dtype=float)

    x, f, xu, itr, lag, act = quadprog.solve_qp(G, a, C.T, b, meq=0, factorized=True)
    
    return x

def proj_grad_alg(u, Q, q, proj, equations, h=1e-1, numiter=1000):
    for _ in range(numiter):
        grad_u = 2*Q@u + q.T
        u = (proj((u + h*grad_u).flatten(), equations)).T
    return u

def chebychev_center(A, b):
    halfspaces = np.block([A, b]) 
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
        (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((-halfspaces[:, :-1], norm_vector))
    b = halfspaces[:, -1:]
    sol_compromise = optimize.linprog(c, A_ub=A, b_ub=b, bounds=(None, 1e3))
    return sol_compromise.x

def max_min_element(q, equations):
    l = equations.shape[0]
    n, m = q.shape

    c = np.hstack((np.array([[1]]), np.zeros((1,m))))

    A = np.block([[np.zeros((l,1)), -equations[:,:-1]],
                  [np.ones((n,1)), -q]])
    
    b = np.vstack((equations[:,-1:], np.zeros((n,1))))

    sol = optimize.linprog(c, A_ub=A, b_ub=b)
    return sol.x

def closest_point(A_req, b_req, U_lim):
    num_inputs = U_lim.shape[1] - 1
    Q = np.block([[np.eye(num_inputs), -np.eye(num_inputs)],
                  [-np.eye(num_inputs), np.eye(num_inputs)]])
    
    a = np.zeros((2*num_inputs))
    
    C = np.block([[U_lim[:,:-1], np.zeros(U_lim[:,:-1].shape)],
                  [np.zeros(A_req.shape), A_req]])
    
    b = np.block([[-np.atleast_2d(U_lim[:,-1]).T],
                  [-b_req]]).flatten()
    
    x, f, xu, itr, lag, act = quadprog.solve_qp(Q, a, C.T, b, meq=0, factorized=True)
    return x[:num_inputs]