import numpy as np

# Lie derivatives for a given node i assuming the system is control affine
class Dynamics:
    def __init__(self):
        # Inlcude parameters needed to compute dynamics
        pass

    def h(self, x, i):
        # Returns a scalar
        raise NotImplementedError()

    def grad_h(self, x, i):
        # Returns a vector of shape 1 x N_i
        raise NotImplementedError()
    
    def f(self, x, i):
        # Returns a vector of shape N_i x 1
        raise NotImplementedError()

    def g(self, x, i):
        # Returns a vector of shape N_i x M_i
        raise NotImplementedError()
    
    def Lf_i(self, x, i):
        # Returns a scalar
        raise NotImplementedError()
    
    def Lg_i(self, x, i):
        # Returns a vector of shape 1 x M_i
        raise NotImplementedError()
    
    def Lf_jLf_i(self, x, i, j):
        # Returns a scalar
        raise NotImplementedError()
    
    def Lg_jLf_i(self, x, i, j):
        # Returns a vector of shape 1 x M_j
        raise NotImplementedError()
    
    def Lg_iLf_i(self, x, i):
        # Returns a vector of shape 1 x M_i
        raise NotImplementedError()
    
    def Lf_iLg_i(self, x, i):
        # Returns a vector of shape M_i x 1
        raise NotImplementedError()
    
    def L2f_i(self, x, i):
        # Returns a scalar
        raise NotImplementedError()
    
    def L2g_i(self, x, i):
        # Returns a matrix of shape M_i x M_i
        raise NotImplementedError()
    
    def q(self, x, i, eta, kappa):
        # Returns a matrix of shape 1 x M_i
        return self.Lf_iLg_i(x, i,).T + self.Lg_iLf_i(x,i) + (eta+kappa)*self.Lg_i(x,i)
    
    def b(self, x, i, eta, kappa, N_in):
        # Returns a scalar
        sub_tot = sum([self.Lf_jLf_i(x, i, j) for j in N_in])
        return sub_tot + self.L2f_i(x,i) + (eta+kappa)*self.Lf_i(x,i) + eta*kappa*self.h(x,i)
    
    def c(self,x,u,i, eta, kappa, N_in):
        # Returns a scalar
        return u.T@self.L2g_i(x,i)@u + self.q(x, i, eta, kappa)@u + self.b(x, i, eta, kappa, N_in)
    
    def psi_1_eq(self, x, i, eta):
        A = self.Lg_i(x,i)
        b = self.Lf_i(x,i) + eta*self.h(x,i)
        return np.hstack([A, np.atleast_2d(b)])