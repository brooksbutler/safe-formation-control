import numpy as np

from .base_dynamics import Dynamics

class Formation2D(Dynamics):
    def __init__(self, m, R, K, B, r):
        """_summary_

        Args:
            m (_type_): Mass of each object 
            R (_type_): Relaxed length of springs
            K (_type_): Spring constants
            B (_type_): Dampening coeficients
            r (_type_): Safety radi for each object
        """
        self.m = m 
        self.R = R
        self.K = K
        self.B = B
        self.r = r
        self.L = None
        self.L_3o2 = None
        self.S = None
        self.sinTh = None
        self.cosTh = None
        self.alpha = 10

    #### THIS SHOULD BE CALLED EVERY TIME-STEP DYNAMICS ARE COMPUTED #### 
    def compute_lengths(self, x):
        px = np.array([[x[i][0] for i in range(len(x))]])
        py = np.array([[x[i][1] for i in range(len(x))]])

        diff_x = px - px.T
        diff_y = py - py.T

        self.L = np.sqrt(diff_x**2 + diff_y**2)
        self.L_3o2 = np.multiply(self.L, self.L**2)
        self.S = self.L - self.R

        divide = self.L > 0
        self.sinTh = np.divide(diff_x, self.L, where = divide)
        self.cosTh = np.divide(diff_y, self.L, where = divide)

    #####################################################################
    
    def a_x(self, x, i):
        return 1/self.m[i]*(np.sum(np.multiply(np.multiply(self.K[i,:],self.S[i,:]),self.sinTh[i,:])) - np.sum(self.B[i,:])*x[i][2])
    
    def a_y(self, x, i):
        return 1/self.m[i]*(np.sum(np.multiply(np.multiply(self.K[i,:],self.S[i,:]),self.cosTh[i,:])) - np.sum(self.B[i,:])*x[i][3])
    
    def da_dp_i(self, x, i, resp_to):
        px_i, py_i, _, _ = x[i].flatten()
        tot_sum = 0
        for j in range(len(self.m)):
            if j != i:
                px_j, py_j, _, _ = x[j].flatten()
                if resp_to == 'x':
                    d_dp_i = (py_i - py_j)**2/self.L_3o2[i,j]
                elif resp_to == 'y':
                    d_dp_i = (px_i - px_j)**2/self.L_3o2[i,j]
                
                tot_sum += self.K[i,j]*(1 - self.R[i,j]*d_dp_i)
        
        return 1/self.m[i]*tot_sum 
    
    def dayx_dpxy_i(self, x, i):
        tot_sum = 0
        px_i, py_i, _, _ = x[i].flatten()
        for j in range(len(self.m)):
            if j != i:
                px_j, py_j, _, _ = x[j].flatten()
                d_dp_i = -((px_i - px_j)*(py_i - py_j))/self.L_3o2[i,j]
                tot_sum += self.K[i,j]*(-self.R[i,j]*d_dp_i)
        
        return 1/self.m[i]*tot_sum 
    
    
    def da_dp_j(self, x, i, j, resp_to):
        px_i, py_i, _, _ = x[i].flatten()
        px_j, py_j, _, _ = x[j].flatten()
        if resp_to == 'x':
            d_dp_j = -(py_i - py_j)**2/self.L_3o2[i,j]
        elif resp_to == 'y':
            d_dp_j = -(px_i - px_j)**2/self.L_3o2[i,j]

        return -1/self.m[i]*(1 + self.R[i,j]*d_dp_j)
    
    def dayx_dpxy_j(self, x, i, j):
        px_i, py_i, _, _ = x[i].flatten()
        px_j, py_j, _, _ = x[j].flatten()
        d_dp_j = ((px_i - px_j)*(py_i - py_j))/self.L_3o2[i,j]

        return -1/self.m[i]*self.R[i,j]*d_dp_j
     
    def da_dv(self, i):
        return -1/self.m[i]*np.sum(self.B[i,:])
    
    #####################################################################
    
    def h(self, x, i, po):
        px, py, vx, vy = x[i].flatten()
        pox, poy = po.flatten() 

        return 2*(vx*(px - pox) + vy*(py - poy)) + self.alpha*((pox - px)**2 + (poy - py)**2 - self.r[i]**2)
    
    def grad_h(self, x, i, po):
        px, py, vx, vy = x[i].flatten()
        pox, poy = po.flatten()
        return 2*np.array([[vx+self.alpha*(px-pox), vy+self.alpha*(py-poy), px - pox, py - poy]])
    
    def f(self, x, i):
        return np.array([[x[i][2], x[i][3], self.a_x(x,i), self.a_y(x,i)]]).T

    def g(self, x, i):
        mi_inv = 1/self.m[i]
        g = np.array([[0,0],[0,0],[mi_inv,0],[0,mi_inv]])
        return g
    
    def Lf_i(self, x, i, po):
        return self.grad_h(x, i, po)@self.f(x,i)
    
    def Lg_i(self, x, i, po):
        return self.grad_h(x, i, po)@self.g(x,i)
    
    #####################################################################

    def gradi_Lf(self, x, i, po):
        px, py, vx, vy = x[i].flatten()
        pox, poy = po.flatten()
        
        dLf_dpx_i = 2*vx + self.a_x(x, i) + self.da_dp_i(x, i, 'x')*(px - pox) + self.dayx_dpxy_i(x, i)*(py - poy)
        dLf_dpy_i = 2*vy + self.a_y(x, i) + self.da_dp_i(x, i, 'y')*(py - poy) + self.dayx_dpxy_i(x, i)*(px - pox)
        dLf_dvx_i = 4*vx + self.alpha*(px-pox) + self.da_dv(i)*(px-pox)
        dLf_dvy_i = 4*vy + self.alpha*(py-poy) + self.da_dv(i)*(py-poy)

        return np.array([[dLf_dpx_i, dLf_dpy_i, dLf_dvx_i, dLf_dvy_i]])
    
    def gradj_Lf(self, x, i, j, po):
        px, py, _, _ = x[i].flatten()
        pox, poy = po.flatten()

        dLf_dpx_j = self.da_dp_j(x, i, j, 'x')*(px - pox) + self.dayx_dpxy_j(x, i, j)*(py - poy)
        dLf_dpy_j = self.da_dp_j(x, i, j, 'y')*(py - poy) + self.dayx_dpxy_j(x, i, j)*(px - pox)

        return  np.array([[dLf_dpx_j, dLf_dpy_j, 0, 0]])
    
    def gradi_Lg(self):
        return np.array([[2, 0, 0, 0],
                         [0, 2, 0, 0]])
    
    #####################################################################
    
    def Lf_jLf_i(self, x, i, j, po):
        return self.gradj_Lf(x,i,j,po)@self.f(x, j)
    
    def Lg_jLf_i(self, x, i, j, po):
        #### THIS ASSUMES NEIGHBORS CAN DIRECTLY CONTROL VELOCITY #######
        gj = np.array([[1,0],[0,1],[0,0],[0,0]])
        #################################################################
        return self.gradj_Lf(x,i,j,po)@gj
    
    def Lf_iLg_i(self, x, i):
        return self.gradi_Lg()@self.f(x, i)

    
    def Lg_iLf_i(self, x, i, po):
        return self.gradi_Lf(x, i, po)@self.g(x, i)
    
    def L2f_i(self, x, i, po):
        return self.gradi_Lf(x, i, po)@self.f(x, i)
    
    def L2g_i(self, x, i):
        return np.array([[0,0],[0,0]])
    
    def q(self, x, obs, i, eta, kappa):
        # Returns a matrix of shape O x M_i
        q = []
        for po in obs:
            q.append(self.Lf_iLg_i(x, i).T + self.Lg_iLf_i(x, i, po) + (eta+kappa)*self.Lg_i(x, i, po))
        return np.vstack(q)
    
    def b(self, x, obs, i, eta, kappa, N_in):
        # Returns a scalar
        b = []
        for po in obs:
            sub_tot = sum([self.Lf_jLf_i(x, i, j, po) for j in N_in])
            b.append([(sub_tot + self.L2f_i(x, i, po) + (eta+kappa)*self.Lf_i(x, i, po) + eta*kappa*self.h(x, i, po)).flatten()])

        return np.vstack(b)
    
    def c(self, x, obs, u, i, eta, kappa, N_in):
        # Returns a matrix of shape O X 1 
        return  self.q(x, obs, i, eta, kappa)@u + self.b(x, obs, i, eta, kappa, N_in)
    
    def psi_1_eq(self, x, obs, i, eta):
        A, b = [], []
        for po in obs:
            A.append(self.Lg_i(x, i, po))
            b.append(np.atleast_2d(self.Lf_i(x, i, po) + eta*self.h(x, i, po)).T)

        return np.hstack([np.vstack(A), np.vstack(b)])