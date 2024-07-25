import numpy as np

from .agent import Agent
from .utils import proj2hull, max_min_element

class Agent2D(Agent):
    def __init__(self, agent_id, DynEqs, U_lim, etakappa, dt):
        super().__init__(agent_id, DynEqs, U_lim, etakappa, dt)

    def updateMaxCapability(self, x, N_in, args, eta_kappa=None, ret_argmax=False):
        """ Computes and updates the maximum capability of this agent

        Args:
            x (list of numpy arrays): x[i] returns the state vector of agent i 
            N_in (list of integers): List of ids for neighbors influencing this agent
        """
        obs = args
        if not eta_kappa:
            eta = self.eta
            kappa = self.kappa
        else:
            eta, kappa = eta_kappa
        
        if not isinstance(self.U_bar, list):
            if self.U_bar is None:
                equations = self.U_lim
            else:
                equations = np.block([[self.U_lim], 
                                      [self.U_bar]])
            
            q = self.DynEqs.q(x, obs, self.id, eta, kappa)

            sol_vec = max_min_element(q, equations)
            maxmin = sol_vec[0]
            u_max = np.atleast_2d(sol_vec[1:]).T

        else:
            u_max = self.U_bar[0]
        
        self.capability = (self.DynEqs.c(x, obs, u_max, self.id, self.eta, self.kappa, N_in)).flatten()

        if ret_argmax:
            return u_max

    def getSafeControl(self, x, u_desired, args):
        obs = args
        if not isinstance(self.U_bar, list):
            self.U_psi = self.DynEqs.psi_1_eq(x, obs, self.id, self.eta)
            if self.U_bar is None:
                equations = np.block([[self.U_lim],
                                      [self.U_psi]])
            else:
                equations = np.block([[self.U_lim],
                                      [self.U_psi], 
                                      [self.U_bar]])
                
            try:
                u_safe = proj2hull(u_desired.flatten(), equations)
            except Exception as e:
                # print(equations)
                # print('No solution to safe control from neighbors')
                equations = np.block([[self.U_lim],
                                      [self.U_psi]])
                u_safe = proj2hull(u_desired.flatten(), equations)
                
                # raise e
        else:
            u_safe = self.U_bar[0]
        
        return np.atleast_2d(u_safe).reshape((2,1))
    
    def get_request_mat(self):
        As, bs = [], []
        for k, delta_ki, a_ki in self.requests:
            num_obs = len(delta_ki)
            if k not in self.responsibility:
                self.responsibility[k] = np.zeros((num_obs))
            
            bs.append(np.atleast_2d(self.responsibility[k]+delta_ki).T)
            As.append([a_ki])
            
        ### Divide A by dt to convert the contraint from velocity to acceleration ##
        A = np.block(As)/self.dt
        b = np.vstack(bs)
        return A, b
    
    def compute_response(self, u_bb):
        if u_bb is not None:
            for k, delta_ki, a_ki in self.requests:
                eps_ki = ((a_ki@u_bb).flatten() + self.responsibility[k] + delta_ki).flatten()
                if (eps_ki < 0).any():
                    self.responses.append((k, - eps_ki))
                    self.responsibility[k] += (delta_ki - eps_ki)
                else:
                    self.responses.append((k, np.zeros(delta_ki.shape)))
                    self.responsibility[k] += delta_ki

            # Update U_bar to the single point of compromise
            self.U_bar = [u_bb]
        
        else:
            for k, delta_ki, a_ki in self.requests:
                self.responses.append((k, np.zeros(delta_ki.shape)))
                self.responsibility[k] += delta_ki