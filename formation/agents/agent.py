import numpy as np
from scipy import optimize

from .utils import proj2hull, proj_grad_alg, chebychev_center, closest_point

# Basic agent properties
class Agent:
    def __init__(self, agent_id, DynEqs, U_lim, etakappa, dt):
        self.id = agent_id
        self.DynEqs = DynEqs
        self.U_lim = U_lim
        self.U_bar = None
        self.U_psi = None
        self.num_inputs = U_lim.shape[1] - 1 # subtract 1 due to extra collum from b
        self.eta, self.kappa = etakappa
        self.dt = dt
        
        self.responsibility = {}
        self.capability = 0

        self.requests = []
        self.responses = []
        self.N_in_constrained = []
    
    
    def updateMaxCapability(self, x, N_in, args = None, eta_kappa=None, ret_argmax=False):
        """ Computes and updates the maximum capability of this agent

        Args:
            x (list of numpy arrays): x[i] returns the state vector of agent i 
            N_in (list of integers): List of ids for neighbors influencing this agent
        """
        if not eta_kappa:
            eta = self.eta
            kappa = self.kappa
        else:
            eta, kappa = eta_kappa
        
        if not isinstance(self.U_bar, list):
            Q = self.DynEqs.L2g_i(x, self.id)
            Q = .5*(Q.T + Q)
            q = self.DynEqs.q(x, self.id, eta, kappa)
            if self.U_bar is None:
                equations = self.U_lim
            else:
                equations = np.block([[self.U_lim], 
                                      [self.U_bar]])

            u0 = np.zeros((self.num_inputs, 1))
            u_max  = proj_grad_alg(u0, Q, q, proj2hull, equations)
        else:
            u_max = self.U_bar[0]

        self.capability = (self.DynEqs.c(x, u_max, self.id, self.eta, self.kappa, N_in)).flatten()

        if ret_argmax:
            return u_max

    def getSafeControl(self, x, u_desired, args=None):
        if not isinstance(self.U_bar, list):
            self.U_psi = self.DynEqs.psi_1_eq(x, self.id, self.eta)
            if self.U_bar is None:
                equations = np.block([[self.U_lim],
                                      [self.U_psi]])
            else:
                equations = np.block([[self.U_lim],
                                      [self.U_psi], 
                                      [self.U_bar]])
           
            u_safe = proj2hull(u_desired.flatten(), equations)

        else:
            u_safe = self.U_bar[0]
            
        return u_safe
    
    def recieveRequest(self, request):
        self.requests.append(request)

    def clear_requests(self):
        self.requests = []

    def clear_responses(self):
        self.responses = []
    
    def clear_responsibility(self):
        self.responsibility = {}
    
    def clear_constrained(self):
        self.N_in_constrained = []
    
    def clear_constraints(self):
        self.U_bar = None
        
    def get_request_mat(self):
        As, bs = [], []
        for k, delta_ki, a_ki in self.requests:
            if k not in self.responsibility:
                self.responsibility[k] = 0
            
            bs.append([self.responsibility[k]+delta_ki])
            As.append([a_ki])
        return np.block(As), np.block(bs)
    
    def compute_response(self, u_bb):
        if u_bb is not None:
            for k, delta_ki, a_ki in self.requests:
                esp_ki = (a_ki@u_bb + self.responsibility[k] + delta_ki).flatten()
                if esp_ki < 0:
                    self.responses.append((k, -esp_ki))
                    self.responsibility[k] += delta_ki - esp_ki
                else:
                    self.responses.append((k, 0))
                    self.responsibility[k] += delta_ki
            # Update U_bar to the single point of compromise
            self.U_bar = [u_bb]
        
        else:
            for k, delta_ki, a_ki in self.requests:
                self.responses.append((k, 0))
                self.responsibility[k] += delta_ki

    def processRequests(self):
        u_bb = None
        # Check if the interseciton of halfspaces is non-empty using LP
        c = np.ones(self.num_inputs)
        A_req, b_req = self.get_request_mat()
        
        sol = optimize.linprog(c, -A_req, b_req)

        # If the intersection is nonempty
        if sol.success:
            # Check if the requests intersect with the control contraints
            A = np.block([[A_req],
                          [self.U_lim[:, :-1]]])
            b = np.block([[b_req],
                          [self.U_lim[:, -1:]]])
            sol_constrained = optimize.linprog(c, -A, b)
            
            # If there is an intersection, then add the constraints to self.U_bar
            if sol_constrained.success:
                # print('intersection is nonempty')
                self.U_bar = np.block([A_req, b_req])

            # If not, then compute the closest point between the two convex spaces
            else:
                # print('intersection is empty')
                if self.num_inputs > 2:
                    raise NotImplementedError()
                elif self.num_inputs == 2:
                    # closest_lim, _ = closest_point_2D(A_req, b_req, self.U_lim)
                    closest_lim = closest_point(A_req, b_req, self.U_lim)
                    u_bb = np.atleast_2d(closest_lim).T
                elif self.num_inputs == 1:
                    u_bb = np.atleast_2d(self.U_lim[-1,-1])

        # If there is no feasible point
        else:
            # Compute the chebychev center of the infeasible space
            u_compromise = chebychev_center(-A_req, -b_req)

            # Project sol_conpromise.x onto self.U_lim to comupte u_bb
            u_bb = np.atleast_2d(proj2hull(u_compromise[:-1], self.U_lim)).T
             
        # If a compromise is needed, compute the adjustments needed to each request and update responsibility taken
        self.compute_response(u_bb)

    def feedBackControl(self, x):
        # This should be overwritten in child agent classes with different controllers
        return np.zeros((self.num_inputs, 1))