import numpy as np
import networkx as nx
from tqdm import tqdm
from functools import partial
from ..agents.agent import Agent
from .utils import RK4

class Network:
    def __init__(self, LDs, U_lims, etakappas, edges, dt, AgentType=Agent):
        self.n = len(U_lims)
        self.agents = []
        
        for i in range(self.n):
            self.agents.append(AgentType(i, LDs[i], U_lims[i], etakappas[i], dt))
         
        self.DG = nx.DiGraph()
        self.DG.add_weighted_edges_from(edges)  
        
        self.M = ((nx.adjacency_matrix(self.DG)).todense()).T

    def get_weights(self, xs, args):
        A = []
        weights = np.zeros((self.n, self.n))
        for i in range(self.n):
            A_i = []
            for j in range(self.n):
                if self.M[i,j] == 1:
                    a_ij = self.agents[i].DynEqs.Lg_jLf_i(xs,i,j)
                    A_i.append(a_ij)
                    weights[i,j] = np.linalg.norm(a_ij, ord=1)
                else:
                    A_i.append(None)
            A.append(A_i)

        return weights, A
    
    def get_neighbor_responsibility(self, i, N_in):
        c_bar_in = 0
        for j in N_in:
            if i in self.agents[j].responsibility:
                c_bar_in += self.agents[j].responsibility[i]

        return c_bar_in
    

    @staticmethod    
    def isNotSafe(delta):
        return delta < 0
    
    @staticmethod
    def isConstrained(esp):
        return esp > 0

    def collaborate(self, xs, maxiter, args=None):
        for agent in self.agents:
            agent.clear_constraints()
        
        viable = [False]*self.n
        first_round = [True]*self.n
        tau = 1

        weights, A = self.get_weights(xs, args)
        
        while (not all(viable)) and (tau <= maxiter):
            # SEND REQUESTS
            for i, agent_i in enumerate(self.agents):
                N_in = list(self.DG.predecessors(i))
                agent_i.updateMaxCapability(xs, N_in, args)
                c_bar_i = agent_i.capability
                c_bar_in = self.get_neighbor_responsibility(i, N_in)
 
                delta_i = c_bar_i - c_bar_in

                if len(delta_i) == 576:
                    print('c_bar_i', c_bar_i.shape)
                    print('c_bar_in', c_bar_in.shape)


                if self.isNotSafe(delta_i):                
                    available_N_in = list(filter(lambda j: j not in agent_i.N_in_constrained, N_in))
                    
                    if not available_N_in:
                        agent_i.clear_constrained()
                        available_N_in = N_in

                    w_tot = np.sum(weights[i, available_N_in])
                    
                    # Send requests
                    for j in available_N_in:
                        agent_j = self.agents[j]
                        delta_ij = delta_i*weights[i,j]/w_tot
                        agent_j.recieveRequest((i, delta_ij, A[i][j]))

                    if first_round[i]:
                        first_round[i] = False
                        
            # PROCESS PREQUESTS
            for i, agent_i in enumerate(self.agents):
                if agent_i.requests:
                    # print("agent {}".format(i))
                    agent_i.processRequests()
            
            # Update constrained neighbors and clear requests
            for i, agent_i in enumerate(self.agents):
                not_adjusted = []
                for k, eps_ki in agent_i.responses:
                    if self.isConstrained(eps_ki):
                        self.agents[k].N_in_constrained.append(i)
                        not_adjusted.append(False)
                    else:
                        not_adjusted.append(True)
                viable[i] = all(not_adjusted)
                
                agent_i.clear_responses()
                agent_i.clear_requests()
            
            tau += 1

        for agent in self.agents:
            agent.clear_responsibility()

        if tau == maxiter:
            print("Maximum iterations reached")

        return viable
    
    def f(self, x):
        return np.vstack([agent_i.DynEqs.f(x,i) for i, agent_i in enumerate(self.agents)])
    
    def g(self, x):
        gs = [agent_i.DynEqs.g(x,i) for i, agent_i in enumerate(self.agents)]

        hs, ws = [0], [0]
        for g in gs:
            if g.shape == (1,):
                h, w = 1, 1
            else:
                h, w = g.shape
            hs.append(h)
            ws.append(w)

        g_net = np.zeros((sum(hs), sum(ws)))

        for i, g in enumerate(gs):
            g_net[sum(hs[:i+1]):sum(hs[:i+2]), sum(ws[:i+1]):sum(ws[:i+2])] = g

        return g_net
    
    def u(self, x, args=None, safe=True):
        u_c = [agent_i.feedBackControl(x) for agent_i in self.agents]

        if safe:
            u_s = np.vstack([agent_i.getSafeControl(x, u_c[i], args) for i, agent_i in enumerate(self.agents)])
        else:
            u_s = np.vstack(u_c)
        return u_s

    def dynamics(self, x, u):
        return self.f(x) + self.g(x)@u

    def simulate(self, x0s, t0, tf, dt, collaborate=True, isSafe=True):
        numsamples = int((tf-t0)/dt)
        ts = np.linspace(t0, tf, numsamples)

        dims_x = [len(x0) for x0 in x0s]
        dims_u = [agent.num_inputs for agent in self.agents]
        N, M = sum(dims_x), sum(dims_u)
        xs = np.zeros((numsamples, N))
        us = np.zeros((numsamples, M))

        xs[0,:] = np.vstack(x0s).flatten()

        for k in tqdm(range(1, numsamples)):
            x = [xs[k - 1, sum(dims_x[:i]):sum(dims_x[:i+1])] for i in range(len(dims_x))]
            if collaborate:
                viable = self.collaborate(x,maxiter=6)
            
            u_x = self.u(x, safe=isSafe)
            digital_control = partial(self.dynamics, u=u_x)
            results = RK4.step(digital_control, x, dt)
            
            xs[k, :] = results.flatten()
            us[k,:] = u_x.flatten()

        return xs, us, ts