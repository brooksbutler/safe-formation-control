import numpy as np

from tqdm import tqdm
from functools import partial
from .network import Network
from .utils import RK4

from ..agents.agent2D import Agent2D


class Formation2DNetwork(Network):
    def __init__(self, LDs, U_lims, etakappas, edges, dt):
        super().__init__(LDs, U_lims, etakappas, edges, dt, Agent2D)

    def get_weights(self, xs, args):
        obs = args
        A = []
        weights = np.zeros((self.n, self.n))
        for i in range(self.n):
            A_i = []
            for j in range(self.n):
                if self.M[i,j] == 1:
                    a_ij = []
                    for po in obs:
                        a_ij.append([self.agents[i].DynEqs.Lg_jLf_i(xs, i, j, po)])
                    
                    a_ij = np.block(a_ij)
                    A_i.append(a_ij)
                    
                    weights[i,j] = np.average(np.sum(np.abs(a_ij), axis=1))
                else:
                    A_i.append(None)
            A.append(A_i)

        return weights, A
    
    def get_neighbor_responsibility(self, i, N_in):
        c_bar_in = np.zeros(self.agents[i].capability.shape)
        for j in N_in:
            if i in self.agents[j].responsibility:
                c_bar_in += self.agents[j].responsibility[i]

        return c_bar_in
    
    @staticmethod
    def isNotSafe(delta):
        return (delta < 0).any()
    
    @staticmethod
    def isConstrained(eps):
        return (eps > 0).any()
    
    def u(self, x, args=None, safe=True, leader_pull=None):
        u_c = [agent_i.feedBackControl(x) for agent_i in self.agents]
        if leader_pull is not None:
            for i in range(len(u_c)):
                u_c[i] = leader_pull
        if safe:
            u_s = np.vstack([agent_i.getSafeControl(x, u_c[i], args) for i, agent_i in enumerate(self.agents)])
        else:
            u_s = np.vstack(u_c)
        return u_s
    
    def dynamicObs(self, k, obs0, numrows, numcols):
        obs = []
        ind = 0
        for r in range(numrows):
            for c in range(numcols):
                obs.append(obs0[ind] + np.array([0, 2*np.sin(.25*np.pi*0.01*k)*((-1)**c)]))
                ind += 1
        return obs

    def simulate(self, x0s, obs0, t0, tf, dt, collaborate=True, isSafe=True, leader_pull=None, dynamic_obs=None):
        maxiter = 12
        numsamples = int((tf-t0)/dt)
        ts = np.linspace(t0, tf, numsamples)

        dims_x = [len(x0) for x0 in x0s]
        dims_u = [agent.num_inputs for agent in self.agents]
        dims_o = [len(o) for o in obs0]
        N, M = sum(dims_x), sum(dims_u)
        xs = np.zeros((numsamples, N))
        us = np.zeros((numsamples, M))
        os = np.zeros((numsamples, sum(dims_o))) 
        taus = np.zeros(numsamples)

        xs[0,:] = np.vstack(x0s).flatten()
        os[0,:] = np.vstack(obs0).flatten()

        for k in tqdm(range(1, numsamples)):
            x = [xs[k - 1, sum(dims_x[:i]):sum(dims_x[:i+1])] for i in range(len(dims_x))]
            ### Compute the current spring lengths for all agents ###
            for agent_i in self.agents:
                agent_i.DynEqs.compute_lengths(x)
            
            if dynamic_obs is not None:
                numrows, numcols = dynamic_obs
                obs = self.dynamicObs(k, obs0, numrows, numcols)
            else:
                obs = obs0

            if collaborate:
                viable, tau = self.collaborate(x,maxiter,obs)
                taus[k] = tau 
            
            u_x = self.u(x, args=obs, safe=isSafe, leader_pull=leader_pull)

            digital_control = partial(self.dynamics, u=u_x)
            results = RK4.step(digital_control, x, dt)
            
            xs[k, :] = results.flatten()
            us[k, :] = u_x.flatten()
            os[k, :] = np.vstack(obs).flatten()
           
        return xs, us, os, ts, taus