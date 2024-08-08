import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import animation

from formation.dynamics.formation2D import Formation2D
from formation.networks.formation_network import Formation2DNetwork
from plot import plot_trajectories


def test(args):
    np.random.seed(args.seed)

    n = args.num_agents
    A = np.ones((n,n)) - np.eye(n)
    
    safe_dist = args.safe_dist
    m = [args.mass]*n
    r = [safe_dist]*n
    K = args.spring_constant*(A)
    R = args.rest_length*(A)
    B = args.dampen_constant*(A)
    
    x0s = [args.rest_length*np.random.normal(loc=np.array([-5,0,0,0]), scale=1) for _ in range(n)]

    if n ==  3:
        x0s = [np.array([-12,0,0,0]),np.array([-15,2,0,0]),np.array([-15,-2,0,0])]
    else:
        for x0 in x0s:
            x0[2:] = np.zeros((2))

    # Define graph edges
    edges = []
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                edges.append((i,j,1))
    
    # Define uniform dynamics and control limits
    dynamics = Formation2D(m, R, K, B, r)

    u_limx, u_limy = args.control_lim, args.control_lim

    U_lim = np.array([[ 1,  0, u_limx],
                      [-1,  0, u_limx],
                      [ 0,  1, u_limy],
                      [ 0, -1, u_limy]])

    LDs = [dynamics]*n
    U_lims = [U_lim]*n
    e_k = args.eta_kappa
    etakappas = [(e_k,e_k)]*n

    obs0 = []
    numrows, numcols = 6, 4
    dynamic_obstacles = None
    # dynamic_obstacles = (numrows, numcols)
    vspace, hspace = 3.5, 3.5
    shift = 2
    start = np.array([-6, -10.1])

    for row in range(numrows):
        for col in range(numcols):
            offsety = row*vspace
            offsetx = col*hspace
            if col % 2 == 0:
                offsety += shift
            
            obs0.append(start+np.array([offsetx, offsety]))

    leader_pull  = np.array([[10], [0]])

    dt = args.dt

    formation_network = Formation2DNetwork(LDs, U_lims, etakappas, edges, dt)

    t0, tf = 0, args.sim_length

    print('Simulating system...')

    # Set the constant control signal applied to the first agent
    leader_pull  = np.array([[14], [1]])
    collab = True
    xs, us, os, ts, taus, times = formation_network.simulate(x0s, obs0, t0, tf, dt, collaborate=collab, isSafe=True, leader_pull=leader_pull, dynamic_obs=dynamic_obstacles)

    print('Plotting trajectories...')
    plot_trajectories(args.path, xs, us, ts, obs0, taus, safe_dist)

    return xs, us, os, ts, obs0, A
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-agents', type=int, default=8)
    parser.add_argument('--safe-dist', type=float, default=1)
    parser.add_argument('--mass', type=float, default=.5)
    parser.add_argument('--spring-constant', type=float, default=3)
    parser.add_argument('--dampen-constant', type=float, default=1)
    parser.add_argument('--rest-length', type=float, default=3)
    parser.add_argument('--control-lim', type=float, default=30)
    parser.add_argument('--eta-kappa', type=float, default=10)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--sim-length', type=float, default=25)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--animate', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    
    xs, us, os, ts, obs0, A = test(args)

    if args.animate:
        print('Creating animation...')
        n = args.num_agents
        pxs = xs[:,::4]
        pys = xs[:,1::4]
        obs_x = os[:,::2]
        obs_y = os[:,1::2]

        bounds = [-4, 4, -4, 4]
        view = 15
        draw_lines = True

        #------------------------------------------------------------
        # set up figure and animation
        fig = plt.figure(figsize=(6,8))
        # fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=True,
                            xlim=(-view, view), ylim=(-view, view))
        # ax.axis('off')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        # particles holds the locations of the particles
        particles, = ax.plot([], [], 'bo', ms=6)
        obsticles, = ax.plot([], [], 'ro', ms=25, alpha=0.5)
        lines = []
        for i in range(n):
            for j in range(n):
                if i > j and A[i,j] == 1:
                    line, = ax.plot([], [], 'k-', lw=.5)
                    lines.append((i,j,line))

        # rect is the box edge
        rect = plt.Rectangle(bounds[::2],
                            bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            ec='none', lw=2, fc='none')
        ax.add_patch(rect)

        def init():
            """initialize animation"""
            global rect
            particles.set_data([], [])
            obsticles.set_data([], [])
            rect.set_edgecolor('none')
            return particles, rect


        def animate(i):
            """perform animation step"""

            global rect, ax, fig        

            particles.set_data(pxs[i, :], pys[i, :])
            obsticles.set_data(obs_x[i,:], obs_y[i,:])
            for j, k, l in lines:
                lx = [pxs[i,j], pxs[i,k]]
                ly = [pys[i,j], pys[i,k]]
                l.set_data(lx,ly)

            return particles, obsticles, rect, 

        ani = animation.FuncAnimation(fig, animate, frames=len(pxs), interval=10, blit=True, init_func=init)

        ani.save(args.path + 'animations/animation.mp4', fps=int(1/args.dt), extra_args=['-vcodec', 'libx264'])
    print('Done.')