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
    safe_dist = args.safe_dist
    m = [args.mass]*n
    r = [safe_dist]*n
    K = args.spring_constant*(np.ones((n,n)) - np.eye(n))
    R = args.rest_length*(np.ones((n,n)) - np.eye(n))
    B = args.dampen_constant*(np.ones((n,n)) - np.eye(n))

    if n ==  3:
        x0s = [np.array([-12,0,0,0]),np.array([-15,2,0,0]),np.array([-15,-2,0,0])]
    else:
        x0s = [R*np.random.normal(loc=np.array([-12,0,0,0]), scale=R) for _ in range(n)]
        for x0 in x0s:
            x0[2:] = np.zeros((2))

    # Define fully connected spring network
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i,j,1))
                
    
    # Define uniform dynamics and control limits
    dynamics = Formation2D(m, R, K, B, r)

    u_limx, u_limy = args.control_lim, args.control_lim

    U_lim = np.array([[1, 0, u_limx],
                    [-1, 0, u_limx],
                    [0, 1, u_limy],
                    [0, -1, u_limy]])

    LDs = [dynamics]*n
    U_lims = [U_lim]*n
    e_k = args.eta_kappa
    etakappas = [(e_k,e_k)]*n

    
    # Define stationary obstacle field
    numrows, numcols = 6, 4
    vspace, hspace = 4, 4
    shift = 2
    start = np.array([-6, -10.1])
    obs = []

    for row in range(numrows):
        for col in range(numcols):
            offsety = row*vspace
            offsetx = col*hspace
            if col % 2 == 0:
                offsety += shift
            
            obs.append(start+np.array([offsetx, offsety]))

    dt = args.dt

    formation_network = Formation2DNetwork(LDs, U_lims, etakappas, edges, dt)

    t0, tf = 0, args.sim_length

    print('Simulating system...')

    # Set the constant control signal applied to the first agent
    leader_pull  = np.array([[14], [1]])

    xs, us, ts = formation_network.simulate(x0s, obs, t0, tf, dt, collaborate=True, isSafe=True, leader_pull=leader_pull)

    print('Plotting trajectories...')
    plot_trajectories(args.path, xs, us, ts, obs, safe_dist)

    return xs, us, ts, obs
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--safe-dist', type=float, default=1)
    parser.add_argument('--mass', type=float, default=.5)
    parser.add_argument('--spring-constant', type=float, default=3)
    parser.add_argument('--dampen-constant', type=float, default=1)
    parser.add_argument('--rest-length', type=float, default=3)
    parser.add_argument('--control-lim', type=float, default=15)
    parser.add_argument('--eta-kappa', type=float, default=10)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--sim-length', type=float, default=13)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--animate', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    
    xs, us, ts, obs = test(args)

    if args.animate:
        print('Creating animation...')
        pxs = xs[:,::4]
        pys = xs[:,1::4]
        
        obs_x, obs_y = [],[]
        for o in obs:
            x, y = o
            obs_x.append(x*np.ones((len(xs),1)))
            obs_y.append(y*np.ones((len(xs),1)))

        obs_x, obs_y = np.hstack(obs_x), np.hstack(obs_y)

        bounds = [-4, 4, -4, 4]
        view = 10

        #------------------------------------------------------------
        # set up figure and animation
        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-2*view, 2*view), ylim=(-view, view))

        # particles holds the locations of the particles
        particles, = ax.plot([], [], 'bo', ms=6)
        obsticles, = ax.plot([], [], 'ro', ms=6)

        # rect is the box edge
        rect = plt.Rectangle(bounds[::2],
                            bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            ec='none', lw=2, fc='none')
        ax.add_patch(rect)
        
        for o in obs:
            px, py = o
            circle = plt.Circle((px , py), args.safe_dist, color='red', alpha=0.5)
            ax.add_patch(circle)

        def init():
            """initialize animation"""
            global rect
            particles.set_data([], [])
            rect.set_edgecolor('none')
            return particles, rect
        
        
        def animate(i):
            """perform animation step"""

            global rect, ax, fig        

            particles.set_data(pxs[i, :], pys[i, :])

            return particles, obsticles, rect

        ani = animation.FuncAnimation(fig, animate, frames=len(pxs), interval=10, blit=True, init_func=init)

        ani.save(args.path + 'animation.mp4', fps=int(1/args.dt), extra_args=['-vcodec', 'libx264'])

    print('Done.')