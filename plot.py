import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(filepath, xs, us, ts, obs, safe_dist):
    pxs = xs[:,::4]
    pys = xs[:,1::4]
    uxs = us[:,::2]
    uys = us[:,1::2]

    obs_x, obs_y = [],[]
    for o in obs:
        x, y = o
        obs_x.append(x*np.ones((len(xs),1)))
        obs_y.append(y*np.ones((len(xs),1)))

    obs_x, obs_y = np.hstack(obs_x), np.hstack(obs_y)

    view = 4.9
    #------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure(figsize=(6,8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-16, 17), ylim=(-view, view))

    for o in obs:
        px, py = o
        circle = plt.Circle((px , py), safe_dist, color='red', alpha=0.5)
        ax.add_patch(circle)

    plt.plot(pxs, pys, '--', lw=2)
    
    n = uxs.shape[1]
    
    if n == 3:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, c in enumerate(colors):
            plt.plot(pxs[0,i], pys[0,i], 'o', color=c, markersize=8)
            plt.plot(pxs[-1,i], pys[-1,i], '*', color=c, markersize=8)
    
    plt.tight_layout()
    plt.savefig(filepath + 'formation_trajectory_obs_field.png', facecolor='white', dpi=500)

    legend = [i for i in range(n)]

    ax_label_size = 16
    plt.figure(figsize=(4,2))
    plt.plot(ts[1:], uxs[1:,:])
    plt.legend(legend, loc='lower left')
    plt.xlabel('t')
    plt.ylabel(r'$u_s^x$', size=ax_label_size)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath + 'uxs.png', facecolor='white', dpi=300)

    plt.figure(figsize=(4,2))
    plt.plot(ts[1:], uys[1:,:])
    plt.legend(legend, loc='lower left')
    plt.xlabel('t')
    plt.ylabel(r'$u_s^y$', size=ax_label_size)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath + 'uys.png', facecolor='white', dpi=300)