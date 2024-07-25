import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(filepath, xs, us, ts, obs0, taus, safe_dist, figsizes=[(6,8),(4,2)]):
    pxs = xs[:,::4]
    pys = xs[:,1::4]
    uxs = us[:,::2]
    uys = us[:,1::2]

    obs_x, obs_y = [],[]
    for o in obs0:
        x, y = o
        obs_x.append(x*np.ones((len(xs),1)))
        obs_y.append(y*np.ones((len(xs),1)))

    obs_x, obs_y = np.hstack(obs_x), np.hstack(obs_y)

    view = 7
    figsizes = [(6,8), (5,2.5)]
    xlims = [(-21,27), ()]
    #------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure(figsize=figsizes[0])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-23.5, 21), ylim=(-view, view))

    for o in obs0:
        px, py = o
        circle = plt.Circle((px , py), safe_dist, color='red', alpha=0.5)
        ax.add_patch(circle)

    plt.plot(pxs, pys, '--', lw=2)

    n = uxs.shape[1]

    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    if n <= 10:
        for i, c in enumerate(colors[:n]):
            plt.plot(pxs[0,i], pys[0,i], 'o', color=c, markersize=8)
            plt.plot(pxs[-1,i], pys[-1,i], '*', color=c, markersize=8)

    plt.tight_layout()
    plt.savefig(filepath + 'figures/formation_trajectory.png', facecolor='white', dpi=500)

    legend = [i for i in range(n)]

    ax_label_size = 16
    plt.figure(figsize=(5,2.5))
    plt.plot(ts[1:], uxs[1:,:])
    plt.legend(legend, loc='lower right')
    plt.xlabel('t')
    plt.ylabel(r'$u_s^x$', size=ax_label_size)
    # plt.xlim(0.00001, 21)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath + 'figures/uxs.png', facecolor='white', dpi=300)

    plt.figure(figsize=(5,2.5))
    plt.plot(ts[1:], uys[1:,:])
    plt.legend(legend, loc='lower right')
    plt.xlabel('t')
    plt.ylabel(r'$u_s^y$', size=ax_label_size)
    # plt.xlim(0.00001, 21)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath + 'figures/uys.png', facecolor='white', dpi=300)

    plt.figure(figsize=(5,2.5))
    plt.plot(ts, taus)
    # plt.ylim(0, 2.1)
    # plt.xlim(0.00001, 21)
    plt.grid()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\tau$',size=16)
    # plt.title('Numer of Communication Rounds')
    plt.tight_layout()
    plt.savefig('figures/num_tau.png', facecolor='white', dpi=300)