import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(filename, x, u, T):
    f = plt.figure(figsize=(12, 6))

    # Plot position
    ax2 = f.add_subplot(311)
    x1 = x[0, :]
    plt.plot(x1)
    plt.ylabel(r"$p$ $(m)$", fontsize=14)
    plt.yticks([np.min(x1), 0, np.max(x1)])
    plt.ylim([np.min(x1) - 0.1, np.max(x1) + 0.1])
    plt.xlim([0, T])
    plt.grid()

    # Plot velocity
    ax3 = plt.subplot(3, 1, 2)
    x2 = x[1, :]
    plt.plot(x2)
    plt.yticks([np.min(x2), 0, np.max(x2)])
    plt.ylim([np.min(x2) - 0.1, np.max(x2) + 0.1])
    plt.xlim([0, T])
    plt.ylabel(r"$v$ $(m/s)$", fontsize=14)
    plt.grid()

    # Plot acceleration (input)
    ax1 = plt.subplot(3, 1, 3)

    plt.plot(u[0, :])
    plt.ylabel(r"$a$ $(m/s^2)$", fontsize=14)
    plt.yticks([np.min(u), 0, np.max(u)])
    plt.ylim([np.min(u) - 0.1, np.max(u) + 0.1])
    plt.xlabel(r"$t$", fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.xlim([0, T])
    plt.show()

def plot_comparison(filename, x_without, x_with, u_without, u_with, T):
    f = plt.figure(figsize=(12, 6))

    # Plot position
    plt.title('Comparison of MPC methods')
    ax2 = f.add_subplot(311)
    plt.plot(x_without[0, :], label='Without Terminal Set')
    plt.plot(x_with[0, :], label='With Terminal Set')
    ax2.legend()
    x1 = x_without[0, :]
    plt.ylabel(r"$p$ $(m)$", fontsize=14)
    plt.yticks([np.min(x1), 0, np.max(x1)])
    plt.ylim([np.min(x1) - 0.1, np.max(x1) + 0.1])
    plt.xlim([0, T])
    plt.grid()

    # Plot velocity
    ax3 = plt.subplot(3, 1, 2)
    x2 = x_without[1, :]
    plt.plot(x_without[1, :], label='Without Terminal Set')
    plt.plot(x_with[1, :], label='With Terminal Set')
    ax3.legend()
    plt.yticks([np.min(x2), 0, np.max(x2)])
    plt.ylim([np.min(x2) - 0.1, np.max(x2) + 0.1])
    plt.xlim([0, T])
    plt.ylabel(r"$v$ $(m/s)$", fontsize=14)
    plt.grid()

    # Plot acceleration (input)
    ax1 = plt.subplot(3, 1, 3)
    plt.plot(u_without[0, :], label='Without Terminal Set')
    plt.plot(u_with[0, :], label='With Terminal Set')
    ax1.legend()
    plt.ylabel(r"$a$ $(m/s^2)$", fontsize=14)
    plt.yticks([np.min(u_without), 0, np.max(u_without)])
    plt.ylim([np.min(u_without) - 0.1, np.max(u_without) + 0.1])
    plt.xlabel(r"$t$", fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.xlim([0, T])
    # plt.savefig(filename, bbox_inches='tight')
    plt.show()