import matplotlib.pyplot as plt
import numpy as np


def plot_ptychi_probe(probe) -> plt.Figure:
    """
    Plot a ptychographic probe in Pty-Chi's format.

    Parameters
    ----------
    probe : np.ndarray
        The (n_opr, n_mode, h, w) probe to plot.

    Returns
    -------
        The figure.
    """
    fig, ax = plt.subplots(
        probe.shape[0], 
        probe.shape[1], 
        squeeze=False,
        figsize=(4 * probe.shape[1], 4 * probe.shape[0])
    )
    for i in range(probe.shape[0]):
        for j in range(probe.shape[1]):
            ax[i, j].imshow(probe[i, j])
            if i == 0:
                ax[i, j].set_title(f"Mode {j}")
            if j == 0:
                ax[i, j].set_ylabel(f"OPR mode {i}")
    
    plt.tight_layout()
    return fig


def plot_ptychi_object(object) -> plt.Figure:
    """
    Plot a ptychographic object.

    Parameters
    ----------
    object : np.ndarray
        The (n_slices, h, w) object to plot.

    Returns
    -------
        The figure.
    """
    fig, ax = plt.subplots(
        object.shape[0], 
        2, 
        squeeze=False,
        figsize=(8, 4 * object.shape[0])
    )
    for i in range(object.shape[0]):
        ax[i, 0].imshow(np.abs(object[i]))
        ax[i, 1].imshow(np.angle(object[i]))
        if i == 0:
            ax[i, 0].set_title("Magnitude")
            ax[i, 1].set_title("Phase")
    plt.tight_layout()
    return fig


def plot_ptychi_positions(positions) -> plt.Figure:
    """
    Plot the positions of the ptychographic object.

    Parameters
    ----------
    positions : np.ndarray
        The (n_positions, 2) positions to plot. Positions are given
        in row-major order, i.e. (y, x).

    Returns
    -------
        The figure.
    """
    fig, ax = plt.subplots(1, 1)
    ax.scatter(positions[:, 1], positions[:, 0])
    ax.plot(positions[:, 1], positions[:, 0], "--", linewidth=0.5, color="gray")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    return fig
