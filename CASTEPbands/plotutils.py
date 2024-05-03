"""Contains convenience functions that are useful for creating more advanced plots.

This allows for the creation of more complicated plots.
In general, the axes must already be created as a pre-requisitie for these functions.
"""
# Created by: V. Ravindran, 22/04/2024
import matplotlib as mpl
import numpy as np

from CASTEPbands import Spectral


def plot_bands(spec: Spectral.Spectral, ax: mpl.axes._axes.Axes, band_ids: np.ndarray,
               color: 'str | list' = None, label_first: str = None, labels: list = None,
               linewidth: float = 1.2, linestyle: str = '-',
               spin_index: int = None):
    """Add a specified set of bands to an existing bandstructure axis.

    By default, bands for both spin channels will be done if spin polarised but only the first spin
    channel will have labels added.

    Parameters
    ----------
    spec : Spectral.Spectral
        the bands data read from CASTEP
    ax : mpl.axes._axes.Axes
        the axes to plot the bands on
    band_ids : np.ndarray
        indices of band to plot
    color : str or list
        colour to use for bands. If str, will use same colour for all bands, otherwise if list,
        colours will be set on a band-by-band basis.
        (default : none - will use colours from rcParams)
    label_first : str
        the label for the first band in the set (default : none)
    labels : list
        labels to use for each band (default : none)
    linewidth : float
        width of lines for bands (default = 1.2)
    linestyle : str
        style of lines for bands (default = '-')
    spin_index : int
        spin channel to plot (default : none)

    Raises
    ------
    IndexError
        colour/labels do not have the same length as band_ids
    """
    band_ids = band_ids.astype(int)
    if isinstance(color, list):
        if band_ids.shape != len(color):
            raise IndexError('Number of colours does not match number of bands')

    if labels is not None:
        labels = list(labels)
        if band_ids.shape != len(labels):
            raise IndexError('Number of labels does not match number of bands')

    def _add_bands_for_spin(ns: int, add_labels=True):
        """Add bands for a given spin channel to the plot."""
        do_label_first = False
        if label_first is not None and add_labels is True:
            do_label_first = True

        do_labels = False
        if labels is not None and add_labels is True:
            do_labels = True

        banddata = spec.BandStructure
        kpts = spec.kpoints

        # Add bands for specified spin channel
        for i, nb in enumerate(band_ids):
            if do_label_first is True and i == 0:
                # Label the first band in the set using a separate label if requested
                ax.plot(kpts, banddata[nb, :, ns], color=color,
                        linestyle=linestyle, linewidth=linewidth,
                        label=label_first)
                # Make sure to set the flag to false, otherwise, otherwise all bands
                # will be labelled - not what we want!
                do_label_first = False

            elif do_labels is True:
                # Label bands
                ax.plot(kpts, banddata[nb, :, ns], color=color,
                        linestyle=linestyle, linewidth=linewidth,
                        label=labels[i])
            else:
                # Just plot the bands
                ax.plot(kpts, banddata[nb, :, ns], color=color)

    # Decide on spins to do. By default, do both spins
    if spin_index is not None:
        _add_bands_for_spin(spin_index)
    else:
        # Do both spin channels, we will only add labels for the first spin channel.
        add_labels = True
        for ns in range(spec.nspins):
            if ns == 1:
                add_labels = False
            _add_bands_for_spin(ns, add_labels=add_labels)


def add_vb_cb(spec: Spectral.Spectral, ax: mpl.axes._axes.Axes,
              colors: list = ['b', 'r'], labels: list = None,
              linewidth: float = 1.2, linestyle: str = '-',
              spin_index: int = None):
    """Plot only the valence and conduction bands on the plot

    By default, bands for both spin channels will be done if spin polarised but only the first spin
    channel will have labels added.

    Parameters
    ----------
    spec : Spectral.Spectral
        the bands data read from CASTEP
    ax : mpl.axes._axes.Axes
        the axes to plot the bands on
    colors : list
        colour to use for valence band and conduction band (for both spin channels if spin_index is None)
    labels : list
        labels to use for valence band and conduction band
    linewidth : float
        width of lines for bands (default = 1.2)
    linestyle : str
        style of lines for bands (default = '-')
    spin_index : int
        spin channel to plot (default : none)

    Raises
    ------
    IndexError
        colour/labels do not have the same length as band_ids
    """
    if labels is not None:
        labels = list(labels)
        if len(labels) != 2:
            raise IndexError('Labels must specified for both VBM and CBM')

    # Get the number of electrons in each spin channel
    if spec.nspins == 1:
        nelec = np.array([spec.electrons], dtype=int)
    else:
        nelec = np.array([spec.no_up, spec.no_down], dtype=int)

    def _get_vb_cb(ns: int):
        """Get valence and conduction bands for each spin channel."""
        vb_eigs = spec.BandStructure[int(nelec[ns] / spec.occ) - 1, :, ns]
        cb_eigs = spec.BandStructure[int(nelec[ns] / spec.occ), :, ns]
        return vb_eigs, cb_eigs

    # Decide which valence and conduction band we are going to plot
    if spin_index is not None:
        # Plot only the VB and CB for the given spin channel
        vb_eigs, cb_eigs = _get_vb_cb(spin_index)
        ax.plot(spec.kpoints, vb_eigs, color=colors[0],
                linestyle=linestyle, linewidth=linewidth,
                label=labels[0])
        ax.plot(spec.kpoints, cb_eigs, color=colors[1],
                linestyle=linestyle, linewidth=linewidth,
                label=labels[1])
    else:
        # Plot both spin channels - label only the first spin
        for ns in range(spec.nspins):
            if ns == 1:
                # Do not want to add labels for second spin channel
                labels = [None, None]

            vb_eigs, cb_eigs = _get_vb_cb(ns)
            ax.plot(spec.kpoints, vb_eigs, color=colors[0],
                    linestyle=linestyle, linewidth=linewidth,
                    label=labels[0])
            ax.plot(spec.kpoints, cb_eigs, color=colors[1],
                    linestyle=linestyle, linewidth=linewidth,
                    label=labels[1])


def compare_bands():
    raise NotImplementedError
