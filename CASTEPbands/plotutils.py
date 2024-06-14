"""Contains convenience functions that are useful for creating more advanced plots.

This allows for the creation of more complicated plots.
In general, the axes must already be created as a pre-requisitie for these functions.
"""
# Created by: V. Ravindran, 22/04/2024
import warnings
from copy import deepcopy

import matplotlib as mpl
import numpy as np

from CASTEPbands import Spectral


def plot_bands(spec: Spectral.Spectral, ax: mpl.axes._axes.Axes, band_ids: np.ndarray = None,
               color: 'str | list' = None, label_first: str = None, labels: list = None,
               linewidth: float = 1.2, linestyle: str = '-',
               marker: str = None, markersize: float = None,
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
        indices of band to plot. If None provided, then all the bands will be plot
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
    marker : str
        marker style to use
    markersize : float
        size of markers to use
    spin_index : int
        spin channel to plot (default : none)

    Raises
    ------
    IndexError
        colour/labels do not have the same length as band_ids
    """
    if band_ids is not None:
        band_ids = np.array(band_ids, dtype=int)
    else:
        # 09/05/2024 - Plot all bands by default
        nbands = spec.BandStructure.shape[0]
        band_ids = np.arange(nbands, dtype=int)

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
                        # Marker style added 09/05/2024
                        marker=marker, markersize=markersize,
                        label=label_first)
                # Make sure to set the flag to false, otherwise, otherwise all bands
                # will be labelled - not what we want!
                do_label_first = False

            elif do_labels is True:
                # Label bands
                ax.plot(kpts, banddata[nb, :, ns], color=color,
                        linestyle=linestyle, linewidth=linewidth,
                        # Marker style added 09/05/2024
                        marker=marker, markersize=markersize,
                        label=labels[i])
            else:
                # Just plot the bands
                ax.plot(kpts, banddata[nb, :, ns], color=color,
                        linestyle=linestyle, linewidth=linewidth,
                        # Marker style added 09/05/2024
                        marker=marker, markersize=markersize)

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
        nelec = np.array([spec.nup, spec.ndown], dtype=int)

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


def align_bands(spec: Spectral.Spectral, spec_ref: Spectral.Spectral,
                band_id_ref: 'int | str' = 'VBM', spin_index: int = 0,
                force_align=False, silent=False):
    """Shift the bands so that a given eigenvalue is the same in both band structures.

    By default, a shift is not performed if the bandstructures do not have the same calculation
    parameters, e.g. spins kpoints, etc. This behaviour can be disabled using force_align=True in
    which case warnings will be raised (that can be surpressed by silent=False)

    Parameters
    ----------
    spec : Spectral.Spectral
        the actual bands data to shift
    spec_ref : Spectral.Spectral
        the reference bands data
    band_id_ref : int or str
        the band whose maximum eigenvalue will be used as the reference
        Alternatively, aliases can be used for the 'VBM' (or 'HOMO') as well 'CBM' (or 'LUMO')
    spin_index : int
        spin channel to use (default : 0 / spin #1)
    force_align : bool
        force bands to be shifted even if they are not equivalent band structures
    silent : bool
        surpress warnings

    Returns
    -------
    newspec : Spectral.Spectral
        the shifted bands

    Raises
    ------
    AssertionError
        Bands are not equivalent
    TypeError
        invalid type for band_id_ref
    IndexError
        band index out of bounds in band_id_ref
        spin_index out of bounds
    ValueError
        invalid band_id_ref string alias
    """

    if spin_index != 0 and spec_ref.nspins != 1:
        raise IndexError('Reference band structure only has 1 spins but second spin channel requested')

    def _get_max_eigenvalue(spec_ref: Spectral.Spectral, band_id_ref: 'int | str'):
        """Gets the maximum eigenvalue for a given band within the bandstructure"""
        if isinstance(band_id_ref, str):
            band_id_ref = band_id_ref.upper()

            bandinfo = spec_ref.get_band_info(silent=True, band_order='C')

            # Get VBM/CBM as required
            if band_id_ref in ('VBM', 'HOMO'):
                eig_ref = bandinfo['vbm'][spin_index]
            elif band_id_ref in ('CBM', 'LUMO'):
                eig_ref = bandinfo['cbm'][spin_index]
            else:
                raise ValueError(f'Unknown band label {band_id_ref}, '
                                 + 'must be VBM/HOMO or CBM/LUMO')

        elif isinstance(band_id_ref, int):
            if band_id_ref >= spec_ref.BandStructure.shape[0]:
                raise IndexError(f'Band index {band_id_ref} is out of bounds for '
                                 + f'bandstructure with {spec_ref.BandStructure.shape[0]+1} bands.')

            # Get maximum eigenvalue for the specified band
            eig_ref = np.max(spec_ref.BandStructure[band_id_ref, :, spin_index])

        else:
            raise TypeError('band_id_ref must be str or int type')
        return eig_ref

    # Check if the two band structures are the same / comparable (except for eigenvalues themselves)
    # This basically involves checking if the number of kpoints, paths, spins etc are the same.
    def _warn_bs_mismatch(errmsg):
        """Raises error/warning as appropriate if bands do not match"""
        if force_align is False:
            # Do not align bands if mismatch, raise error and halt
            raise AssertionError(errmsg + ' Use force_align=False  to bypass')
        if silent is False:
            # Allow alignment but print a warning message unless silent
            warnings.warn(errmsg + ' Use silent=True to surpress warnings')

    if spec.BandStructure.shape != spec_ref.BandStructure.shape:
        errmsg = f'Ref. bandstructure has (nk, nb, ns)={spec_ref.BandStructure.shape}' + \
            f' but actual has {spec.BandStructure.shape}.'
        _warn_bs_mismatch(errmsg)
    elif spec.have_ncm != spec_ref.have_ncm:
        errmsg = f'Ref. bandstructure have_ncm={spec_ref.have_ncm}' + \
            f' but actual have_ncm={spec.have_ncm}.'
        _warn_bs_mismatch(errmsg)
    elif np.allclose(spec.kpoint_list, spec_ref.kpoint_list, atol=1e-10, rtol=1e-7) is False:
        errmsg = 'Paths between reference and actual band structure do not appear to be the same.'
        _warn_bs_mismatch(errmsg)

    # Now let's get down to business.
    # Get the reference eigenvalue.
    eig_ref = _get_max_eigenvalue(spec_ref, band_id_ref)
    eig_act = _get_max_eigenvalue(spec, band_id_ref)

    # Make a (deep)copy of the spectral data -  this may cause problems!
    # Without deep copy, the new class merely references the orignal so the original data is overriden.
    newspec = deepcopy(spec)

    # Shift the eigenvalues so that the reference eigenvalues align
    eng_shift = eig_act - eig_ref
    newspec.shift_bands(eng_shift)

    return newspec
