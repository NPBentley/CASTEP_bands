"""Plot the phonon dispersion from a CASTEP DFPT or finite displacement phonon calculation.

Default energy/frequency units are cm^-1 unless specified otherwise.
"""
# TODO Do we want phonon density of states?
# TODO Ability to select units?
# Created by: V. Ravindran, 05/11/2024

import ase
import matplotlib as mpl
import numpy as np
import spglib

from CASTEPbands import spgutils
from matplotlib import ticker


def _read_phonon_freqs(phononfile: str):
    """Read the phonon frequencies and wavevectors (qpoints) from a .phonon file.

    In addition, we also read the cell vectors and atomic positions in fractional
    coordinates. Default CASTEP units are assumed.

    Parameters
    ----------
    phononfile : str
        CASTEP .phonon file to read

    Returns
    -------
    qpt_array : ndarray shape=(nqpts, 3)
        list of phonon wavevectors (qpoints)
    qpt_weights : ndarray shape=(nqpoints)
        weights for each phonon wavevector
    phonon_freqs : ndarray shape=(nbranch,nqpoints)
        list of phonon frequencies for each branch and wavevector
    cell_vecs : np.ndarray shape=(3,3)
        unit cell vectors
    species_list : list
        list of ions/species in the unit cell
    frac_coords : ndarray shape=(nspecies, 3)
        position of each ion in fractional coordinates

    Raises
    ------
    AssertionError
        Expected a certain number of q-points from the header but we appear to have read less than
        this amount.
    IOError
        missing header information
    ValueError
        unknown unit for a physical quantity

    """

    def _check_header_line(line: str, quantity: list, case_insens: bool = False):
        # Check if header contains quantity in format
        # <quantity> : valence
        # To avoid weird spacing issues that sometimes exists in CASTEP output,
        # split by whitespace and compare the list
        line = line.strip()
        if case_insens:
            line = line.lower()

        if line.split()[:len(quantity)] == quantity:
            return True
        else:
            return False

    with open(phononfile, 'r', encoding='ascii') as file:
        # Read the header - NB: open file and read line by line to avoid loading entirety into memory
        line = file.readline()
        if line.strip().startswith('BEGIN header') is False:
            raise IOError('Could not find start of phonon header in ' + str(phononfile))

        nion, nbranch, nqpts = -1, -1, -1
        file_freq_unit, file_ir_unit, file_raman_unit = 'NULL', 'NULL', 'NULL'
        for lineno, line in enumerate(file):
            if lineno >= 6:
                # 6 lines to read containing information on rest of file
                break
            if _check_header_line(line, ['Number', 'of', 'ions']):
                nion = int(line.split()[-1])
            elif _check_header_line(line, ['Number', 'of', 'branches']):
                nbranch = int(line.split()[-1])
            elif _check_header_line(line, ['Number', 'of', 'wavevectors']):
                nqpts = int(line.split()[-1])
            elif _check_header_line(line, ['Frequencies', 'in']):
                file_freq_unit = line.split()[-1]
            elif _check_header_line(line, ['IR', 'intensities', 'in']):
                file_ir_unit = line.split()[-1]
            elif _check_header_line(line, ['Raman', 'activities', 'in']):
                file_raman_unit = ' '.join(line.split()[3:])  # HACK Raman units have spaces

        # Check we got all the necessary bits from the header
        for quant, val in zip(['ions', 'branches', 'wavevectors'],
                              [nion, nbranch, nqpts]):
            if val == -1:
                raise IOError('Could not find number of ' + str(quant))

        for quant, unit in zip(['frequency', 'IR intensity', 'Raman activity'],
                               [file_freq_unit, file_ir_unit, file_raman_unit]):
            if unit == 'NULL':
                raise IOError(f'Could not find {quant} unit')

        # Block of unsupported units
        if file_freq_unit != 'cm-1':
            raise ValueError('Unsupported frequency unit ', file_freq_unit)
        if file_ir_unit != '(D/A)**2/amu':
            raise ValueError('Unsupported IR intensity unit ', file_ir_unit)
        if file_raman_unit != 'A**4 amu**(-1)':
            raise ValueError('Unsupported Raman activity unit ', file_raman_unit)

        # Unlike .bands file, we have all the information in .phonon to construct the cell.
        # First read unit cell vectors
        if line.strip().startswith('Unit cell vectors') is False:
            raise IOError('Missing unit cell vectors in header')
        cell_vecs = np.empty((3, 3), dtype=float)
        for i in range(3):
            cell_vecs[i] = np.array(file.readline().split(), dtype=float)

        # Read the fractional coordinates and atoms in the cell.
        line = file.readline()
        if line.strip().startswith('Fractional Co-ordinates') is False:
            raise IOError('Missing fractional coordinates in header')

        frac_coords = np.empty((nion, 3), dtype=float)
        species_list = ['' for i in range(nion)]
        for i in range(nion):
            line = file.readline()
            split_line = line.split()
            frac_coords[i] = split_line[1:4]
            species_list[i] = split_line[4]

        # DEBUG header read
        # print('Unit cell vectors')
        # for i in range(3):
        #     print(cell_vecs[i])
        # print('Fractional coordinates')
        # for i in range(nion):
        #     print(species_list[i], frac_coords[i, :])

        # We should have finished reading the header but let's be sure!
        line = file.readline()
        if line.strip().startswith('END header') is False:
            raise AssertionError('End of header expected after fractional coordinates ' +
                                 'but not found')

        # Now let's get down to business - read in phonon frequencies for each wavevector
        # This file can be potentially very large so to avoid reading it all in memory
        # in one go, we use a combination of status flags to determine what we are reading.
        phonon_freqs = np.empty((nbranch, nqpts), dtype=float)
        qpt_weights = np.empty(nqpts, dtype=float)
        qpt_array = np.empty((nqpts, 3), dtype=float)  # array of wavevectors
        qpt_list = []  # list of wavevector indices

        # QUESTION Do we want to read phonon eigenvectors?
        have_eigenmodes, have_qpt = False, False
        imodes, ibranch = 0, 0
        read_qpts = 0
        for line in file:
            if line.strip().startswith('q-pt='):
                # We have a q-point so read it
                iq, qx, qy, qz, qw = line.split()[1:]
                iq = int(iq)-1  # CASTEP uses Fortran ordering which starts from 1
                qpt_list.append(iq)
                qpt_array[iq] = np.array([qx, qy, qz], dtype=float)
                qpt_weights[iq] = float(qw)
                read_qpts += 1
                have_qpt = True
                continue  # go next line after reading this one

            if have_qpt is True:
                # Read phonon frequencies for current qpt
                phonon_freqs[ibranch, iq] = float(line.split()[1])
                ibranch += 1
                if ibranch == nbranch:
                    have_qpt = False
                    ibranch = 0
                continue

            if line.split() == ['Phonon', 'Eigenvectors']:
                # We have phonon eigenmodes - set flag and go to next line
                have_eigenmodes = True
                continue
            if line.split() == ['Mode', 'Ion', 'X', 'Y', 'Z']:
                # Skip the mode line header
                continue
            if have_eigenmodes:
                # If we have eigenmodes, skip them
                imodes += 1
                if imodes == nbranch * nion:
                    # Have read all eigenmodes reset
                    imodes = 0
                    have_eigenmodes = False
                continue

    # Finished reading the file - check we actually have all of it
    qpt_list = np.array(qpt_list, dtype=int)
    if read_qpts != nqpts:
        raise AssertionError(f'Expected {nqpts} wavevectors in file but only have read {read_qpts}')
    if qpt_list.shape[0] != nqpts:
        raise AssertionError(f'Expected {nqpts} wavevectors in file but only have {read_qpts} qpoint indices')

    # Sort phonon wavevectors in order - this should not be necessary but let's be safe
    sort_indx = qpt_list.argsort()
    qpt_list = qpt_list[sort_indx]
    qpt_array = qpt_array[sort_indx]
    qpt_weights = qpt_weights[sort_indx]
    for ibranch in range(nbranch):
        phonon_freqs[ibranch, :] = phonon_freqs[ibranch, sort_indx]

    return qpt_array, qpt_weights, phonon_freqs, cell_vecs, species_list, frac_coords


class Phonon:
    """
    The phonon information from a CASTEP .phonon file.

    The actual data is stored in the array freqs with the following shape
    freqs[branches/modes, q-points]
    where q-points refers to the k-points at which the perturbation is computed.

    Attributes
    ----------
    nqpoint : int
        number of q-points in phonon calculation
    nbranch : int
        number of branches/modes in the phonon calculation
    qpoints : ndarray
        q-points for phonon calculation in fractional coordinates
    qpoint_weights : ndarray
        weights for each q-point
    cell : ase.atoms.Atoms
        unit cell for CASTEP calculation
    high_sym : ndarray
        qpoint index of high-symmetry points
    high_sym : list
        labels for each high-symmetry points

    Methods
    -------
    plot_dispersion
        plot the phonon dispersion or band structure
    """

    def __init__(self,
                 phononfile: str,
                 verbose: bool = False,
                 high_sym_spacegroup: bool = True,
                 override_bv: str = None,
                 ):
        """Initialises the phonon data.

        Parameters
        ----------
        seedname : str
            CASTEP seedname
        verbose : bool
            enable high output verbosity
        high_sym_spacegroup : bool
            uses high symmetry labels based on Bravais lattice
            for spacegroup (CASTEP Default)
        override_bv : str
            override the high-symmetry labels and use specified
            Bravais lattice instead for labels
        """

        # Open the phonon output file
        try:
            open(phononfile, 'r', encoding='ascii')
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                'Could not open phonon output file ' + str(phononfile)
            ) from exc

        # Read phonon file
        self.qpoints, self.qpoint_weights, self.freqs, cell_vecs, species_list, frac_coords = _read_phonon_freqs(phononfile)
        self.nbranch = self.freqs.shape[0]
        self.nqpoint = self.freqs.shape[1]

        if verbose:
            print('Number of wavevectors ', self.nqpoint)
            print('Number of branches    ', self.nbranch)
            print('Unit Cell Vectors (A)')
            for i in range(3):
                print('{:12.6f} {:12.6f} {:12.6f}'.format(*cell_vecs[i]))
            print('Number of ions        ', len(species_list))
            print('Number of species     ', len(set(species_list)))

        # DEBUG - Check q-points and frequencies same as written by CASTEP
        # for iq in range(self.nqpoint):
        #     print('q-pt= {:5n} {:10.6f} {:10.6f} {:10.6f} {:14.10f}'.format(iq+1, *self.qpoints[iq, :], self.qpoint_weights[iq]))
        #     for ibranch in range(self.nbranch):
        #         print(f'{ibranch+1:8n} {self.freqs[ibranch, iq]:15.6f}')

        # Create the unit cell for this calculation
        self.cell = ase.Atoms(symbols=species_list, scaled_positions=frac_coords, cell=cell_vecs, pbc=True)

        if verbose:
            vol = self.cell.get_volume()
            total_mass = np.sum(self.cell.get_masses())
            density = total_mass/vol
            print(f'Density =  {density:14.7f}  amu/A^3= ',
                  f'{density*1.6605391:15.7f} g/cm^3')

            # Get spacegroup of cell
            # ase.spacegroup.get_spacegroup is depreciated so now use spglib directly V Ravindran USE_SPGLIB 23/01/2025
            # spglib expects cell as a tuple with in the order of                     V Ravindran USE_SPGLIB 23/01/2025
            # lattice vectors, fractional coords and species (by atomic number)       V Ravindran USE_SPGLIB 23/01/2025
            spg_cell = (self.cell.cell[:], self.cell.get_scaled_positions(), self.cell.get_atomic_numbers())
            spg_symb, spgno_str = spglib.get_spacegroup(spg_cell).split()
            # Remove the brackets returned around number in the above                 V Ravindran USE_SPGLIB 23/01/2025
            spg_no = int(spgno_str[spgno_str.find('(') + 1: spgno_str.find(')')])
            print(f'Space group: {spg_no} {spg_symb}')

        # Finally, create the high-symmetry lines
        self.high_sym, self.high_sym_labels = spgutils._get_high_sym_lines(self.qpoints, self.cell,
                                                                           high_sym_spacegroup=high_sym_spacegroup,
                                                                           override_bv=override_bv
                                                                           )

    def plot_branch(self, ax,
                    linestyle='-',
                    linewidth=1.2
                    ):
        # TODO Choose which branches/mode to plot
        # Loop over all modes and plot them
        qpt_indx = np.arange(self.nqpoint)
        for ibranch in range(self.nbranch):
            ax.plot(qpt_indx, self.freqs[ibranch])

    def plot_dispersion(self, ax: mpl.axes.Axes,
                        axes_only: bool = False,
                        linestyle: str = '-',
                        linewidth: float = 1.2,
                        freq_lim: list = None,
                        klim: list = None,
                        sym_lines: bool = True,
                        fontsize: float = 20.0
                        ):
        """Plot the phonon dispersion/band structure.

        Parameters
        ----------
        ax : mpl.axes.Axes
            axes to plot phonon data
        axes_only : do not plot data, initialise high symmetry lines and axes labels only
            do not plot data, initialise high symmetry lines and axes labels only
        linestyle : str
            type of line to use when plotting phonon frequencies
        linewidth : float
            thickness of line for phonon frequencies
        freq_lim : ndarray
            limits of frequency axis (in cm^-1)
        klim : list(dtype=str) or list(dtype=float) - see below
            Limit of k-point axes.
            This can be specified in by either the high-symmetry point labels (string) or kpoint index (integer)
        sym_lines : bool
            display horizontal lines indicating high-symmetry points
        fontsize : float
            font size to use in plot
        """

        # TODO Choose which branches/mode to plot
        # TODO Set k-point limits

        # Set axis labels and tick intervals and all that boring stuff
        eng_unit = r'$\text{cm}^{-1}$'
        ax.set_ylabel(r'$\omega$ ' + f'({eng_unit})', fontsize=fontsize)

        ax.set_xlim(0, len(self.qpoints)-1)
        ax.set_ylim(np.amin(self.freqs[0, :])-10, np.amax(self.freqs[-1, :]+10))
        if klim is not None:
            plot_klim = spgutils.get_klim(self.high_sym, self.high_sym_labels, klim)
            ax.set_xlim(plot_klim[0], plot_klim[1])
        if freq_lim is not None:
            ax.set_ylim(freq_lim[0], freq_lim[1])

        ax.tick_params(axis='both', direction='in', which='major', labelsize=fontsize * 0.8, length=12, width=1.2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize * 0.8, length=6,
                       right=True, top=False, bottom=False, left=True, width=1.2)

        ax.set_xticks(self.high_sym)
        ax.set_xticklabels(self.high_sym_labels)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # Draw high-symmetry lines
        if sym_lines:
            for i in self.high_sym:
                ax.axvline(i, color='k', linewidth=1)

        # We're now in business - decide to stop here or continue
        if axes_only:
            return

        # Plot phonon dispersion
        self.plot_branch(ax, linestyle=linestyle, linewidth=linewidth)
