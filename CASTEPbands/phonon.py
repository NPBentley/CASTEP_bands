"""Plot the phonon dispersion from a CASTEP DFPT or finite displacement phonon calculation.

Default energy/frequency units are cm^-1 unless specified otherwise.
"""
# TODO Do we want phonon density of states?
# TODO Ability to select units?
# Created by: V. Ravindran, 05/11/2024

import matplotlib as mpl
import numpy as np


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
    qpt_list : ndarray shape=(nqpts, 3)
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
        qpt_list = np.empty((nqpts, 3), dtype=float)

        # QUESTION Do we want to read phonon eigenvectors?
        have_eigenmodes, have_qpt = False, False
        imodes, ibranch = 0, 0
        read_qpts = 0
        for line in file:
            if line.strip().startswith('q-pt='):
                # We have a q-point so read it
                iq, qx, qy, qz, qw = line.split()[1:]
                iq = int(iq)-1  # CASTEP uses Fortran ordering which starts from 1
                qpt_list[iq] = np.array([qx, qy, qz], dtype=float)
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
    if read_qpts != nqpts:
        raise AssertionError(f'Expected {nqpts} wavevectors in file but only have {read_qpts}')

    return qpt_list, qpt_weights, phonon_freqs, cell_vecs, species_list, frac_coords


class Phonon:
    # TODO Documentation

    def __init__(self, seedname: str,
                 ):
        # TODO Documentation

        # Open the phonon output file
        phononfile = seedname + '.phonon'
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

        # DEBUG - Check same as written by CASTEP
        for iq in range(self.nqpoint):
            print('q-pt= {:5n} {:10.6f} {:10.6f} {:10.6f} {:14.10f}'.format(iq+1, *self.qpoints[iq, :], self.qpoint_weights[iq]))
            for ibranch in range(self.nbranch):
                print(f'{ibranch+1:8n} {self.freqs[ibranch, iq]:15.6f}')
