import time
import warnings
from itertools import cycle

import ase
import ase.dft.bz as bz
import ase.io as io
import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers


class Spectral:
    """
    The band information from a CASTEP .bands file.

    The actual data is stored in a numpy array band_structure that has the following shape
    band_structure[max_eig, no_kpoints, nspins]
    where max_eig is the maximum number of eigenvalues (in spin polarised calculations, these may not match perfectly).

    Attributes
    -----------
    seed : string
         The seedname of the CASTEP run, e.g.  <seedname>.bands.
    use_vbm_fermi : boolean
         Use the valence band maximum (VBM) as the Fermi energy.
         This is particularly useful for instulators (default : False)
    zero_fermi : boolean
         Should the eigenvalues be shifted such that the Fermi energy is at the zero of energy (Default : True)
    zero_vbm : boolean
         Shift eigenvalues such that valence band maximum lies as zero of energy
    zero_cbm : boolean
         Same as zero_vbm but for conduction band
    zero_shift : shift all eigenvalues DOWN by a constant value (in eV)
         NB: To shift UPWARDS - a NEGATIVE shift must be specified.
    convert_to_eV : boolean
         Convert eigenvalues from atomic units (Hartrees) to electronvolts
    flip_spins : boolean
         Swap the spin channels when making the plot (Default : True)

    Methods
    ----------
    plot_bs
         Plots the band structure from the .bands file
    pdos_filter
         Function for separating the partial density of states by species,ion and angular momentum.
    plot_dos
         Plots the density of states
    kpt_where
         Obtains the label of a given high-symmetry point from its coordinates.
    get_band_info
         Get a summary of information in the band structure.
    """

    def __init__(self,
                 seed,
                 zero_fermi=True,
                 use_vbm_fermi=False,
                 zero_vbm=False,
                 zero_cbm=False,
                 zero_shift=None,
                 convert_to_eV=True,
                 flip_spins=False):
        ''' Initalise the class, it will require the CASTEP seed to read the file '''
        self.start_time = time.time()
        self.pdos_has_read = False

        # Functions
        def _check_sym(vec):
            fracs = np.array([0.5, 0.0, 0.25, 0.75, 0.33333333, 0.66666667])
            frac = []
            for i in vec:
                # frac.append(i.as_integer_ratio()[0])
                # frac.append(i.as_integer_ratio()[1])
                buff = []
                for j in fracs:
                    buff.append(np.isclose(i, j))
                frac.append(any(buff))

            if all(frac):
                # print(vec)
                return True
            else:
                return False

        if convert_to_eV:
            eV = 27.2114
            self.eV = eV
        else:
            eV = 1
            self.eV = eV
        self.convert_to_eV = convert_to_eV
        self.seed = seed
        self.zero_fermi = zero_fermi
        self.zero_vbm = zero_vbm
        self.zero_cbm = zero_cbm
        self.use_vbm_fermi = use_vbm_fermi

        # First we try to open the file

        # Open the bands file
        try:
            bands_file = seed + ".bands"
            bands = open(bands_file, 'r')
        except:
            raise FileNotFoundError("No .bands file")

        lines = bands.readlines()

        no_spins = int(lines[1].split()[-1])
        no_kpoints = int(lines[0].split()[-1])
        fermi_energy = float(lines[4].split()[-1])  # TODO This only returns one of the Fermi energies (problem if open-shell system)

        if no_spins == 1:
            fermi_energy = float(lines[4].split()[-1])
            no_electrons = float(lines[2].split()[-1])
            no_eigen = int(lines[3].split()[-1])
            max_eig = no_eigen
            no_eigen_2 = None
            spin_polarised = False
        if no_spins == 2:
            spin_polarised = True
            no_eigen = int(lines[3].split()[-2])
            no_eigen_2 = int(lines[3].split()[-1])
            max_eig = np.max([no_eigen, no_eigen_2])
            n_up = float(lines[2].split()[-2])
            n_down = float(lines[2].split()[-1])

        # Set all of the bands information
        self.spin_polarised = spin_polarised
        self.Ef = fermi_energy * eV
        # NB: Shifting of Fermi energy is done once we decide on energy shift for the eigenvalues later - V Ravindran 31/01/2024
        self.n_kpoints = no_kpoints
        if spin_polarised:
            self.nup = n_up
            self.ndown = n_down
            self.electrons = n_up + n_down
        else:
            self.nup = None
            self.ndown = None
            self.electrons = no_electrons
        self.eig_up = no_eigen
        self.eig_down = no_eigen_2
        self.n_kpoints = no_kpoints

        band_structure = np.zeros((max_eig, no_kpoints, no_spins))  # bands, kpt, spin

        kpt_weights = np.zeros(no_kpoints)

        kpoint_array = np.empty(shape=(no_kpoints))  # the array holding the number of the kpoint
        kpoint_list = []  # array of the kpoint vectors

        if no_spins == 1:
            kpoint_string = lines[9::no_eigen + 2]
        else:
            kpoint_string = lines[9::no_eigen + 3 + no_eigen_2]
        for i in range(len(kpoint_string)):
            kpt_weights[i] = float(kpoint_string[i].split()[-1])

        for i in range(len(kpoint_string)):
            kpoint_array[i] = int(kpoint_string[i].split()[1])

            # Empty list for vectors
            vec = []
            vec.append(float(kpoint_string[i].split()[2]))
            vec.append(float(kpoint_string[i].split()[3]))
            vec.append(float(kpoint_string[i].split()[4]))
            kpoint_list.append(vec)

        # fill up the arrays
        for k in range(0, no_kpoints):
            if no_spins == 1:
                ind = 9 + k * no_eigen + 2 * (k + 1)
                band_structure[:, k, 0] = eV * np.array([float(i) for i in lines[ind:ind + no_eigen]])

            if no_spins == 2:
                ind = 9 + k * (no_eigen + no_eigen_2 + 1) + 2 * (k + 1)
                band_structure[:, k, 0] = eV * np.array([float(i) for i in lines[ind:ind + no_eigen]])
                band_structure[:, k, 1] = eV * np.array([float(i) for i in lines[ind + no_eigen + 1:ind + no_eigen + 1 + no_eigen_2]])

        # Get valence and conduction bands
        if no_spins == 1:
            vb_eigs = band_structure[int(no_electrons / 2) - 1, :, 0]
            cb_eigs = band_structure[int(no_electrons / 2), :, 0]
        else:
            vb_eigs = band_structure[int(n_up) - 1, :, 0]
            cb_eigs = band_structure[int(n_up), :, 0]

        # Decide if we now want to keep the CASTEP Fermi energy or use the VBM - V Ravindran 30/04/2024
        if use_vbm_fermi is True:
            fermi_energy = np.amax(vb_eigs)
            self.Ef = fermi_energy * eV

        # Decide on how we want to shift the bands based on user's preference - V Ravindran 31/01/2024
        # NB: For zero_cbm and zero_vbm, we take the VBM/CBM from the first spin channel if spin polarised (arbitrarily).
        # The error handling for this is a bit of a pain in the arse for this one...
        # In order of preference(highest to lowest): zero_vbm,zero_shift,zero_cbm,zero_fermi (since it's default)
        if zero_vbm is True:
            eng_shift = np.amax(vb_eigs)
        elif zero_cbm is True:
            eng_shift = np.amin(cb_eigs)
        elif zero_shift is not None:
            eng_shift = float(zero_shift)
            if convert_to_eV is False:
                # NB: User specifies shift in eV so convert to Hartrees.
                eng_shift = eng_shift / eV
        elif zero_fermi is True:
            # Since this is a default, to minimise number of kwargs user has to use in class,
            # this needs to be far down as possible
            eng_shift = fermi_energy * eV
        else:
            eng_shift = 0.0

        # Now perform the shift for real
        # for k in range(no_kpoints):
        #     for ns in range(no_spins):
        #         band_structure[:, k, ns] = band_structure[:, k, ns] - eng_shift
        band_structure = band_structure - eng_shift  # Vectorized shift V Ravindran 30/04/2024

        # Don't forget to shift the Fermi energy as well V Ravindran 31/01/2024
        self.Ef = self.Ef - eng_shift

        sort_array = kpoint_array.argsort()
        kpoint_array = kpoint_array[sort_array]
        kpoint_list = np.array(kpoint_list)[sort_array]
        self.kpt_sort = sort_array
        for nb in range(max_eig):
            for ns in range(no_spins):
                band_structure[nb, :, ns] = band_structure[nb, :, ns][sort_array]

        if no_spins == 2 and flip_spins:
            band_structure[:, :, [0, 1]] = band_structure[:, :, [1, 0]]

        self.kpoints = kpoint_array
        self.kpoint_list = kpoint_list
        self.kpt_weights = kpt_weights[sort_array]
        self.BandStructure = band_structure
        self.nbands = max_eig
        self.nspins = no_spins

        # do the high symmetry points
        k_ticks = []
        for i, vec in enumerate(kpoint_list):
            if _check_sym(vec):
                k_ticks.append(kpoint_array[i])

        tol = 1e-5

        kpoint_grad = []
        for i in range(1, len(kpoint_list)):
            diff = kpoint_list[i] - kpoint_list[i - 1]
            kpoint_grad.append(diff)

        kpoint_2grad = []
        high_sym = [0]
        for i in range(1, len(kpoint_grad)):
            diff = kpoint_grad[i] - kpoint_grad[i - 1]
            kpoint_2grad.append(diff)
            # print(diff)

            if any(np.abs(diff) > tol):

                # print(diff)
                high_sym.append(i)
        high_sym.append(len(kpoint_list) - 1)
        high_sym = np.array(high_sym) + 1
        self.high_sym = high_sym

        # Set up the special points
        # This used to write to os.devnull because ASE would whinge about a missing CASTEP executable. V Ravindran CELL_READ 01/05/2024
        # There is a way to correct this however using an ASE keyword.                                 V Ravindran CELL_READ 01/05/2024
        # Moreover, this ensures that the JSON file for CASTEP will not be set up if the               V Ravindran CELL_READ 01/05/2024
        # CASTEP_COMMAND environmental variable is set                                                 V Ravindran CELL_READ 01/05/2024
        cell = io.read(seed + ".cell",
                       # Do not check CASTEP keywords, the calculation is presumably correct! V CELL_READ Ravindran 01/05/2024
                       calculator_args={"keyword_tolerance": 3}
                       )
        bv_latt = cell.cell.get_bravais_lattice()
        special_points = bv_latt.get_special_points()

        atoms = np.unique(cell.get_chemical_symbols())[::-1]
        mass = []
        for i in atoms:
            mass.append(atomic_numbers[i])
        atom_sort = np.argsort(mass)
        mass = np.array(mass)[atom_sort]
        atoms = np.array(atoms)[atom_sort]
        self.atoms = atoms
        self.mass = mass

        # except:
        # sys.stdout = sys.__stdout__
        #    warnings.warn("No .cell file found for generating high symmetry labels")

        ticks = [""] * len(high_sym)
        found = False
        for k_count, k in enumerate(kpoint_list[high_sym - 1]):
            found = False

            for i in special_points:

                if abs(special_points[i][0] - k[0]) < tol and abs(special_points[i][1] - k[1]) < tol and abs(special_points[i][2] - k[2]) < tol:
                    if i == "G":
                        ticks[k_count] = r"$\Gamma$"
                    else:
                        ticks[k_count] = i
                    found = True

        self.high_sym_labels = ticks
        self.dk = np.sum((np.sum(kpoint_list, axis=0) / no_kpoints)**2)
        # We have all the info now we can break it up
        # warnings.filterwarnings('always')

    def shift_bands(self, eng_shift, use_eng_unit=None):
        """Shift all eigenvalues DOWNWARDS by a constant.
        To shift the eigenvalues up, a negative shift must be specified.

        Author: V Ravindran (26/02/2024)

        Parameters
        ----------
        eng_shift : float
            shift to apply to eigenvalues.
            The energy units by default will be set to the same units as the bands data.
        use_eng_unit : str
            override the energy units to use when specifying the energy shift.
            Acceptable values are either 'hartrees' or 'ev'  (case-insensitive).

        Raises
        ------
        ValueError
            Invalid energy units specifed for use_eng_unit, must be either 'hartrees' or 'ev'.
        """

        eV = 27.2114
        # print('Band structure is in units of eV?: ', self.convert_to_eV)
        # Decide what energy units the bands are in
        band_units = 'hartrees'
        if self.convert_to_eV is True:
            band_units = 'ev'
        # Assume the same energy units for the user unless overriden
        shift_units = band_units
        if use_eng_unit is not None:
            shift_units = use_eng_unit.strip().lower()
        if shift_units not in ('hartrees', 'ev'):
            raise ValueError('Invalid unit specified for energy shift. Must be "hartrees" or "ev" (case-insensitive)')

        # If the user's energy shift and band data are not the same (what a weirdo),
        # then convert the energy shift to match the energy units of the bands.
        if shift_units != band_units:
            if shift_units == 'hartrees' and band_units == 'ev':
                eng_shift *= eV
            elif shift_units == 'ev' and band_units == 'hartrees':
                eng_shift /= eV

        # print(f'Energy units and shift {band_units=}, {shift_units=}, {eng_shift=}')
        # print(f'{self.nbands=}, {self.n_kpoints=}, {self.nspins=}')
        # print(f'{self.BandStructure.shape=}')

        # Now shift the eigenvalues...
        for ns in range(self.nspins):
            for nk in range(self.n_kpoints):
                self.BandStructure[:, nk, ns] = self.BandStructure[:, nk, ns] - eng_shift

        # ... and the Fermi energy
        self.Ef -= eng_shift

        return

    def get_band_info(self, silent=False, bandwidth=None, band_order='F', ret_vbm_cbm=False):
        """Get a summary of the band structure.

        Author: V Ravindran (30/01/2024)

        Updated: 01/05/2024 to use vectorised operations
        Corrected ret_vbm_cbm to return vbms for each spin channel

        Parameters
        ----------
        silent : boolean
            Do not print out information to standard output (Default: False)
        bandwidth : integer
            Obtain the band width for a specific band (see band_order)
        band_order : string
            Type of array ordering to use when deciding which band to use for band width
            measurements. CASTEP uses Fortran ordering (arrays start from 1).
            (default : 'F')
        ret_vbm_cbm : boolean
            Return the index of the kpoint required to get the valence band maximum and conduction band minimum
            together with the respective eigenvalues.

        Returns
        -------
        dictionary
            a dictionary containing the various properties of the band structure.

        Raises
        ------
        NameError
            Invalid band_order specified - Invalid value specified - must be either 'C' or 'F'

        """

        def _get_bandwidth(self, band, band_order):
            """Helper function to get the width of a given band for each spin
            Author: V Ravindran (30/01/2024)
            """

            # Decide on how the user wants to order the bands.
            if band_order == 'F':
                band = band - 1
            elif band_order == 'C':
                pass
            else:
                raise NameError("band_order be either 'C' or 'F'")

            # Now get the width of the band for each spin.
            width = np.empty(self.nspins)
            for ns in range(self.nspins):
                eigs = self.BandStructure[band, :, ns]
                width[ns] = np.max(eigs) - np.min(eigs)
            return width

        if bandwidth is not None:
            # If the user just wants the width of a specific band, just return that instead.
            return _get_bandwidth(self, bandwidth, band_order)

        # Get a summary of the band structure instead, start with the bits we already have in the class
        band_info = {'nelec': None, 'nbands': None, 'nspins': self.nspins, 'nkpts': self.n_kpoints,
                     'gap_indir': None, 'gap_dir': None, 'loc_indir': None, 'loc_dir': None,
                     'fermi': self.Ef, 'vb_width': None, 'cb_width': None, 'eng_unit': None}

        # Get the number of electrons and bands
        nelecs = np.empty(self.nspins, dtype=int)
        nbands = np.empty(self.nspins, dtype=int)
        if (self.nspins == 2):
            nelecs[0] = self.nup
            nelecs[1] = self.ndown
            nbands[0] = self.eig_up
            nbands[1] = self.eig_down
        else:
            nelecs[0] = self.electrons
            nbands[0] = self.eig_up

        # Set occupancies
        occ = 1
        if (self.nspins == 1):
            # If not spin polarised, then we are doubly occupying levels
            occ = 2

        # Vectorised operations V Ravindran 01/05/2024
        # Get valence and conduction bands for each spin
        vb_eigs = np.empty((self.nspins, self.n_kpoints))
        cb_eigs = np.empty((self.nspins, self.n_kpoints))
        for ns in range(self.nspins):
            vb_eigs[ns] = self.BandStructure[int(nelecs[ns] / occ) - 1, :, ns]
            cb_eigs[ns] = self.BandStructure[int(nelecs[ns] / occ), :, ns]

        # Determine valence band maximum and conduction band minimum to get (indirect) gap
        # NB: It may not actually be indirect, in direct gapped insulators, gap_dir = gap_in
        vbm_i = np.argmax(vb_eigs, axis=1)  # valence band maximum kpt index
        cbm_i = np.argmin(cb_eigs, axis=1)  # conduction band minimum kpt index
        vbm_eig = np.max(vb_eigs, axis=1)
        cbm_eig = np.min(cb_eigs, axis=1)
        gap_in = cbm_eig - vbm_eig

        # At this point, decide if we just want to know where the VBM and CBM are
        if ret_vbm_cbm is True:
            return vbm_i, cbm_i, vbm_eig, cbm_eig

        # Determine locations of the valence band minimum and conduction band maximum
        loc_in = np.empty((self.nspins, 2, 3))  # spin, VBM/CBM, coordinates
        loc_dir = np.empty((self.nspins, 3))
        gap_dir = np.empty(self.nspins)
        for ns in range(self.nspins):
            loc_in[ns, 0, :] = self.kpoint_list[vbm_i[ns]]
            loc_in[ns, 1, :] = self.kpoint_list[cbm_i[ns]]

            # Now do the direct gap
            gap_dir[ns] = cb_eigs[ns, vbm_i[ns]] - vb_eigs[ns, vbm_i[ns]]
            loc_dir[ns, :] = self.kpoint_list[vbm_i[ns]]

        # Get the band widths
        vb_width = np.max(vb_eigs, axis=1) - np.min(vb_eigs, axis=1)
        cb_width = np.max(cb_eigs, axis=1) - np.min(cb_eigs, axis=1)

        # Decide on the energy unit
        eng_unit = 'Hartrees'
        if self.convert_to_eV is True:
            eng_unit = 'eV'
        # Now put everything into band_info
        band_info['nelec'] = nelecs
        band_info['nbands'] = nbands
        band_info['vbm'] = vbm_eig
        band_info['cbm'] = cbm_eig
        band_info['gap_indir'] = gap_in
        band_info['loc_indir'] = loc_in
        band_info['gap_dir'] = gap_dir
        band_info['loc_dir'] = loc_dir
        band_info['vb_width'] = vb_width
        band_info['cb_width'] = cb_width
        band_info['eng_unit'] = eng_unit

        # Write out the data in a pretty format
        # Really we should be using f-strings for this but on the off chance someone has an older version of Python...
        if silent is False:
            print('Number of spins:     ', self.nspins)
            print('Number of k-points:  ', self.n_kpoints)
            print('Fermi Energy:        {:.6f} {}'.format(self.Ef, eng_unit))

            # Print information for each spin channel
            for ns in range(self.nspins):
                if (self.nspins == 2):
                    print('Spin Channel {}'.format(ns + 1))
                    print('-' * 50)

                print('No. of electrons: ', nelecs[ns])
                print('No. of bands:     ', nbands[ns])
                kx_vbm, ky_vbm, kz_vbm = loc_in[ns, 0, :]
                kx_cbm, ky_cbm, kz_cbm = loc_in[ns, 1, :]
                print(f'Valence band maximum    (VBM): {vbm_eig[ns]:.6f} {eng_unit}')
                print(f'Conduction band maximum (CBM): {cbm_eig[ns]:.6f} {eng_unit}')
                print('Indirect gap: {:.6f} {}  from {:.6f} {:.6f} {:.6f} --> {:.6f} {:.6f} {:.6f}'.format(
                    gap_in[ns], eng_unit,
                    kx_vbm, ky_vbm, kz_vbm,
                    kx_cbm, ky_cbm, kz_cbm
                ))
                print('Direct gap:   {:.6f} {}  at   {:.6f} {:.6f} {:.6f}'.format(
                    gap_dir[ns], eng_unit,
                    kx_vbm, ky_vbm, kz_vbm,
                ))

                if (self.nspins == 2):
                    print('')
        return band_info

    def _pdos_read(self,
                   species_only=False,
                   popn_select=[None, None],
                   species_and_orb=False,
                   orb_breakdown=False):
        ''' Internal function for reading the pdos_bin file. This contains all of the projected DOS from the Mulliken '''
        # NPBentley: added species_and_orb, allowing for colour plotting by both species and orbital. 18/01/24

        # NPBentley: added the orb_breakdown function, with the aim of splitting the orbitals up into suborbitals.
        # This is not currently fully implemented or tested. 26/02/24

        from scipy.io import FortranFile as FF

        f = FF(self.seed + '.pdos_bin', 'r', '>u4')
        self.pdos_has_read = True

        version = f.read_reals('>f8')
        header = f.read_record('a80')[0]
        num_kpoints = f.read_ints('>u4')[0]
        num_spins = f.read_ints('>u4')[0]
        num_popn_orb = f.read_ints('>u4')[0]
        max_eigenvalues = f.read_ints('>u4')[0]

        orbital_species = f.read_ints('>u4')
        num_species = len(np.unique(orbital_species))
        orbital_ion = f.read_ints('>u4')
        orbital_l = f.read_ints('>u4')

        self.orbital_species = orbital_species
        self.num_species = num_species
        self.orbital_ion = orbital_ion
        self.orbital_l = orbital_l

        kpoints = np.zeros((num_kpoints, 3))
        pdos_weights = np.zeros((num_popn_orb, max_eigenvalues, num_kpoints, num_spins))

        pdos_orb_spec = np.zeros((num_species, 4, max_eigenvalues, num_kpoints, num_spins))

        # NPBentley - Initalise array for containing orbitals subdivided up into their suborbitals in the pdos calculation.
        # 4 corresponds to the
        if orb_breakdown:
            pdos_suborb_spec = np.zeros((num_species, 4, 7, max_eigenvalues, num_kpoints, num_spins))

        # Read the pdos weights from the .pdos_bin file and read them into pdos_weights, before normalising the weights.
        for nk in range(0, num_kpoints):
            record = f.read_record('>i4', '>3f8')
            kpt_index, kpoints[nk, :] = record
            for ns in range(0, num_spins):
                spin_index = f.read_ints('>u4')[0]
                num_eigenvalues = f.read_ints('>u4')[0]

                for nb in range(0, num_eigenvalues):
                    pdos_weights[0:num_popn_orb, nb, nk, ns] = f.read_reals('>f8')
                    norm = np.sum((pdos_weights[0:num_popn_orb, nb, nk, ns]))
                    pdos_weights[0:num_popn_orb, nb, nk, ns] = pdos_weights[0:num_popn_orb, nb, nk, ns] / norm

        pdos_orb_spec = np.zeros((num_species, 4, max_eigenvalues, num_kpoints, num_spins))

        # Sort the weights based on kpoint ordering
        for i in range(len(pdos_weights[:, 0, 0, 0])):
            for nb in range(num_eigenvalues):
                for ns in range(num_spins):
                    pdos_weights[i, nb, :, ns] = pdos_weights[i, nb, :, ns][self.kpt_sort]

        # Return the raw weights - these are divided up into suborbitals as is laid out in the .castep file.
        self.raw_pdos = pdos_weights

        # reshape so we can work out which bands are which - this combines all of the suborbitals for a given species together
        for i in range(len(orbital_species)):

            l_ind = orbital_l[i]
            spec_ind = orbital_species[i] - 1

            pdos_orb_spec[spec_ind, l_ind, :, :, :] = pdos_orb_spec[spec_ind, l_ind, :, :, :] + pdos_weights[i, :, :, :]

        # Go through each kpoint, band and spin to find the species and orbital with highest occupancy.
        # Then we can set it to 1 and all other weights to 0 in order to find the mode,
        # i.e. corresponding species_orbital.
        for nk in range(num_kpoints):
            for nb in range(max_eigenvalues):
                for ns in range(num_spins):
                    max_spec, max_l = np.where(pdos_orb_spec[:, :, nb, nk, ns] == np.max(pdos_orb_spec[:, :, nb, nk, ns]))

                    pdos_orb_spec[:, :, nb, nk, ns] = 0
                    pdos_orb_spec[max_spec[0], max_l[0], nb, nk, ns] = 1

        # Sum over all the kpoints of pdos_orb_spec, in order to give the weights for the given band
        # across the chosen k-point path.
        pdos_bands = np.sum(pdos_orb_spec, axis=3)

        # Define an array used to find which species orbital combination has the max weight for each
        # band. Then associate this species orbital combination to the given band in the band_char array.
        band_char = np.zeros((2, max_eigenvalues, num_spins))

        for nb in range(0, max_eigenvalues):
            for ns in range(0, num_spins):
                max_spec, max_l = np.where(pdos_bands[:, :, nb, ns] == np.max(pdos_bands[:, :, nb, ns]))

                band_char[0, nb, ns] = max_spec[0] + 1  # Define species
                band_char[1, nb, ns] = max_l[0]  # Define orbital

        # Save the band_char array for when plotting by species and orbital.
        if species_and_orb:
            self.band_char = band_char

        # Now filter based on user input
        popn_bands = np.zeros((max_eigenvalues, num_spins), dtype=bool)
        if popn_select[0] is not None:
            for nb in range(max_eigenvalues):
                for ns in range(num_spins):
                    if band_char[0, nb, ns] == popn_select[0] and band_char[1, nb, ns] == popn_select[1]:
                        popn_bands[nb, ns] = 1
            self.popn_bands = popn_bands
            return

        if species_only:
            num_species = len(np.unique(orbital_species))
            pdos_weights_sum = np.zeros((num_species, max_eigenvalues, num_kpoints, num_spins))

            for i in range(0, num_species):
                loc = np.where(orbital_species == i + 1)[0]
                pdos_weights_sum[i, :, :, :] = np.sum(pdos_weights[loc, :, :, :], axis=0)

        else:
            num_orbitals = 4
            pdos_weights_sum = np.zeros((num_orbitals, max_eigenvalues, num_kpoints, num_spins))
            pdos_colours = np.zeros((3, max_eigenvalues, num_kpoints, num_spins))

            r = np.array([1, 0, 0])
            g = np.array([0, 1, 0])
            b = np.array([0, 0, 1])
            k = np.array([0, 0, 0])

            for i in range(0, num_orbitals):
                loc = np.where(orbital_l == i)[0]
                if len(loc) > 0:
                    pdos_weights_sum[i, :, :, :] = np.sum(pdos_weights[loc, :, :, :], axis=0)

        pdos_weights_sum = np.where(pdos_weights_sum > 1, 1, pdos_weights_sum)
        pdos_weights_sum = np.where(pdos_weights_sum < 0, 0, pdos_weights_sum)
        self.pdos = np.round(pdos_weights_sum, 7)

        # NPBentley - Code used for subdividing the orbitals up into their suborbitals in the pdos calculation.  TO DO - complete
        if orb_breakdown:
            suborb_list = []
            try:

                castep_file = self.seed + ".castep"
                castepfile = open(castep_file, 'r')
            except:
                raise Exception("No .castep file")

            castep_lines = castepfile.readlines()
            for line in castep_lines:
                if line.find("Orbital Populations") != -1:
                    orb_pop_index = castep_lines.index(line)

            # print(orb_pop_index)
            orb_info = castep_lines[orb_pop_index + 4:orb_pop_index + 4 + len(orbital_species)]
            orb_mapping = {
                "S": 0,
                "Px": 0,
                "Py": 1,
                "Pz": 2,
                "Dzz": 0,
                "Dzy": 1,
                "Dzx": 2,
                "Dxx-yy": 3,
                "Dxy": 4,
                "Fxxx": 0,
                "Fyyy": 1,
                "Fzzz": 2,
                "Fxyz": 3,
                "Fz(xx-yy)": 4,
                "Fy(zz-xx)": 5,
                "Fx(yy-zz)": 6
            }
            for i in range(len(orbital_species)):
                orb_lab = orb_info[i].split()[2]
                suborb_list.append(orb_lab)
                # print(orb_mapping.get(orb_lab))
                l_ind = orbital_l[i]
                spec_ind = orbital_species[i] - 1

                pdos_suborb_spec[spec_ind, l_ind, orb_mapping.get(orb_lab), :, :, :] = pdos_suborb_spec[spec_ind, l_ind, orb_mapping.get(orb_lab), :, :, :] \
                    + self.raw_pdos[i, :, :, :]
            # NPBentley - this function now needs the functionality to find the dominant suborbital for each band and an associated character array
            # as has been done when the species_and_orb option is used. It would also be useful to integrate this in with the dos plotting, as it
            # likely be more useful to plot suborbitals in dos plots rather than in bands. 26/02/24

    def _gradient_read(self):
        ''' Internal function for reading the gradient file .dome_bin. This is used in the calculation of the adaptive broadening. If using  cite Jonathan R. Yates, Xinjie Wang, David Vanderbilt, and Ivo Souza
        Phys. Rev. B 75, 195121 '''
        from scipy.io import FortranFile as FF
        try:
            f = FF(self.seed + '.dome_bin', 'r', '>u4')
        except:
            raise Exception('Unable to read .dome_bin file, change broadening="gaussian" or "lorentzian".')
        version = f.read_reals('>f8')
        header = f.read_record('a80')[0]

        bands_grad = np.zeros((3, self.nbands, self.n_kpoints, self.nspins))

        for nk in range(0, self.n_kpoints):
            for ns in range(0, self.nspins):
                # for nb in range(0,self.nbands):
                bands_grad[:, :, nk, ns] = f.read_reals('>f8').reshape(3, self.nbands)

        # Convert the gradients to eV
        bands_grad = bands_grad * self.eV * 0.52917720859
        grad_bands_2 = np.sqrt(np.sum((bands_grad**2), axis=0))

        for nb in range(self.nbands):
            for ns in range(self.nspins):
                grad_bands_2[nb, :, ns] = grad_bands_2[nb, :, ns][self.kpt_sort]

        adaptive_weights = grad_bands_2 * self.dk

        adaptive_weights[adaptive_weights < 1e-2] = 1e-2

        self.adaptive_weights = adaptive_weights

    def _split_pdos(self, species, ion=1):
        '''Internal function for splitting the pdos into various components'''

        self.castep_parse()
        self.pdos_read()
        # except:
        #    raise Exception("No .castep file.")
        # Do the masking
        mask = np.where(self.orbital_species == species)[0]
        orbital_species = self.orbital_species[mask]
        orbital_l = self.orbital_l[mask]
        orbital_ion = self.orbital_ion[mask]
        orbital_n = []
        if ion is not None:
            mask2 = np.where(orbital_ion == ion)[0]
            orbital_species = orbital_species[mask2]
            orbital_l = orbital_l[mask2]
            orbital_ion = orbital_ion[mask2]

        sn = self.low_n[species - 1]
        pn = self.low_n[species - 1]
        dn = self.low_n[species - 1]
        fn = self.low_n[species - 1]

        si = 0
        pi = 0
        di = 0
        fi = 0

        s = ['s']
        p = ['p$_{x}$', 'p$_{y}$', 'p$_{z}$']
        d = ['d$_{z^2}$', 'd$_{zy}$', 'd$_{zx}$', 'd$_{x^2-y^2}$', 'd$_{xy}$']
        f = ['f$_{x^3}$', 'f$_{y^3}$', 'f$_{z^3}$', 'f$_{xyz}$', 'f$_{z(x^2-y^2)}$', 'f$_{y(z^2-x^2)}$', 'f$_{x(y^2-z^2)}$']
        labels = ['' for i in range(len(orbital_l))]

        for i in range(len(orbital_l)):
            if i > 0:
                if orbital_ion[i] != orbital_ion[i - 1]:
                    sn = self.low_n[species - 1]
                    pn = self.low_n[species - 1]
                    dn = self.low_n[species - 1]
                    fn = self.low_n[species - 1]

                    si = 0
                    pi = 0
                    di = 0
                    fi = 0

            if orbital_l[i] == 0:
                # s
                if sn <= dn and di > 3:
                    sn = dn + 1
                labels[i] = str(sn) + s[si]
                orbital_n.append(sn)
                sn += 1
            elif orbital_l[i] == 1:
                if pi > 2:
                    pi = 0
                    pn += 1
                if pn <= dn and di > 3:
                    pn = dn + 1
                labels[i] = str(pn) + p[pi]
                orbital_n.append(pn)
                pi += 1
            elif orbital_l[i] == 2:
                if di > 4:
                    di = 0
                    dn += 1
                labels[i] = str(dn) + d[di]
                orbital_n.append(dn)
                di += 1
            elif orbital_l[i] == 1:
                if fi > 6:
                    fi = 0
                    fn += 1
                labels[i] = str(fn) + f[fi]
                orbital_n.append(fn)
                fi += 1

        labels = labels
        return labels

    def _plot_gle(self, spin_polarised=False, spin_index=[0], species_and_orb=False):
        '''Function for getting data into a GLE readable format and producing the template for a gle input file so that it can be used to produce
        band structures.
        :param: spin_polarised: Indicate if the system is spin polarised or not.
        :param: spin_index: Indicates spin component desired for plotting.
        :param: species_and_orb: Indicate if a plot highlighting the majority orbital and species
        of a given band is wanted.
        '''

        ns_gle = 0
        if spin_polarised:
            if spin_index[0] == 0:
                filename = "spin_up"
                gle_color = "red"
                ns_gle = spin_index[0]
            else:
                filename = "spin_down"
                gle_color = "blue"
                ns_gle = spin_index[0]
        else:
            filename = "gle_bands"
            gle_color = "black"

        if species_and_orb:
            filename = filename + "_specorb"

        gle_data = np.zeros((len(self.kpoints), self.nbands + 1))
        gle_data[:, 0] = self.kpoints
        gle_data[:, 1:] = np.swapaxes(self.BandStructure[:self.nbands, :, ns_gle], 0, 1)
        np.savetxt(filename + ".dat", gle_data)

        gle_graph = open(filename + ".gle", "w")
        gle_graph.write('size 10 6')
        # include the package for drawing horizontal and vertical lines, to highlight the Fermi energy and high symmetry kpoints.
        gle_graph.write('\ninclude "graphutil.gle"')
        gle_graph.write('\nset font texcmr')
        gle_graph.write('\n\nf_e = 0')
        gle_graph.write('\n\nsub graph_Fermi')
        gle_graph.write('\n   set lstyle 3 just lc')
        gle_graph.write('\n   graph_hline f_e')
        gle_graph.write('\nend sub')
        gle_graph.write('\n\nsub graph_vline x y1 y2')
        gle_graph.write('\n   default y1 ygmin')
        gle_graph.write('\n   default y2 ygmax')
        gle_graph.write('\n   amove xg(x) yg(y1)')
        gle_graph.write('\n   aline xg(x) yg(y2)')
        gle_graph.write('\nend sub')
        gle_graph.write('\n\nbegin graph\n')
        gle_graph.write(r'   ytitle "\tex{$E - E_{\rm F}$} (eV)"')
        gle_graph.write('\n   xaxis max ' + str(len(self.kpoints)))
        gle_graph.write('\n   xlabels off')
        gle_graph.write('\n   xticks off')
        gle_graph.write('\n   yaxis nticks 5 min -2 max 2')
        gle_graph.write('\n   ysubticks off')
        gle_graph.write('\n   data "' + filename + '.dat"')
        if not species_and_orb:
            gle_graph.write('\n   for alpha = 1 to ' + str(self.nbands))
            gle_graph.write('\n      d[alpha] line color ' + gle_color)
            gle_graph.write('\n   next alpha')
        else:
            for i in range(self.nbands):
                gle_graph.write(
                    '\n   d' + str(i) + ' line color spec'
                    + str(int(self.band_char[0, i, 0])) + '_orb'
                    + str(int(self.band_char[1, i, 0])))
        gle_graph.write('\n   draw graph_Fermi')
        gle_graph.write('\nend graph')
        gle_graph.close()

    def plot_bs(self,
                ax,
                mono=False,
                mono_color='k',
                spin_polarised=False,
                spin_up_color='red',
                spin_down_color='blue',
                spin_up_color_hi='black',
                spin_down_color_hi='black',
                pdos=False,
                fontsize=20,
                cmap='tab20c',
                show_fermi=True,
                fermi_line_style="--",
                fermi_line_color='0.5',
                fermi_linewidth=1,
                linestyle="-",
                linewidth=1.2,
                sym_lines=True,
                spin_index=None,
                Elim=None,
                klim=None,
                axes_only=False,
                pdos_species=False,
                pdos_popn_select=[None, None],
                band_ids=None,
                band_colors=None,
                output_gle=False,
                species_and_orb=False,
                orb_breakdown=False,
                mark_gap=False,
                mark_gap_color=None,
                mark_gap_headwidth=None,
                mark_gap_linewidth=None,
                band_labels=None,
                legend_loc='best'
                ):
        """Plot the band structure from a .bands file.

        Parameters
        ----------
        ax : matplotlib axes
            The axis object to which the band structure should be added.
            Multiple bandstructures can be plotted on top of each other by specifying the same axes.
        mono : boolean
            Make bandstructure a single colour (default : False)
        mono_color : string
            Colour to use if band structure is a single colour (default : black).
        spin_polarised : boolean
            Is .bands for a spin polarised calculation? (default: Set by bands data)
        spin_up_color : string
            colour to use for spin up channel (default : red)
        spin_down_color : string
            colour to use for spin down channel (default : blue)
        spin_up_color_hi : string
            mono colour for spin up channel (default : black)
        spin_down_color_hi : string
            mono colour for spin down channel (default : black)
        pdos : boolean
            Perform Mulliken projections to project out bands by the orbitals.
        fontsize : integer
            Fontsize to use in the plot
        cmap : matplotlib colourmap
            colour map to use in partial density of states.
        show_fermi : boolean
            Show the Fermi energy of the plot
        fermi_line_style : string
            matplotlib line style to use for the Fermi energy line
        fermi_line_color : string
            matplotlib colour to use for Fermi energy line
        fermi_linewidth : float
            line width to use for Fermi energy
        linestyle : string
            matplotlib line style to use when plotting the band structure.
        linewidth : float
            width of lines when plotting each band.
        sym_lines : boolean
            Show lines indicating high-symmetry points in the Brillouin zone.
        spin_index : integer
            Plot only a specific spin channel (NB: indices are specified in C/Python convention)
        Elim : ndarray(dtype=float)
            Limit of energy scale (in units of plot)
        klim : list(dtype=str) or list(dtype=float) - see below
            Limit of k-point axes.
            This can be specified in by either the high-symmetry point labels (string) or kpoint index (integer)
        axes_only : boolean
            Return the formatted band structure axes (including high symmetry lines if requested)
            but do not actually plot the band structure or density of states.
        pdos_species : ndarray(dtype=int)
            atoms (indexed from 0) to include in the partial density of states.
        pdos_popn_select : ndarray
            population analysis
        band_ids : ndarray(dtype=int)
            plot only specific bands in the band structure.
        band_colors : list(dtype=str)
            colours to use when plotting the bands according to band_ids.
        output_gle : Boolean
            Produce the output ".dat" and ".gle" files for plotting bandstructures using GLE.
        species_and_orb : Boolean
            Produce the output ".dat" and ".gle" files for producing a plot where the bands are
            colour coded by majority orbital and species.
        orb_breakdown : Boolean
            Identify the bands by suborbital, ready for plotting (when NPBentley gets around to
            implementing it).
        mark_gap : boolean
            mark the band gap on the plot
        mark_gap_color : string or list(dtype=string)
            colours to use when marking the band gap (list to specify for each spin channel)
            (default : red for spin 1, blue for spin 2)
        mark_gap_headwidth : float or list(dtype=float)
            width of arrow head used to mark the gap. (list to specify for each spin channel)
            (default : 0.75 for both spin channels)
        mark_gap_linewidth : float or list(dtype=float)
            width of arrow tail used to mark the gap. (list to specify for each spin channel)
            (default : 0.15 for both spin channels)
        band_labels : list(dtype=str)
            labels for specific bands specified by bands_ids.
        legend_loc : str or pair of floats
            position of the legend for the plot if a legend is created (default: 'best')

        Raises
        ------
        Exception
            Population analysis is unavailable in a monochromatic plot.

        """

        ''' Function for plotting a Band structure, provide an ax object'''
        import matplotlib

        # cycle_color = plt.get_cmap(cmap).colors
        # plt.rcParams['axes.prop_cycle'] = cycler(color=cycle_color)
        # Set dedaults for spins

        if self.spin_polarised and spin_index is None:
            spin_index = [0, 1]
        elif not self.spin_polarised and spin_index is None:
            spin_index = [0]
        if spin_index is not None:
            if not isinstance(spin_index, list):
                spin_index = [spin_index]
                spin_polarised = True
        # spin colors
        if spin_polarised:
            spin_colors = [spin_up_color, spin_down_color]
            spin_colors_select = [spin_up_color_hi, spin_down_color_hi]

        # Set up the band ids
        band_ids_mask = np.ones((self.nbands, self.nspins), dtype=bool)
        if band_ids is not None:
            band_ids = np.array(band_ids)
            if band_ids.ndim == 2:
                # We have different spins for the different bands
                for nb in range(self.nbands):
                    for ns in range(self.nspins):

                        if nb not in band_ids[:, ns]:
                            band_ids_mask[nb, ns] = False
            elif band_ids.ndim == 1:
                # We have only one spin
                for nb in range(self.nbands):
                    for ns in spin_index:
                        if nb not in band_ids[:]:
                            band_ids_mask[nb, ns] = False

        # Set the boring stuff
        # V Ravindran 26/02/2024 - tidied up if statement for the y-axis  label
        eng_unit = 'Ha'
        if self.convert_to_eV is True:
            eng_unit = 'eV'
        eng_label = 'E'
        if self.zero_fermi is True:
            eng_label = r'E-E$_{\mathrm{F}}$'
        elif self.zero_vbm is True:
            eng_label = r'E-E$_{\mathrm{VBM}}$'
        elif self.zero_cbm is True:
            eng_label = r'E-E$_{\mathrm{CBM}}$'
        ax.set_ylabel(eng_label + f' ({eng_unit})', fontsize = fontsize)

        ax.set_xlim(1, len(self.kpoints))
        ax.tick_params(axis='both', direction='in', which='major', labelsize=fontsize * 0.8, length=12, width=1.2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize * 0.8, length=6,
                       right=True, top=False, bottom=False, left=True, width=1.2)
        ax.set_xticks(self.high_sym)
        ax.set_xticklabels(self.high_sym_labels)
        ax.minorticks_on()

        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

        def _get_klim(self, user_klim):
            """Helper function to parse the user's kpoint limits
            Author : V Ravindran 01/04/2024
            """
            if len(user_klim) != 2:
                raise IndexError('You must pass both the lower and upper kpoint limit')

            plot_klim = np.empty(2, dtype=int)

            if isinstance(user_klim[0], str) is True:
                # Using high-symmetry points to decide label
                for n, label in enumerate(user_klim):
                    if label.strip() == 'G':
                        label = r'$\Gamma$'
                    indx = [i for i, pt in enumerate(self.high_sym_labels) if label == pt]

                    if len(indx) > 1:
                        # If a high symmetry point appears more than once, let the user choose interactively.
                        print('{} point appears more than once in band structure.'.format(label))
                        print('Path in calculation: ', end='')
                        for i, pt in enumerate(self.high_sym_labels):
                            if pt == r'$\Gamma$':
                                pt = 'G'
                            if i == len(self.high_sym_labels) - 1:
                                print(pt)
                            else:
                                print(pt, end='->')

                        print('Choose index from the following: ', end=' ')
                        for i, label_i in enumerate((indx)):
                            print('{}: {}.kpt'.format(i, label_i + 1), end='   ')
                        indx = input('')
                        indx = int(indx)
                    elif len(indx) == 0:
                        raise IndexError('{} not in high symmetry points'.format(label))
                    else:
                        indx = indx[0]

                    # Now get the actual kpoint index associated with the desired high-symmetry point.
                    plot_klim[n] = self.high_sym[indx]
            else:
                # Set index by kpoint number - NB: C/Python ordering!
                plot_klim = np.array(user_klim, dtype=int)
            return plot_klim

        # energy lims
        if Elim is not None:
            ax.set_ylim(Elim[0], Elim[1])

        # kpoint/Brillouin zone limits V Ravindran - 01/02/2024
        if klim is not None:
            plot_klim = _get_klim(self, klim)
            ax.set_xlim(plot_klim[0], plot_klim[1])

        # Add in all the lines
        if show_fermi:
            ax.axhline(self.Ef, linestyle=fermi_line_style, c=fermi_line_color, linewidth=fermi_linewidth)

        if sym_lines:
            for i in self.high_sym:
                ax.axvline(i, color='k', linewidth=1)

        # We have set up the axes, not we terminate if the user only wants axes
        if axes_only:
            return

        def _setup_str_mask(user_strings, band_ids_mask):
            """Setup a character array according to the bands mask.

            The same convention is followed as above with regard to spin channels.
            Author : V Ravindran, 12/04/2024
            """
            # Strip any whitespace.
            user_strings = np.char.strip(np.array(user_strings))

            # Get the bands to label - spins done in the loop below
            do_bands_b = np.where(band_ids_mask)[0]
            if band_ids_mask.ndim == 2:
                # Decided to label for different bands in different spin channels
                do_bands_s = np.where(band_ids_mask)[1]

            assert len(user_strings) == do_bands_b.shape[0]

            # Now set up the labels following the mask provided.
            string_masked = np.empty(band_ids_mask.shape, dtype=user_strings.dtype)
            for i in range(len(do_bands_b)):
                nb = do_bands_b[i]
                if band_ids_mask.ndim == 2:
                    ns = do_bands_s[i]
                    # A label was provided for different bands across different spins
                    string_masked[nb, ns] = user_strings[i]
                else:
                    # Do the label only for a specific spin channel based on spin_index provided
                    for do_spin in spin_index:
                        string_masked[nb, do_spin] = user_strings[i]

            return string_masked

        def _check_user_input_str_mask(user_strings, band_ids, errstr):
            """Check the user provided the appropriate format for strings.

            This function should be called before actually getting the masked character array
            using _setup_str_mask.
            Author : V Ravindran, 14/04/2024
            """
            # Check the user provided band_ids.
            # Although the local variable is band_ids_mask, this is initialised to
            # true everywhere so we cannot actually mask the actual character array according to bands.
            # Therefore, we have to make sure it is actually provided and check against the user's
            # data, NOT the local copy.
            errstr = errstr.strip()
            if band_ids is None:
                raise TypeError(
                    f'You must supply bands to band_ids to indicate which bands to {errstr}'
                )
            elif np.array(user_strings).ndim != 1:
                raise IndexError(
                    f'Band {errstr}s should be a 1D array array with length of number of bands to act on.'
                )
            elif band_ids.shape[0] != len(user_strings):
                raise IndexError(
                    f'Number of bands {band_ids.shape[0]} does not match number of {errstr}s {len(user_strings)}.'
                )

        # Check if we want to use user-defined colours on a per-band-basis - band_colors V Ravindran 14/04/2024
        if band_colors is not None:
            # Check the user gave the data in the correct format/data structure.
            # NB must check against user input not the local copy (band_ids_mask)
            _check_user_input_str_mask(band_colors, band_ids, 'colour')

            # Set up colours according to the band_ids_mask
            band_colors = _setup_str_mask(band_colors, band_ids_mask)

        # Check if we want to use custom labels for the bands - band_labels V Ravindran 12/04/2024
        if band_labels is not None:
            # Check the user gave the data in the correct format/data structure.
            # NB must check against user input not the local copy (band_ids_mask)
            _check_user_input_str_mask(band_labels, band_ids, 'label')

            # Set up labels according to the band_ids_mask
            band_labels = _setup_str_mask(band_labels, band_ids_mask)

        # Do the standard plotting, no pdos here
        if not pdos:
            # Here we plot all of the bands. We can provide a mechanism latter for plotting invididual ones

            # Store the lines for each band in a list - band_labels V Ravindran 12/04/2024
            # so we can add them to legend when labelling. -  band_labels V Ravindran 12/04/2024
            custom_lines = []
            l_labels = []  # local copy of band_labels as a list
            for nb in range(self.nbands):
                for ns in spin_index:

                    if not band_ids_mask[nb, ns]:
                        # If the band is not within the mask,
                        # then do not plot it and skip to the next band.
                        continue
                        # Mono

                    # Updated colour selection V Ravindran 14/04/2024
                    line_color = ''
                    if mono:
                        line_color = mono_color
                    elif band_colors is not None:
                        line_color = band_colors[nb, ns]
                    elif spin_polarised:
                        line_color = spin_colors[ns]

                    if line_color == '':
                        # No colour specified so let pick from rcParams
                        line, *_ = ax.plot(self.kpoints, self.BandStructure[nb, :, ns],
                                           linestyle=linestyle, linewidth=linewidth)
                    else:
                        line, *_ = ax.plot(self.kpoints, self.BandStructure[nb, :, ns],
                                           linestyle=linestyle, linewidth=linewidth, color=line_color)

                    if band_labels is not None:  # band_labels V Ravindran 12/04/2024
                        # V Ravindran: The check further up should have caught the fact that band_ids
                        # needs to be supplied together with band_labels.
                        # This *should* prevent every band from being labelled.
                        # Thus only bands we want to label should be added.
                        #
                        # Unfortunately, it requires this routine to be called twice,
                        # once for the overall band structure, and the second for the labels...
                        custom_lines.append(line)
                        l_labels.append(band_labels[nb, ns])

            # Add a legend with the band labels if requested band_labels V Ravindran 12/04/2024
            if band_labels is not None:
                ax.legend(custom_lines, l_labels, loc=legend_loc)
                # Return the lines and labels so the user can customise the legend how they wish

            if output_gle:
                self._plot_gle(spin_polarised, spin_index)

        # now pdos is a thing
        else:
            # calculate the pdos if needed
            self._pdos_read(pdos_species, pdos_popn_select, species_and_orb, orb_breakdown)

            # first do the plotting with the popn_select
            if pdos_popn_select[0] is not None:
                for nb in range(self.nbands):
                    for ns in spin_index:
                        if not band_ids_mask[nb, ns]:
                            continue

                        # Mono
                        if mono:
                            if self.popn_bands[nb, ns]:
                                ax.plot(self.kpoints, self.BandStructure[nb, :, ns],
                                        linestyle=linestyle, linewidth=linewidth, color=mono_color_select)

                            else:
                                ax.plot(self.kpoints, self.BandStructure[nb, :, ns],
                                        linestyle=linestyle, linewidth=linewidth, color=mono_color)

                        elif spin_polarised:
                            if self.popn_bands[nb, ns]:
                                ax.plot(self.kpoints, self.BandStructure[nb, :, ns], linestyle=linestyle,
                                        linewidth=linewidth, color=spin_colors_select[ns])
                            else:
                                ax.plot(self.kpoints, self.BandStructure[nb, :, ns],
                                        linestyle=linestyle, linewidth=linewidth, color=spin_colors[ns])
                        else:
                            raise Exception("Highlighting by population analysis unavailable for non-mono plots.")

            elif (species_and_orb and output_gle):
                self._plot_gle(spin_polarised, spin_index, species_and_orb)

            else:
                import matplotlib.collections as mcoll
                import matplotlib.path as mpath
                from matplotlib import colors
                from matplotlib.colors import ListedColormap
                from matplotlib.lines import Line2D

                # Define the colours we'll use for the plotting
                n_colors = cycle(['blue', 'red', 'green', 'black', 'purple', 'orange', 'yellow', 'cyan'])

                def make_segments(x, y):
                    """
                    Create list of line segments from x and y coordinates, in the correct format
                    for LineCollection: an array of the form numlines x (points per line) x 2 (x
                    and y) array
                    """

                    points = np.array([x, y]).T.reshape(-1, 1, 2)

                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    return segments

                def colorline(
                        x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
                        linewidth=3, alpha=1.0):
                    """
                    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
                    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
                    Plot a colored line with coordinates x and y
                    Optionally specify colors in the array z
                    Optionally specify a colormap, a norm function and a line width
                    """

                    # Default colors equally spaced on [0,1]:
                    if z is None:
                        z = np.linspace(0.0, 1.0, len(x))
                    z = np.asarray(z)
                    segments = make_segments(x, y)
                    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                              linewidth=linewidth, alpha=alpha)
                    ax.add_collection(lc)
                    return lc

                if pdos_species:
                    n_cat = len(self.atoms)
                else:
                    n_cat = 4

                basis = []
                for i in range(n_cat):
                    basis.append(np.array(colors.to_rgba(next(n_colors))))

                for nb in range(self.nbands):
                    for ns in spin_index:

                        if not band_ids_mask[nb, ns]:
                            continue

                        # calculate the colour
                        cmap_array = np.zeros((len(self.kpoints), 4))
                        for i in range(n_cat):

                            cmap_array[:, 0] += self.pdos[i, nb, :, ns] * basis[i][0]  # /n_cat
                            cmap_array[:, 1] += self.pdos[i, nb, :, ns] * basis[i][1]  # /n_cat
                            cmap_array[:, 2] += self.pdos[i, nb, :, ns] * basis[i][2]  # /n_cat
                            cmap_array[:, 3] += self.pdos[i, nb, :, ns] * basis[i][3]  # /n_cat

                            # cmap_array[:,0:3]=cmap_array[:,0:3]/n_cat
                            cmap_array = np.where(cmap_array > 1, 1, cmap_array)
                            cmap = ListedColormap(cmap_array)

                        z = np.linspace(0, 1, len(self.kpoints))
                        colorline(self.kpoints, self.BandStructure[nb, :, ns], z, cmap=cmap, linewidth=3)
                        ax.plot(self.kpoints, self.BandStructure[nb, :, ns], linewidth=linewidth, alpha=0)

                custom_lines = []
                labels = []
                for i in range(n_cat):
                    custom_lines.append(Line2D([0], [0], color=basis[i], lw=3))
                    if pdos_species:
                        labels.append(self.atoms[i])
                    else:
                        labels = ["s", "p", "d", "f"]

                ax.legend(custom_lines, labels, fontsize=fontsize)

        # Mark the band gap on the plot V Ravindran 31/04/2024
        if mark_gap is True:
            vbm_i, cbm_i, *_ = self.get_band_info(ret_vbm_cbm=True)
            # Decide on occupancies for each band depending on spin polarised or not
            if self.spin_polarised is True:
                nelec = np.array([self.nup, self.ndown], dtype=int)
                occ = 1
            else:
                nelec = np.array([self.electrons], dtype=int)
                occ = 2

            # Now decide on aesthetics
            if mark_gap_color is not None:
                if isinstance(mark_gap_color, list):
                    if len(mark_gap_color) != len(spin_index):
                        raise IndexError('You need to provide a colour for each spin channel')
                elif isinstance(mark_gap_color, str):
                    col = mark_gap_color
                    mark_gap_color = [col, col]
                else:
                    raise TypeError('mark_gap_color not string or list.')
            else:
                mark_gap_color = ['red', 'blue']

            if mark_gap_headwidth is not None:
                if isinstance(mark_gap_headwidth, list):
                    if len(mark_gap_headwidth) != len(spin_index):
                        raise IndexError('You need to provide a headwidth for each spin channel')
                elif isinstance(mark_gap_headwidth, float):
                    arrw = mark_gap_headwidth
                    mark_gap_headwidth = [arrw, arrw]
                else:
                    raise TypeError('mark_gap_headwidth not a float or list.')
            else:
                mark_gap_headwidth = [0.75, 0.75]

            if mark_gap_linewidth is not None:
                if isinstance(mark_gap_linewidth, list):
                    if len(mark_gap_linewidth) != len(spin_index):
                        raise IndexError('You need to provide a linewidth for each spin channel')
                elif isinstance(mark_gap_linewidth, float):
                    lw = mark_gap_linewidth
                    mark_gap_linewidth = [lw, lw]
                else:
                    raise TypeError('mark_gap_linewidth not a string or list.')
            else:
                mark_gap_linewidth = [0.15, 0.15]

            # Finally, mark the gaps
            for ns in spin_index:
                vb_eigs = self.BandStructure[int(nelec[ns] / occ) - 1, :, ns]
                cb_eigs = self.BandStructure[int(nelec[ns] / occ), :, ns]

                vbm_k, vbm_eng = self.kpoints[vbm_i[ns]], vb_eigs[vbm_i[ns]]
                cbm_k, cbm_eng = self.kpoints[cbm_i[ns]], cb_eigs[cbm_i[ns]]
                ax.scatter(vbm_k, vbm_eng, color=mark_gap_color[ns])
                ax.scatter(cbm_k, cbm_eng, color=mark_gap_color[ns])
                ax.arrow(vbm_k, vbm_eng, cbm_k - vbm_k, cbm_eng - vbm_eng,
                         width=mark_gap_linewidth[ns],
                         head_width=mark_gap_headwidth[ns],
                         color=mark_gap_color[ns],
                         length_includes_head=True,
                         zorder=50000  # HACK Set this really high and hope this lies on top of everything.
                         )

        return

    def pdos_filter(self, species, l, ion=None):
        ''' Function for filtering the pdos for a particular species, ion and angular momentum'''
        ls = np.where(self.orbital_l == l)[0]
        ss = np.where(self.orbital_species == species + 1)[0]

        cross = np.intersect1d(ls, ss)

        if ion is not None:
            ions = np.where(self.orbital_ion == ion)[0]
            cross = np.intersect1d(cross, ions)
        return self.raw_pdos[cross, :, :, :]

    def plot_dos(self,
                 ax,
                 spin_polarised=None,
                 color='black',
                 spin_up_color='red',
                 spin_down_color='blue',
                 spin_share_axis=False,
                 dE=None,
                 fontsize=20,
                 cmap='tab20c',
                 show_fermi=True,
                 fermi_line_style="--",
                 fermi_line_color='0.5',
                 fermi_linewidth=1,
                 zero_line=True,
                 zero_linestyle="-",
                 zero_linewidth=1,
                 zero_line_color="black",
                 linestyle="-",
                 linewidth=1.2,
                 spin_index=None,
                 Elim=None,
                 glim=None,
                 swap_axes=False,
                 axes_only=False,
                 labelx=True,
                 labely=True,
                 pdos=False,
                 pdos_colors=None,
                 show_total=False,
                 pdos_species=None,
                 pdos_orbitals=None,
                 shade=False,
                 alpha=0.4,
                 temperature=None,
                 broadening="adaptive",
                 width=0.05,
                 loc='upper right'):
        ''' Function for calculating and plotting a DOS '''

        def _fermi_dirac(T, E):

            if T == None:
                return 1.0

            elif T == 0:
                fd = np.ones(len(E))
                fd[E > self.Ef] = 0
                return fd
            else:
                K = 8.617333e-5
                beta = 1 / (K * T)

                return 1 / (np.exp(beta * (E - self.Ef)) + 1)

        def _gaussian(Ek, E, width):
            dist = Ek - E
            mask = np.where(np.abs(dist) < 5 / self.eV)
            result = np.zeros(np.shape(dist))
            factor = 1 / (width * np.sqrt(2 * np.pi))
            exponent = np.exp(-0.5 * np.square(dist[mask] / (width)))
            result[mask] = factor * exponent
            return result

        def _adaptve(Ek, E, width):
            dist = Ek - E
            mask = np.where(np.abs(dist) < 5 / self.eV)
            result = np.zeros(np.shape(dist))
            factor = 1 / (width * np.sqrt(2 * np.pi))
            exponent = np.exp(-0.5 * np.square(dist[mask] / (width[mask])))
            result[mask] = factor * exponent
            return result

        def _lorentzian(Ek, E, width=width):
            return 1 / (np.pi * width) * (width**2 / ((Ek - E)**2 + width**2))

        def _adaptive(Ek, E, width):
            return 1 / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((Ek - E) / (width))**2)

        self.start_time = time.time()
        # Set dedaults for spins
        if spin_polarised is None:
            spin_polarised = self.spin_polarised
        if self.spin_polarised and spin_index is None:
            spin_index = [0, 1]
        elif not self.spin_polarised and spin_index is None:
            spin_index = [0]
        if spin_index is not None:
            if not isinstance(spin_index, list):
                spin_index = [spin_index]
                spin_polarised = True
        # spin colors
        if spin_polarised:
            spin_colors = [spin_up_color, spin_down_color]

        # Orbital defs
        orbs = ['$s$', '$p$', '$d$', '$f$']
        # Set energy spacing
        if dE is None:
            dE = self.dk

        # Set the boring stuff
        if swap_axes:

            # V Ravindran 26/02/2024 - tidied up if statement for the y-axis label
            eng_unit = 'Ha'
            if self.convert_to_eV is True:
                eng_unit = 'eV'
            eng_label = 'E'
            if self.zero_fermi is True:
                eng_label = r'E-E$_{\mathrm{F}}$'
            elif self.zero_vbm is True:
                eng_label = r'E-E$_{\mathrm{VBM}}$'
            elif self.zero_cbm is True:
                eng_label = r'E-E$_{\mathrm{CBM}}$'
            ax.set_ylabel(eng_label + f' ({eng_unit})', fontsize = fontsize)

            ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.8, length=12, width=1.2)
            ax.tick_params(axis='both', which='minor', labelsize=fontsize * 0.8, length=6,
                           right=True, top=False, bottom=False, left=True, width=1.2)

            ax.set_xlabel(r"$\mathit{g}(\mathit{E}$) (states/eV)", fontsize=fontsize)
        else:
            if self.convert_to_eV and self.zero_fermi:
                ax.set_xlabel(r"E-E$_{\mathrm{F}}$ (eV)", fontsize=fontsize)
            elif not self.convert_to_eV and self.zero_fermi:
                ax.set_xlabel(r"E-E$_{\mathrm{F}}$ (Ha)", fontsize=fontsize)
            elif not self.convert_to_eV and not self.zero_fermi:
                ax.set_xlabel(r"E (Ha)", fontsize=fontsize)
            elif self.convert_to_eV and not self.zero_fermi:
                ax.set_xlabel(r"E (eV)", fontsize=fontsize)

            ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.8, length=12, width=1.2)
            ax.tick_params(axis='both', which='minor', labelsize=fontsize * 0.8, length=6,
                           right=True, top=False, bottom=True, left=True, width=1.2)

            ax.set_ylabel(r"$\mathit{g}(\mathit{E}$) (states/eV)", fontsize=fontsize)

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

        if not labely:
            ax.set_ylabel('')
        if not labelx:
            ax.set_xlabel('')

        # energy lims
        if Elim is not None:
            if swap_axes:
                ax.set_ylim(Elim[0], Elim[1])
            else:
                ax.set_xlim(Elim[0], Elim[1])
        if glim is not None:
            if swap_axes:
                ax.set_xlim(glim[0], glim[1])
            else:
                ax.set_ylim(glim[0], glim[1])

        # Add in all the lines
        if show_fermi:
            if swap_axes:
                ax.axhline(self.Ef, linestyle=fermi_line_style, c=fermi_line_color)
            else:
                ax.axvline(self.Ef, linestyle=fermi_line_style, c=fermi_line_color)
        if zero_line:
            if swap_axes:
                ax.axvline(0, linestyle=zero_linestyle, linewidth=zero_linewidth, color=zero_line_color)
            else:
                ax.axhline(0, linestyle=zero_linestyle, linewidth=zero_linewidth, color=zero_line_color)
        # We have set up the axes, not we terminate if the user only wants axes

        if axes_only:
            return

        # Set up the calculation limits
        E = np.arange(np.min(self.BandStructure), np.max(self.BandStructure), dE)
        # E = np.arange(-2,2,dE)

        dos = np.zeros((len(E), self.nspins))
        spin_dir = np.ones((len(E), self.nspins))
        all_dos = np.zeros((self.nbands, self.n_kpoints, self.nspins, len(E)))

        # Initialise the class object to prevent recaculation
        recalculate_dos = True
        recalculate_pdos = True
        try:
            if self.all_dos.shape == all_dos.shape:
                recalculate_dos = False
                all_dos = self.all_dos
        except:
            recalculate_dos = True

        if broadening == 'adaptive':
            print("Please cite 'Jonathan R. Yates, Xinjie Wang, David Vanderbilt, and Ivo Souza Phys. Rev. B 75, 195121' in all publications including these DOS.")
            self._gradient_read()

        # Set up the pdos stuff
        if pdos:

            if not self.pdos_has_read:
                self._pdos_read()

            if pdos_species is None:
                pdos_species = np.arange(0, len(self.atoms))
            if pdos_orbitals is None:
                pdos_orbitals = np.array([0, 1, 2, 3])
            pdos_dos = np.zeros((len(self.atoms), 4, len(E), self.nspins))
            try:
                if self.pdos_dos.shape == pdos_dos.shape:
                    recalculate_pdos = False
                    pdos_dos = self.pdos_dos
            except:
                recalculate_pdos = True

        if len(spin_index) == 2 and spin_polarised:
            if spin_share_axis:
                spin_dir[:, 1] = 1
            else:
                spin_dir[:, 1] = -1

        # We now decide if we are plotting a pdos or not
        if not pdos:
            '''
            for s in spin_index:

                if broadening=='adaptive':
                    # we are going to brute force it for now
                    for nb in range(self.nbands):
                        for nk in range(self.n_kpoints):
                            dos[:,s]+=_adaptive(self.BandStructure[nb,nk,s],E,self.adaptive_weights[nb,nk,s])*self.kpt_weights[nk]
                elif broadening=='gaussian':
                    # we are going to brute force it for now
                    for nb in range(self.nbands):
                        for nk in range(self.n_kpoints):
                            dos[:,s]+=_gaussian(self.BandStructure[nb,nk,s],E,width)*self.kpt_weights[nk]
                elif broadening=='lorentzian':
                    # we are going to brute force it for now
                    for nb in range(self.nbands):
                        for nk in range(self.n_kpoints):
                            dos[:,s]+=_lorentzian(self.BandStructure[nb,nk,s],E)*self.kpt_weights[nk]


                else:
                    raise Exception('Unknown broadening scheme')
            '''
            if recalculate_dos:
                if broadening == 'gaussian':
                    # Lets change the shape of the bandstructure first
                    new_bs = np.repeat(self.BandStructure[:, :, :, np.newaxis], len(E), axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:, np.newaxis], self.nspins, axis=1)
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:, :, np.newaxis], len(E), axis=2)

                    for nb in range(self.nbands):
                        for ns in range(self.nspins):
                            all_dos[nb, :, ns, :] = _gaussian(new_bs[nb, :, ns, :], E, width) * new_kpt_w[:, ns]

                elif broadening == 'adaptive':
                    # Lets change the shape of the bandstructure first
                    new_bs = np.repeat(self.BandStructure[:, :, :, np.newaxis], len(E), axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:, np.newaxis], self.nspins, axis=1)
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:, :, np.newaxis], len(E), axis=2)
                    new_weights = np.repeat(self.adaptive_weights[:, :, :, np.newaxis], len(E), axis=3)

                    for nb in range(self.nbands):
                        for ns in range(self.nspins):
                            all_dos[nb, :, ns, :] = _adaptive(new_bs[nb, :, ns, :], E, new_weights[nb, :, ns, :]) * new_kpt_w[:, ns]
                # Done, now store for next time
                self.all_dos = all_dos
                # print("Recaculated")

            # Multiply in the FD and spin flips
            dos = np.sum(np.sum(all_dos, axis=0), axis=0)
            dos = np.swapaxes(dos, 0, 1)

            dos = dos * spin_dir
            dos = dos * np.expand_dims(_fermi_dirac(temperature, E), axis=-1)

            # Sum over spins if not spin_polarised
            if not spin_polarised and self.spin_polarised:
                dos = np.sum(dos, axis=1)

            # Actually plot now
            if not swap_axes:

                if spin_polarised:
                    for s in spin_index:
                        ax.plot(E, dos[:, s], linestyle=linestyle, linewidth=linewidth, color=spin_colors[s])
                        if shade:
                            ax.fill_between(E, dos[:, s], color=spin_colors[s], alpha=alpha)

                else:
                    ax.plot(E, dos, linestyle=linestyle, linewidth=linewidth, color=color)
                    if shade:
                        ax.fill_between(E, dos, color=color, alpha=alpha)
            else:
                if spin_polarised:
                    for s in spin_index:
                        ax.plot(dos[:, s], E, linestyle=linestyle, linewidth=linewidth, color=spin_colors[s])
                        if shade:
                            ax.fill_betweenx(E, dos[:, s], color=spin_colors[s], alpha=alpha)

                else:
                    ax.plot(dos, E, linestyle=linestyle, linewidth=linewidth, color=color)
                    if shade:
                        ax.fill_betweenx(E, dos, color=color, alpha=alpha)

        else:
            '''
            for s in spin_index:
                for nb in range(self.nbands):
                    for nk in range(self.n_kpoints):
                        if broadening=='adaptive':
                            # we are going to brute force it for now
                            temp_dos=_adaptive(self.BandStructure[nb,nk,s],E,self.adaptive_weights[nb,nk,s])*self.kpt_weights[nk]
                        elif broadening=='gaussian':
                            # we are going to brute force it for now
                            temp_dos=_gaussian(self.BandStructure[nb,nk,s],E,width)*self.kpt_weights[nk]
                        elif broadening=='lorentzian':
                            # we are going to brute force it for now
                            tempdos=_lorentzian(self.BandStructure[nb,nk,s],E)*self.kpt_weights[nk]


                        else:
                            raise Exception('Unknown broadening scheme')

                        # figure out the pdos factor
                        for ispec in pdos_species:
                            for iorb in pdos_orbitals:

                                pdos_factor = np.sum(self.pdos_filter(ispec,iorb)[:,nb,nk,s])
                                pdos_dos[ispec,iorb,:,s] = pdos_dos[ispec,iorb,:,s] + temp_dos*pdos_factor

            '''
            if recalculate_pdos:
                if broadening == 'gaussian':
                    # Lets change the shape of the bandstructure first
                    new_bs = np.repeat(self.BandStructure[:, :, :, np.newaxis], len(E), axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:, np.newaxis], self.nspins, axis=1)
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:, :, np.newaxis], len(E), axis=2)

                    for nb in range(self.nbands):
                        for ns in range(self.nspins):
                            all_dos[nb, :, ns, :] = _gaussian(new_bs[nb, :, ns, :], E, width) * new_kpt_w[:, ns]

                    for ispec in pdos_species:
                        for iorb in pdos_orbitals:
                            pdos_factor = np.sum(self.pdos_filter(ispec, iorb), axis=0)
                            pdos_factor = np.repeat(pdos_factor[:, :, :, np.newaxis], len(E), axis=3)
                            # Multiply in the factor
                            temp_dos = all_dos * pdos_factor

                            # Sum over the bands, kpoints
                            temp_dos = np.sum(temp_dos, axis=0)
                            temp_dos = np.sum(temp_dos, axis=0)
                            pdos_dos[ispec, iorb, :, :] = np.swapaxes(temp_dos, 0, 1)

                elif broadening == 'adaptive':
                    # Lets change the shape of the bandstructure first
                    new_bs = np.repeat(self.BandStructure[:, :, :, np.newaxis], len(E), axis=3)
                    # Change the shape of the kpoint_weights now, add in spin array
                    new_kpt_w = np.repeat(self.kpt_weights[:, np.newaxis], self.nspins, axis=1)
                    # same again for the E axis now
                    new_kpt_w = np.repeat(new_kpt_w[:, :, np.newaxis], len(E), axis=2)
                    new_weights = np.repeat(self.adaptive_weights[:, :, :, np.newaxis], len(E), axis=3)

                    for nb in range(self.nbands):
                        for ns in range(self.nspins):
                            all_dos[nb, :, ns, :] = _adaptive(new_bs[nb, :, ns, :], E, new_weights[nb, :, ns, :]) * new_kpt_w[:, ns]

                    for ispec in pdos_species:
                        for iorb in pdos_orbitals:
                            pdos_factor = np.sum(self.pdos_filter(ispec, iorb), axis=0)
                            # print(self.atoms[ispec],orbs[iorb],np.max(pdos_factor),np.min(pdos_factor))
                            pdos_factor = np.repeat(pdos_factor[:, :, :, np.newaxis], len(E), axis=3)
                            # Multiply in the factor

                            temp_dos = all_dos * pdos_factor

                            # Sum over the bands, kpoints
                            temp_dos = np.sum(temp_dos, axis=0)
                            temp_dos = np.sum(temp_dos, axis=0)
                            pdos_dos[ispec, iorb, :, :] = np.swapaxes(temp_dos, 0, 1)
                # Done, store for subsequent runs
                self.pdos_dos = pdos_dos

            # Multiply in the FD and spin flips

            pdos_dos = pdos_dos * spin_dir
            pdos_dos = pdos_dos * np.expand_dims(_fermi_dirac(temperature, E), axis=-1)

            if show_total:
                all_dos = np.swapaxes(np.sum(np.sum(all_dos, axis=0), axis=0), 0, 1)
                all_dos = all_dos * spin_dir
                all_dos = all_dos * np.expand_dims(_fermi_dirac(temperature, E), axis=-1)

            # Sum over spins if not spin_polarised
            if not spin_polarised and self.spin_polarised:
                pdos_dos = np.sum(pdos_dos, axis=3)

            # Set up the color stuff

            if pdos_colors is not None:
                try:
                    assert len(pdos_colors) == len(pdos_orbitals) * len(pdos_species)
                    color = pdos_colors
                    custom_cycler = (cycler.cycler(color=color))
                    ax.set_prop_cycle(custom_cycler)

                except:
                    warnings.warn("Warning: pdos_colors does not match number of colors")
                    n_lines = len(pdos_orbitals) * len(pdos_species)
                    color = plt.cm.bwr(np.linspace(0, 1, n_lines))
                    for i in range(len(color)):
                        if np.all(np.round(color[i, 0:3]) == 1):
                            color[i, 0:3] = [0.5, 0.5, 0.5]
                    custom_cycler = (cycler.cycler(color=color))
                    ax.set_prop_cycle(custom_cycler)

            else:
                n_lines = len(pdos_orbitals) * len(pdos_species)
                color = plt.cm.bwr(np.linspace(0, 1, n_lines))

                for i in range(len(color)):
                    if np.all(np.round(color[i, 0:3]) == 1):
                        color[i, 0:3] = [0.5, 0.5, 0.5]
                custom_cycler = (cycler.cycler(color=color))
                ax.set_prop_cycle(custom_cycler)

            # Actually plot now
            if not swap_axes:
                for s in spin_index:
                    if show_total:
                        ax.plot(E, all_dos[:, s], linestyle=linestyle, linewidth=linewidth, color='0.5')

                for ispec in pdos_species:
                    for iorb in pdos_orbitals:

                        color = next(ax._get_lines.prop_cycler)['color']

                        for s in spin_index:
                            ax.plot(E, pdos_dos[ispec, iorb, :, s], linestyle=linestyle, linewidth=linewidth,
                                    label=self.atoms[ispec] + "(" + orbs[iorb] + ')', color=color, zorder=len(pdos_species) - ispec)
                            if shade:
                                ax.fill_between(E, pdos_dos[ispec, iorb, :, s], alpha=alpha, color=color, zorder=len(pdos_species) - ispec)

            else:

                for s in spin_index:
                    if show_total:
                        ax.plot(all_dos[:, s], E, linestyle=linestyle, linewidth=linewidth, color='0.5')

                for ispec in pdos_species:
                    for iorb in pdos_orbitals:

                        color = next(ax._get_lines.prop_cycler)['color']

                        for s in spin_index:
                            ax.plot(pdos_dos[ispec, iorb, :, s], E, linestyle=linestyle, linewidth=linewidth,
                                    label=self.atoms[ispec] + "(" + orbs[iorb] + ')', color=color, zorder=len(pdos_species) - ispec)
                            if shade:
                                ax.fill_betweenx(E, pdos_dos[ispec, iorb, :, s], alpha=alpha, color=color, zorder=len(pdos_species) - ispec)

            # end pdos check

            handles, labels = plt.gca().get_legend_handles_labels()

            ids = np.unique(labels, return_index=True)[1]
            labels = np.array(labels)[np.sort(ids)]

            handles = [handles[i] for i in np.sort(ids)]

            ax.legend(handles, labels, loc=loc, fontsize=fontsize * 0.8, ncol=int(np.ceil((len(pdos_orbitals) * len(pdos_species)) / 4)),
                      fancybox=True, frameon=False, handlelength=1.5, handletextpad=0.2)

        # print("Total time =",time.time()-self.start_time)
        # Autoscaling
        if swap_axes:
            if glim is None and Elim is not None:
                self._autoscale(ax, 'x')
        else:
            if Elim is not None and glim is None:
                self._autoscale(ax, 'y')
        return

    def kpt_where(self, label):
        ''' Find the kpoint indices that correspond to a particular high symmetry label'''
        if label == "G":
            label = "$\\Gamma$"
        lab_loc = np.where(np.array(self.high_sym_labels) == label)[0]
        return self.high_sym[lab_loc]

    def _autoscale(self, ax=None, axis='y', margin=0.1):
        '''Autoscales the x or y axis of a given matplotlib ax object
        to fit the margins set by manually limits of the other axis,
        with margins in fraction of the width of the plot

        Defaults to current axes object if not specified.
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        if ax is None:
            ax = plt.gca()
        newlow, newhigh = np.inf, -np.inf

        for artist in ax.collections + ax.lines:
            x, y = self._get_xy(artist)
            if axis == 'y':
                setlim = ax.set_ylim
                lim = ax.get_xlim()
                fixed, dependent = x, y
            else:
                setlim = ax.set_xlim
                lim = ax.get_ylim()
                fixed, dependent = y, x

            low, high = self._calculate_new_limit(fixed, dependent, lim)
            newlow = low if low < newlow else newlow
            newhigh = high if high > newhigh else newhigh

        margin = margin * (newhigh - newlow)

        setlim(newlow - margin, newhigh + margin)

    def _calculate_new_limit(self, fixed, dependent, limit):
        '''Calculates the min/max of the dependent axis given a fixed axis with limits'''
        if len(fixed) > 2:
            mask = (fixed > limit[0]) & (fixed < limit[1])
            window = dependent[mask]
            low, high = window.min(), window.max()
        else:
            low = dependent[0]
            high = dependent[-1]
            if low == 0.0 and high == 1.0:
                # This is a axhline in the autoscale direction
                low = np.inf
                high = -np.inf
        return low, high

    def _get_xy(self, artist):
        '''Gets the xy coordinates of a given artist'''
        if "Collection" in str(artist):
            x, y = artist.get_offsets().T
        elif "Line" in str(artist):
            x, y = artist.get_xdata(), artist.get_ydata()
        else:
            raise ValueError("This type of object isn't implemented yet")
        return x, y
