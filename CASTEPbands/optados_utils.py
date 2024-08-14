"""This module handles visualisation of output from OptaDOS

Currently, both density of states and partial density of states (DOS) and (PDOS) are currently supported.
This module is designed to be independent of the main CASTEPbands module as far as possible,
although it is anticipated both will be used in tandem, for instance a band structure plot alongside its associated PDOS.
"""
# Created by : V Ravindran, 12/08/2024

import re
import warnings

import matplotlib as mpl
import numpy as np

import CASTEPbands.Spectral as spec


EV_TO_HARTREE = 1.0/27.211386245988


class DOSdata:
    # TODO Documenation
    def __init__(self,
                 optadosfile: str,
                 efermi: float = None,
                 zero_fermi: bool = True,
                 convert_to_au: bool = False,
                 is_pdos: bool = None,
                 pdos_type: str = None,
                 optados_shifted: bool = True,
                 ):
        # TODO Documentation
        # First, decide whether we have a PDOS or DOS if not specified
        # by reading the contents of the header.
        dos_header = ' #    Density of States'
        pdos_header = '#|                    Partial Density of States'
        pdos_allow_labels = ('species', 'species_ang', 'sites')
        pdos_valid_types = ('species', 'species_ang', 'sites', 'custom')

        self.have_pdos = None
        if is_pdos is None:
            with open(optadosfile, 'r', encoding='ascii') as file:
                for line in file:
                    if line.startswith(dos_header):
                        self.have_pdos = False
                        break
                    if line.startswith(pdos_header):
                        self.have_pdos = True
                        break
            if self.have_pdos is None:
                raise IOError('Could not determine data contents from file header alone, please specify using is_pdos')
        else:
            self.have_pdos = is_pdos

        # Initialise data
        self.engs, self.dos, self.dos_sum = None, None, None
        self.nproj, self.pdos_type, self.proj_contents = None, None, None
        self.pdos, self.pdos_labels = None, None
        self.nspins = None

        """
        Functions to read the OptaDOS data
        """
        def __determine_pdos_decompose(proj_contents: dict):
            """Determines how the PDOS was decomposed based on contents of the header.

            There is nothing particularly clever being done here, we simply count the number of
            unique species, sites or angular momenta per projector and deduce accordingly what the
            projection was.
            Note this may break for certain custom projections done in OptaDOS.
            """
            def n_uniq(x: list):
                """Get the number of unique elements in a list."""
                return len(set(x))

            # Here follows the most dull book-keeping exercise...
            for contents in proj_contents.values():
                species_list, site_list, ang_mom_list = [], [], []
                # print(f'contents for projector {n}: {contents}')
                for atom in contents:
                    # We don't care about spin channels for this
                    # so only extract species, site and angular momentum.
                    species, site, ang_mom = atom.split()[0:3]
                    species_list.append(species)
                    site_list.append(site)
                    ang_mom_list.append(ang_mom)

                # Count number of unique species, sites and angular momentum for projector
                if n_uniq(species_list) == 1 and n_uniq(site_list) == 1 and n_uniq(ang_mom_list) != 1:
                    # We have a single site/species consisting of multiple angular momentum channels
                    # so we are projecting by site.
                    return 'sites'
                if n_uniq(species_list) == 1 and n_uniq(site_list) != 1 and n_uniq(ang_mom_list) == 1:
                    # We have a single species at various sites consisting of multiple angular momentum channels
                    # so we are projecting by angular momentum channel.
                    return 'species_ang'
                if n_uniq(species_list) == 1 and n_uniq(site_list) != 1 and n_uniq(ang_mom_list) != 1:
                    return 'species'

            # If we reached this point, we found nothing so raise a warning message and set type to custom
            warnings.warn(
                'Could not identify type of PDOS, either reinitialise class with type specified or '
                + ' set projectors manually using DOSdata.set_pdos_labels'
            )
            return 'custom'

        def __pdos_default_labels(nproj: int, proj_contents: str, pdos_type: str):
            """Sets some default labels for the PDOS based on the type of decomposition.

            Note that only species, sites and species_ang are supported types."""

            if pdos_type not in pdos_allow_labels:
                # This should never be raised. EVER!
                raise AssertionError('Custom PDOS cannot have their labels set automatically')

            pdos_labels = [None for n in range(nproj)]
            for n, contents in proj_contents.items():
                # If the book-keeping has been done correctly up to this point,
                # everything we wish to label for a given decomposition should be contained within the
                # first item of the contents for a given projector.
                species, site, ang_mom = contents[0].split()[0:3]
                spin_ch = ''
                if self.nspins == 2:
                    spin_ch = contents[0].split()[3]

                if pdos_type == 'sites':
                    pdos_labels[n] = f'{species} {site} ({spin_ch})'
                elif pdos_type == 'species':
                    pdos_labels[n] = f'{species} ({spin_ch})'
                elif pdos_type == 'species_ang':
                    pdos_labels[n] = f'{species} ({ang_mom}) ({spin_ch})'

            return pdos_labels

        def __pdos_read(pdos_type: str):
            # With PDOS, the header size will change so let's be careful about this
            init_header = 9
            dash_line = '#+----------------------------------------------------------------------------+'

            # Extract and store the relevant portion of the header without the dash lines.
            header, col_indx = [], []
            nspins, nproj, i = 1, 0, 0
            with open(optadosfile, 'r', encoding='ascii') as file:
                for lineno, line in enumerate(file):
                    if lineno < init_header:
                        continue
                    if line.strip() == dash_line:
                        continue
                    if line.strip().startswith('#') is False:
                        break
                    if 'Spin Channel' in line.strip():
                        nspins = 2
                    if 'Column:' in line and 'contains:' in line:
                        # We might as well sneak counting how many projectors we have in here.
                        nproj += 1
                        col_indx.append(i)

                    header.append(line.strip())
                    i += 1

            # Store the raw labels from OptaDOS without any editing
            proj_contents = dict.fromkeys(np.arange(nproj, dtype=int))
            for n in range(nproj):
                if n == nproj-1:
                    col_contents = header[col_indx[n]:]
                else:
                    col_contents = header[col_indx[n]:col_indx[n+1]]

                # Remove the atom and angular momentum labels
                col_contents = col_contents[2:]
                cur_proj = []
                for atom in col_contents:
                    if nspins == 2:
                        species, site, ang_mom, spin_ch = atom.split()[1:5]
                        cur_proj.append(f'{species} {site} {ang_mom} {spin_ch}')
                        # print(f'Projector {n} contains {species} {site} {ang_mom} {spin_ch} {cur_proj}')
                    else:
                        species, site, ang_mom = atom.split()[1:4]
                        cur_proj.append(f'{species} {site} {ang_mom}')
                        # print(f'Projector {n} contains {species} {site} {ang_mom} {cur_proj}')
                proj_contents[n] = cur_proj

            if pdos_type is None:
                # Attempt to determine the PDOS composition if not specified
                pdos_type = __determine_pdos_decompose(proj_contents)

            if pdos_type not in pdos_valid_types:
                raise ValueError(f'Invalid value of pdos_type "{pdos_type}"')

            # If we are not using a custom labels, initialise the pdos labels appropriately.
            if pdos_type != 'custom':
                self.pdos_labels = __pdos_default_labels(nproj, proj_contents, pdos_type)

            # Copy everything into the class
            self.nspins = nspins
            self.nproj, self.pdos_type = nproj, pdos_type
            self.proj_contents = proj_contents

            # Now the easy part, the actual density of states
            pdos_data = np.loadtxt(optadosfile, dtype=float, comments='#', encoding='ascii')
            self.engs = pdos_data[:, 0]  # eV
            self.pdos = pdos_data[:, 1:]  # electron per eV
            self.pdos = self.pdos.transpose()

        def __dos_read():
            # Read the header and check for spin polarisation
            nspins = 1  # assume no spin until told otherwise
            with open(optadosfile, 'r', encoding='ascii') as file:
                for line in file:
                    if line.strip().startswith('#') is False:
                        break
                    if 'spin' in line.strip():
                        nspins = 2
                        break

            # Read data and then perform sanity checks that the format is what we expect
            dosdata = np.loadtxt(optadosfile, dtype=float, comments='#', encoding='ascii')
            if nspins == 1:
                if dosdata.shape[1] != 3:
                    raise AssertionError(
                        f'Expected 3 columns for non spin-polarised calculation but have {dosdata.ndim} instead.'
                    )
            if nspins == 2:
                if dosdata.shape[1] != 5:
                    raise AssertionError(
                        f'Expected 5 columns for spin-polarised calculation but have {dosdata.ndim} instead.'
                    )

            # Store everything in the class
            nengs = dosdata.shape[0]
            self.engs = np.empty(nengs, dtype=float)
            self.dos = np.empty((nspins, nengs), dtype=float)
            self.dos_sum = np.empty((nspins, nengs), dtype=float)

            self.nspins = nspins
            self.engs = dosdata[:, 0]  # eV
            if nspins == 2:
                self.dos[0] = dosdata[:, 1]  # electron per eV
                self.dos[1] = dosdata[:, 2]  # electron per eV
                self.dos_sum[0] = dosdata[:, 3]  # electrons (integrated DOS)
                self.dos_sum[1] = dosdata[:, 4]  # electrons (integrated DOS)
            else:
                self.dos[0] = dosdata[:, 1]  # electron per eV
                self.dos_sum[0] = dosdata[:, 2]  # electrons (integrated DOS)

        # Now read the data accordingly
        if self.have_pdos is True:
            __pdos_read(pdos_type)
        else:
            __dos_read()

        self.eng_unit = 'eV'

        # Convert to atomic units if desired
        if convert_to_au:
            self.eng_unit = 'Hartrees'
            self.engs *= EV_TO_HARTREE
            if self.have_pdos:
                self.pdos *= EV_TO_HARTREE
            else:
                self.dos *= EV_TO_HARTREE

        # OptaDOS does not write the Fermi energy to the output file. We thus need to either
        # get it when initialisng the class or not bother and raise warnings as appropriate.
        self.efermi = None
        self.zero_fermi = zero_fermi
        self.optados_shifted = optados_shifted

        if zero_fermi is False and efermi is None:
            warnings.warn('Fermi energy has not been set to zero but remains unspecified')
        elif zero_fermi is True and efermi is None:
            warnings.warn('Zero Fermi scale set - assuming shift done by OptaDOS, CHECK PLOT CAREFULLY!')
            self.efermi = 0
        elif zero_fermi is True and efermi is not None:
            self.efermi = efermi
            # Shift data if OptaDOS didn't do it for us
            if optados_shifted is False:
                self.shift_dos_eng(-1*efermi)

    def set_pdos_labels(self, pdos_labels: list):
        if len(pdos_labels) != self.nproj:
            raise IndexError(f'PDOS has {self.nproj} projectors but only {len(pdos_labels)} provided')
        self.pdos_labels = pdos_labels

    def shift_dos_eng(self, eng_shift: float, eng_unit: str = None):
        if eng_unit is None:
            eng_unit = self.eng_unit.lower()
        if eng_unit not in ('ev', 'hartrees'):
            raise ValueError('eng_unit must be "ev" or "hartrees"')

        # Apply appropriate conversion
        if eng_unit == 'hartrees' and self.eng_unit.lower() == 'ev':
            eng_shift /= EV_TO_HARTREE
        elif eng_unit == 'ev' and self.eng_unit.lower() == 'hartrees':
            eng_shift *= EV_TO_HARTREE

        self.engs += eng_shift
        if self.efermi is None:
            warnings.warn('Unable to shift Fermi energy as it is not specified')
        else:
            self.efermi += eng_shift

    def plot_data(self, ax: mpl.axes._axes.Axes,
                  fontsize: float = 20,
                  linewidth: float = 1.1,
                  fermi_linestyle: str = '--',
                  fermi_color: str = '0.5',
                  fermi_linewidth: float = 1.3
                  ):

        # Set the label for the energy scale and do Fermi energy
        if self.zero_fermi is True:
            ax.set_xlabel(r'$E - E_\mathrm{F}$ ' + f'({self.eng_unit})', fontsize=fontsize)
            ax.axvline(self.efermi, linewidth=fermi_linewidth, linestyle=fermi_linestyle, color=fermi_color)
        else:
            ax.set_xlabel(r'$E$ ' + f'({self.eng_unit})', fontsize=fontsize)
            if self.efermi is None:
                warnings.warn('Fermi energy not specified - skipping line')
                ax.axvline(self.efermi, linewidth=fermi_linewidth, linestyle=fermi_linestyle, color=fermi_color)

        # Decide what we are plotting and plot it
        if self.have_pdos is True:
            ax.set_ylabel(f'PDOS (electrons per {self.eng_unit})', fontsize=fontsize)
            # For PDOS, loop around all projectors and plot with labels
            for n in range(self.nproj):
                ax.plot(self.engs, self.pdos[n], linewidth=linewidth, label=self.pdos_labels[n])
        else:
            ax.set_ylabel(f'DOS (electrons per {self.eng_unit})', fontsize=fontsize)
            ax.plot(self.engs, self.dos, linewidth=linewidth)

        # Sort out minor ticks, tick labels and font sizes
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

        ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.8, length=8, width=1.2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize * 0.8, length=4, width=1.2)

        if self.efermi is not None:
            ax.axvline(self.efermi, linewidth=fermi_linewidth, linestyle=fermi_linestyle, color=fermi_color)


def get_optados_fermi_eng(optados_outfile: str):
    efermi, zero_fermi = None, None
    with open(optados_outfile, 'r', encoding='ascii') as file:
        # Loop through the OptaDOS file and get the final Fermi energy printed
        # (in case calculation was restarted with different parameters)
        for line in file:
            if re.search('Fermi energy from DOS', line.strip()):
                efermi = line.split()[6]
            if re.search('Shift energy scale so fermi_energy=0', line.strip()):
                logical_str = line.split()[7]
                if logical_str == 'True':
                    zero_fermi = True
                elif logical_str == 'False':
                    zero_fermi = False

    return efermi, zero_fermi


