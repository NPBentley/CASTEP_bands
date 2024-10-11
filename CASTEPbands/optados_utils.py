"""This module handles visualisation of output from OptaDOS.

Currently, both density of states and partial density of states (DOS) and (PDOS) are currently supported.
This module is designed to be independent of the main CASTEPbands module as far as possible,
although it is anticipated both will be used in tandem, for instance a band structure plot alongside its associated PDOS.

Default energy units are eV (unless specified otherwise).
"""
# Created by : V Ravindran, 12/08/2024

import re
import warnings

import matplotlib as mpl
import numpy as np

import CASTEPbands.Spectral as spec


EV_TO_HARTREE = 1.0/27.211386245988


class DOSdata:
    """The density of states data from OptaDOS.

    Actual density of states data is stored in dos and pdos arrays. The dos array has the shape
    (spins, nengs) while the pdos is stored in array shaped (proj, nengs).

    Attributes
    ----------
    have_pdos : bool
        is a partial density of states calculation
    nspins : int
        number of spin components for density of states
    nproj : int
        number of projectors for partial density of states.
    engs : np.ndarray
        energies for density of states
    dos : np.ndarray
        density of states (states per energy unit [eV])
        shape (nspins, len(engs))
    dos_sum : np.ndarray
        integrated density of states (units: states)
        shape (nspins, len(engs))
    pdos : np.ndarray
        partial density of states (states per energy unit [eV])
        shape (nprojectors, len(engs))
    pdos_contents : dict
        contents of each PDOS projector (by 'species ang.mom and spin')
        taken from the OptaDOS header.
    pdos_labels : list
        list of labels to use for PDOS plot
    eng_unit : str
        unit of energy for data
    efermi : float
        Fermi energy
    zero_fermi : bool
        energy scale set such that Fermi energy is 0.
    optados_shifted : bool
        Did OptaDOS perform the shift already

    Methods
    -------
    plot_data
        Plots the density of states data from OptaDOS.
    set_pdos_labels
        Sets the label for the PDOS plot
    shift_dos_eng
        Shifts the energy scale for density of states and partial density of states data.
    autoscale_data
        Obtain axes limits to scale axes to fit data lyinging within a given energy range.
    """

    def __init__(self,
                 optadosfile: str,
                 efermi: float = None,
                 zero_fermi: bool = True,
                 convert_to_au: bool = False,
                 is_pdos: bool = None,
                 pdos_type: str = None,
                 optados_shifted: bool = True,
                 ):
        """Initialises the density of states data.

        Parameters
        ----------
        optadosfile : str
            OptaDOS data file containing PDOS data
        efermi : float
            Fermi energy to use for DOS
        zero_fermi : bool
            set energy scale such that Fermi energy is zero.
        convert_to_au : bool
            converts data from eV to atomic units
        is_pdos : bool
            Is data PDOS data or DOS.
            This option is left as a fall-back option in case the OptaDOS header is changed.
            Otherwise, it will be inferred from the file if not specified.
        pdos_type : str
            type of projection to use for PDOS. Will be inferred from file if not specified.
        optados_shifted : bool
            Did OptaDOS perform the shift already

        Raises
        ------
        AssertionError
            Density of states data file has more columns than header implies.
        ValueError
            Invalid parameter for optional argument specified.
        IOError
            Data contents could not be determinde from file header alone.
            Requires manual intervention using is_pdos.
        """

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
                    spin_ch = f'({contents[0].split()[3]})'

                if pdos_type == 'sites':
                    pdos_labels[n] = f'{species} {site} {spin_ch}'
                elif pdos_type == 'species':
                    pdos_labels[n] = f'{species} {spin_ch}'
                elif pdos_type == 'species_ang':
                    pdos_labels[n] = f'{species} ({ang_mom}) {spin_ch}'

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

            # Copy everything into the class
            self.nspins = nspins
            self.nproj, self.pdos_type = nproj, pdos_type
            self.proj_contents = proj_contents

            # Now the easy part, the actual density of states
            pdos_data = np.loadtxt(optadosfile, dtype=float, comments='#', encoding='ascii')
            self.engs = pdos_data[:, 0]  # eV
            self.pdos = pdos_data[:, 1:]  # electron per eV
            self.pdos = self.pdos.transpose()

            # If we are not using a custom labels, initialise the pdos labels appropriately.
            if pdos_type != 'custom':
                self.pdos_labels = __pdos_default_labels(nproj, proj_contents, pdos_type)

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
            # NB PDOS/DOS have units of states per eV.
            if self.have_pdos:
                self.pdos /= EV_TO_HARTREE
            else:
                self.dos /= EV_TO_HARTREE

        # OptaDOS does not write the Fermi energy to the output file. We thus need to either
        # get it when initialisng the class or not bother and raise warnings as appropriate.
        self.efermi = efermi
        self.zero_fermi = zero_fermi
        self.optados_shifted = optados_shifted

        if zero_fermi is False and efermi is None:
            warnings.warn('Fermi energy has not been set to zero but remains unspecified')
        elif zero_fermi is True and efermi is None:
            warnings.warn('Zero Fermi scale set - assuming shift done by OptaDOS, CHECK PLOT CAREFULLY!')
            self.efermi = 0
        elif efermi is not None:
            # Shift data if OptaDOS didn't do it for us
            if zero_fermi is True and optados_shifted is False:
                self.shift_dos_eng(-1*efermi)

    def set_pdos_labels(self, pdos_labels: list):
        """Set the labels to use for the PDOS manually.

        Parameters
        ----------
        pdos_labels : list
            labels to use for pdos
        """

        if len(pdos_labels) != self.nproj:
            raise IndexError(f'PDOS has {self.nproj} projectors but only {len(pdos_labels)} provided')
        self.pdos_labels = pdos_labels

    def shift_dos_eng(self, eng_shift: float, eng_unit: str = None):
        """Shifts energy scale for density of states plot.

        If eng_unit is not specified, the energy unit taken from the class. Otherwise the
        appropriate conversion will be applied.

        Parameters
        ----------
        eng_shift : float
            amount to shift energy scale by
        eng_unit : str
            energy unit for eng_shift

        Raises
        ------
        ValueError
            invalid energy unit specified


        """

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

    def autoscale_data(self, Elim: list, do_proj: list = None):
        """Obtains the limits of the PDOS data within a given energy range.

        Parameters
        ----------
        Elim : list
            energy range
        do_proj : list
            list of projectors if partial density of states

        Returns
        -------
        list
            range of PDOS data within Elim

        Raises
        ------
        ValueError
            Did not specify energy limits correctly

        """
        if len(Elim) != 2:
            raise ValueError('Must specify lower and upper bound of energy range')

        # Obtain the dos_data values within a given energy range
        eng_indx = np.where(np.logical_and(Elim[0] <= self.engs, self.engs <= Elim[1]))[0]

        # Decide on what we are plotting and extract data in the given energy range.
        if self.have_pdos is True:
            if do_proj is None:
                do_proj = np.arange(self.nproj, dtype=int)

            dos_data = self.pdos[do_proj]  # Care about a projectory only if we plot it.
            dos_data = dos_data[:, eng_indx]
        else:
            dos_data = self.dos[:, eng_indx]

        # Now get limits of DOS data
        return [np.min(dos_data), np.max(dos_data)]

    def plot_data(self, ax: mpl.axes._axes.Axes,
                  Elim: list = None,
                  dos_lim: list = None,
                  do_proj: list = None,
                  do_axes_labels: bool = True,
                  fontsize: float = 20,
                  linewidth: float = 1.1,
                  linecolor: 'str|list' = None,
                  orient: str = 'horizontal',
                  fermi_linestyle: str = '--',
                  fermi_color: str = None,
                  fermi_linewidth: float = 1.3
                  ):
        """Plots the data from OptaDOS.

        Parameters
        ----------
        ax : mpl.axes._axes.Axes
            axes for density of states
        Elim : list
            energy limit (units based on class data)
        dos_lim : list
            range of PDOS plot
            (units will be either states per eV or states depending on whether integrated or not)
        do_proj : list
            projectors to use for plot (specified by index starting from 0 for 1st projector).
        do_axes_labels : bool
            add axes labels
        fontsize : float
            fontsize for plot
        linewidth : float
            widht of lines for density of states data
        linecolor : 'str|list'
            colours for density of states data. If a PDOS calculation, then colours need to be
            specified for each projector as a list.
        orient : str
            orientation of plot ('vertical'/'v' or 'horizontal'/'h')
            Horizontal sets the energy scale on the horizontal/x-axis.
        fermi_linestyle : str
            line style for Fermi energy
        fermi_color : str
            line colour for Fermi energy
        fermi_linewidth : float
            line width for Fermi energy

        Raises
        ------
        IndexError
            maximum projector in do_proj exceeded number of available projectors
            number of lines specied for linecolor does not match number of projectors.
        ValueError
            Invalid parameter specified.


        """

        orient = orient.lower()
        if orient not in ('horizontal', 'vertical'):
            # Aliases for orient argument
            if orient == 'h':
                orient = 'horizontal'
            elif orient == 'v':
                orient = 'vertical'
            else:
                raise ValueError(f"Invalid orientation {orient}: must be 'h', 'horizontal', 'v', 'vertical'")

        # Decide which projectors to use for PDOS
        if do_proj is None and self.have_pdos is True:
            do_proj = np.arange(self.nproj, dtype=int)

        if do_proj is not None:
            if np.amax(do_proj) >= self.nproj:
                raise IndexError(
                    f'do_proj specified maximum projector index of {np.amax(do_proj)} ' +
                    f'but DOSdata only contains {self.nproj} projectors'
                )

        # Set the default line colour (for DOS mainly)
        # Additionally, set the Fermi energy line colour if not set already.
        if linecolor is None and self.have_pdos is False:
            if self.nspins == 2:
                linecolor = ['red', 'blue']
                if fermi_color is None:
                    fermi_color = '0.5'
            else:
                linecolor = 'black'
                if fermi_color is None:
                    fermi_color = 'red'
        elif linecolor is None and self.have_pdos is True:
            # For PDOS, we just run through the user's default colours in rcParams.
            linecolor = [None for n in range(len(do_proj))]

        # Check user's colour choices
        if linecolor is not None and self.have_pdos is True:
            if self.have_pdos and len(linecolor) != len(do_proj):
                raise IndexError(f'Number of line colours ({len(linecolor)}) ' +
                                 f'does not match number of projectors {len(do_proj)}')

        # If Fermi energy line colour is not set (which it won't be if doing a PDOS), set it now.
        if fermi_color is None:
            fermi_color = '0.5'

        # Set the label for the energy scale and do Fermi energy line
        if self.zero_fermi is True:
            eng_label = r'$E - E_\mathrm{F}$ ' + f'({self.eng_unit})'
        else:
            eng_label = r'$E$ ' + f'({self.eng_unit})'

        if self.efermi is None:
            warnings.warn('Fermi energy not specified - skipping line')
        else:
            if orient == 'vertical':
                ax.axhline(self.efermi, linewidth=fermi_linewidth, linestyle=fermi_linestyle, color=fermi_color)
            else:
                ax.axvline(self.efermi, linewidth=fermi_linewidth, linestyle=fermi_linestyle, color=fermi_color)

        # Finally, decide what we are plotting and plot it
        if self.have_pdos is True:
            data_label = f'PDOS (electrons per {self.eng_unit})'

            # For PDOS, loop around all projectors and plot with labels
            if orient == 'vertical':
                for n, proj in enumerate(do_proj):
                    ax.plot(self.pdos[proj], self.engs, label=self.pdos_labels[proj],
                            color=linecolor[n], linewidth=linewidth)
            else:
                for n, proj in enumerate(do_proj):
                    ax.plot(self.engs, self.pdos[proj], label=self.pdos_labels[proj],
                            color=linecolor[n], linewidth=linewidth)
        else:
            data_label = f'DOS (electrons per {self.eng_unit})'
            if orient == 'vertical':
                for ns in range(self.nspins):
                    ax.plot(self.dos[ns], self.engs,
                            color=linecolor[ns], linewidth=linewidth)
            else:
                for ns in range(self.nspins):
                    ax.plot(self.engs, self.dos[ns],
                            color=linecolor[ns], linewidth=linewidth)

        # Add axes labels
        if do_axes_labels is True:
            if orient == 'vertical':
                ax.set_xlabel(data_label, fontsize=fontsize)
                ax.set_ylabel(eng_label, fontsize=fontsize)
            else:
                ax.set_xlabel(eng_label, fontsize=fontsize)
                ax.set_ylabel(data_label, fontsize=fontsize)

        # Sort out minor ticks, tick labels and font sizes
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

        ax.tick_params(axis='both', which='major', labelsize=fontsize * 0.8, length=8, width=1.2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize * 0.8, length=4, width=1.2)

        # Set energy scale limits
        if Elim is None:
            if self.efermi is not None:
                Elim = [self.efermi - 10, self.efermi + 10]

        # Set density of states limits
        if dos_lim is None:
            dos_lim = self.autoscale_data(Elim, do_proj)
            # Add some head room
            dos_lim[1] += 5

        if orient == 'vertical':
            ax.set_xlim(dos_lim)
        else:
            ax.set_ylim(dos_lim)

        if orient == 'vertical':
            ax.set_ylim(Elim)
        else:
            ax.set_xlim(Elim)


def get_optados_fermi_eng(optados_outfile: str, return_fermi_src: bool = False):
    """Obtain the Fermi energy from OptaDOS output file.

    This also checks the OptaDOS calculation header in the output file to see if the Fermi energy
    was zeroed for us.

    Parameters
    ----------
    optados_outfile : str
        OptaDOS output file
    return_fermi_src : bool
        Return the source of the Fermi energy ('castep' or 'optados')

    Returns
    -------
    efermi : float
        Fermi energy (in eV)
    zero_fermi : bool
        Did OptaDOS zero the Fermi energy?
    """

    efermi, zero_fermi, fermi_src = None, None, ''
    with open(optados_outfile, 'r', encoding='ascii') as file:
        # Loop through the OptaDOS file and get the final Fermi energy printed
        # (in case calculation was restarted with different parameters)
        for line in file:
            if re.search('Fermi energy from DOS', line.strip()):
                efermi = float(line.split()[6])
                fermi_src = 'optados'
            elif re.search('Set fermi energy from file', line.strip()):
                efermi = float(line.split()[7])
                fermi_src = 'castep'

            if re.search('Shift energy scale so fermi_energy=0', line.strip()):
                logical_str = line.split()[7]
                if logical_str == 'True':
                    zero_fermi = True
                elif logical_str == 'False':
                    zero_fermi = False

    if return_fermi_src is True:
        return efermi, zero_fermi, fermi_src
    else:
        return efermi, zero_fermi


def plot_bs_with_dos(castep_seed: str,
                     optados_file: str,
                     ax_bs: mpl.axes._axes.Axes,
                     ax_dos: mpl.axes._axes.Axes,
                     Elim: list = None,
                     dos_lim: list = None,
                     zero_fermi: bool = True,
                     optados_shifted: bool = True,
                     fontsize: float = 20.0,
                     use_fermi: str = 'optados',
                     optados_outfile: bool = None,
                     do_proj: bool = None,
                     tick_direction: str = 'in',
                     ):
    """Plot band structure with associated density of states.

    This requires two separate calculations from CASTEP, a band structure calculation and a density
    of states calculation. The actual density of states will be from OptaDOS data along with the
    Fermi energy. This function ensures all the book-keeping is done correctly discarding the CASTEP
    Fermi energy if requested and ensuring both plots use the same energy scale.

    Parameters
    ----------
    castep_seed : str
        CASTEP seedname. This is also the same as the OptaDOS seedname.
    optados_file : str
        OptaDOS DOS or PDOS output file (.dat)
    ax_bs : mpl.axes._axes.Axes
        axes for band structure plot
    ax_dos : mpl.axes._axes.Axes
        axes for density of states
    Elim : list
        energy limit (in eV)
    dos_lim : list
        range of PDOS plot
        (units will be either states per eV or states depending on whether integrated or not)
    zero_fermi : bool
        set energy scale such that Fermi energy is at zero.
    optados_shifted : bool
        OptaDOS performs the energy shift for Zero Fermi for us.

        Ensures that a double shift is not performed.
        If the OptaDOS output file is present, then cross-checks
        will be performed for sanity.
    fontsize : float
        font size to use for plot
    use_fermi : str
        Fermi energy to use for plot ('castep' or 'optados')
        Default : 'optados'
    optados_outfile : bool
        OptaDOS output file (.odo)
    do_proj : bool
        projectors to use for plot (specified by index starting from 0 for 1st projector).
    tick_direction : str
        direction of ticks for axes ('in', 'out' or 'inout')
        Default : 'inout'

    Returns
    -------

    Raises
    ------
    AssertionError
        Conflicting argument options with OptaDOS calculation parameters.
    ValueError
        Invalid parameter specified for use_fermi.


    """

    use_fermi = use_fermi.lower()
    if use_fermi not in ('castep', 'optados'):
        raise ValueError('use_fermi must be one of "castep" or "optados"')

    if optados_outfile is None:
        optados_outfile = castep_seed + '.odo'

    # If the OptaDOS output file exists, we can then do some sanity checks
    try:
        open(optados_file, 'r', encoding='ascii')
    except FileNotFoundError:
        pass
    else:
        *_, did_shift = get_optados_fermi_eng(optados_outfile)
        if did_shift != optados_shifted:
            raise AssertionError(f'OptaDOS reports zero Fermi of {did_shift} but {optados_shifted=}')

    # This requires some careful book-keeping to make sure everything lines up nicely.
    # First read the CASTEP file and note the raw Fermi energy.
    bs_data = spec.Spectral(castep_seed, zero_fermi=False)
    castep_efermi = bs_data.Ef

    # If we want to use the OptaDOS Fermi energy make sure to read it from the file, and
    # while we are at it make sure to check if OptaDOS actually zeroed the Fermi energy.
    if use_fermi == 'optados':
        optados_efermi, did_shift = get_optados_fermi_eng(optados_outfile)
        if did_shift != optados_shifted:
            raise AssertionError(f'OptaDOS reports zero Fermi of {did_shift} but {optados_shifted=}')
        if optados_efermi is None:
            # Likely specified using CASTEP fermi energy for OptaDOS and so OptaDOS will not write it out
            raise AssertionError('Could not find OptaDOS fermi energy, did you make sure OptaDOS actually calculates it?')

        # Set the CASTEP Fermi energy to the OptaDOS Fermi energy
        castep_efermi = optados_efermi

    # Now shift data accordingly
    if zero_fermi is True:
        bs_data.shift_bands(castep_efermi)
        bs_data.zero_fermi = True

    # Read in the OptaDOS data
    dos_data = DOSdata(optados_file, efermi=castep_efermi, zero_fermi=zero_fermi, optados_shifted=optados_shifted)

    # Plot the band structure
    bs_data.plot_bs(ax_bs, Elim=Elim, fontsize=fontsize)

    # Plot the density of states
    dos_data.plot_data(ax_dos, Elim=Elim, dos_lim=dos_lim,
                       fontsize=fontsize, orient='vertical', do_proj=do_proj)

    # Set energy scale
    if Elim is None:
        if zero_fermi is True:
            castep_efermi = 0
        Elim = [castep_efermi - 10, castep_efermi + 10]

    # Remove the y-axis label as they are the same
    ax_dos.set_ylabel('')

    # Adjust the tick labels for the plot
    ax_bs.tick_params(axis='both', which='major', direction=tick_direction, labelsize=fontsize * 0.8, length=8, width=1.2,
                      top=True, left=True, right=True, bottom=True)
    ax_bs.tick_params(axis='y', which='minor', direction=tick_direction, labelsize=fontsize * 0.8, length=4, width=1.2,
                      top=True, left=True, right=True, bottom=True)
    ax_dos.tick_params(axis='both', which='major', direction=tick_direction, labelsize=fontsize * 0.8, length=8, width=1.2,
                       top=True, left=True, right=True, bottom=True)
    ax_dos.tick_params(axis='both', which='minor', direction=tick_direction, labelsize=fontsize * 0.8, length=4, width=1.2,
                       top=True, left=True, right=True, bottom=True)
