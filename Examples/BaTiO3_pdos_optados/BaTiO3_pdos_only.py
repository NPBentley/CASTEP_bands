from CASTEPbands.optados_utils import DOSdata, get_optados_fermi_eng

import matplotlib.pyplot as plt
import numpy as np

castep_seed = 'BaTiO3'
optados_outfile = castep_seed+'.odo'
pdos_file = castep_seed+'.pdos.dat'
fontsize = 16

fig, ax = plt.subplots()

# Get the Fermi energy from the OptaDOS output (.odo). This is a separate file
efermi, optados_shifted = get_optados_fermi_eng(optados_outfile
                                                )
# Initialise the DOSdata class containing the OptaDOS output data
dos_data = DOSdata(pdos_file, efermi, zero_fermi=True,
                   # Did OptaDOS do the shift for us
                   optados_shifted=optados_shifted  # False since from CASTEP
                   )

# Plot the density of states data
dos_data.plot_data(
    ax, Elim=[-10, 10], dos_lim=[0, 20], fontsize=fontsize
)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(castep_seed+'pdos.png')
plt.show()
