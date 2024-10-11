from CASTEPbands.optados_utils import plot_bs_with_dos
from CASTEPbands.Spectral import Spectral

import matplotlib.pyplot as plt
import numpy as np

castep_seed = 'BaTiO3'
pdos_file = castep_seed+'.pdos.dat'
use_fermi = 'castep'

fig, axs = plt.subplots(1, 2, figsize=(10, 8), width_ratios=[3, 1.5], sharey=True)
ax_bs, ax_dos = axs
fontsize = 16
plot_bs_with_dos(castep_seed, pdos_file,
                 ax_bs, ax_dos,
                 Elim=[-8, 8], dos_lim=[0, 20],
                 fontsize=fontsize,
                 optados_shifted=False, use_fermi=use_fermi)
ax_dos.legend(loc='upper right')
plt.savefig(castep_seed+'bs_pdos.png')
plt.show()
