from CASTEPbands.optados_utils import plot_bs_with_dos
from CASTEPbands.Spectral import Spectral

import matplotlib.pyplot as plt
import numpy as np

CASTEP_SEED = 'Si'
PDOS_FILE = 'Si.pdos.dat'

fig, axs = plt.subplots(1, 2, figsize=(10, 8), width_ratios=[3, 1.5], sharey=True)
ax_bs, ax_dos = axs
plot_bs_with_dos(CASTEP_SEED, PDOS_FILE,
                 ax_bs, ax_dos,
                 Elim=[-15, 15],
                 fontsize=16,
                 optados_shifted=False,
                 use_fermi='castep')

ax_dos.legend(loc='upper right')

# Note the default autoscaler within CASTEPbands will work well for quick plots
# but you can of course override this behaviour if you wish.
ax_dos.set_xlim(-2, 2)

plt.show()
