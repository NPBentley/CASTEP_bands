import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from CASTEPbands import Spectral

# Set up matplotlib mirroring the style used by CASTEP's
# postprocessing utility for band visualisation.
rcParams.update({'text.usetex': False,
                 'font.family': 'serif',
                 'ytick.direction': 'in',
                 'ytick.left': True,
                 'ytick.right': True,
                 'ytick.minor.visible': True,
                 'ytick.major.size': 7,
                 'ytick.minor.size': 5
                 })

fontsize = 18
fermi_linewidth = 1.5

# Initialise band data
NaCl_bands = Spectral.Spectral('NaCl', zero_fermi=True)

# Get the band index of the valence and conduction band
# noting that we have double occupancy.
vb_i = int(NaCl_bands.electrons/2) - 1
cb_i = int(NaCl_bands.electrons/2)

fig, ax = plt.subplots()
plt.title('NaCl\nPBESol', fontsize=16)

# Plot the bandstructure first in a single colour
# and mark the gap on the band gap.
NaCl_bands.plot_bs(ax,
                   fontsize=fontsize,
                   # Use a single colour for bands
                   fermi_linewidth=fermi_linewidth,
                   mono=True,
                   mono_color='k',
                   # Set energy limits
                   Elim=(-5, 15),
                   # Mark the gap
                   mark_gap=True,
                   mark_gap_color='red',
                   )

# Label the valence band and conduction band
NaCl_bands.plot_bs(
    ax,
    fontsize=fontsize,
    show_fermi=False,
    band_ids=np.array([vb_i, cb_i]),
    band_labels=np.array(['VB', 'CB']),
    band_colors=np.array(['b', 'r']),
    linewidth=2.0
)


plt.tight_layout()
plt.show()
