import numpy as np
import matplotlib.pyplot as plt

from CASTEPbands.Spectral import Spectral
from CASTEPbands.plotutils import plot_bands, color_by_occ, add_vb_cb
from matplotlib import rcParams

# Optional : Set matplotlib rcParams for finer figure control
rcParams.update({
    # Set to true can give nicer output but take longer to generate
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16
})

fontsize = rcParams['font.size']

castep_seed = 'Si'

# Initialise bands data
spec = Spectral(castep_seed,
                # Set the Fermi energy to be valence band maximum
                # Useful for insulators if calculation run with fix_occupancy=True...
                use_vbm_fermi=False,
                # Energy scale options
                zero_fermi=False,  # Fermi at 0
                # Can also try one of the following
                # zero_cbm=False,  # CBM at 0
                # zero_vbm=False,  # VBM at 0
                )

# Initialise figure
fig, ax = plt.subplots(figsize=(8,6))

# Prepare an empty axes for the band structure
spec.plot_bs(ax, axes_only=True,
             # Control appearance of Fermi energy
             fermi_line_style='--', fermi_line_color='0.5', fermi_linewidth=1
             )

# Plot bands by occupancy
color_by_occ(spec, ax,
             do_bands='both',  # try also "occ", "unocc"
             colors=['b', 'r'],  # occupied, unoccupied
             # linewidth=1.2,
             # linestyle='-'
             )

# Add valence and conduction bands as separate colours
add_vb_cb(spec, ax,
          colors=['orange', 'purple'],
          labels=['VBM', 'CBM']
          )

# Highlight a specific band
plot_bands(spec, ax, band_ids=[6, 9], color=['limegreen', 'cyan'], labels=['my band 1', 'my band 2'])

ax.legend(bbox_to_anchor=(1.05, 0.9))
plt.tight_layout()

plt.savefig('Si_plot_utils.png')
plt.show()
