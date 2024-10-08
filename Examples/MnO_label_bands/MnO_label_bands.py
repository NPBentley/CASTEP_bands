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
MnO_bands = Spectral.Spectral('MnO', zero_fermi=True,
                              # Ensure that we use high-symmetry labels
                              # commensurate with crystallographic space group
                              # rather than just cell parameters (as done in CASTEP)
                              high_sym_spacegroup=True  # Default
                              )

# Get the band index of the valence and conduction band for each spin channel.
nelec = np.array([MnO_bands.nup, MnO_bands.ndown], dtype=int)
vb_i = nelec - 1
cb_i = nelec

fig, ax = plt.subplots()
plt.title('MnO\nr2SCAN', fontsize=16)

# Plot a spin polarised band structure
MnO_bands.plot_bs(ax,
                  # Colour all bands based on spin channel
                  spin_polarised=True,
                  fermi_linewidth=fermi_linewidth,
                  # Set energy limits
                  Elim=(-10, 15),
                  mark_gap=True,
                  mark_gap_color='green'
                  )

# Label the valence band for 'up' spin and valence band for 'down' spin
MnO_bands.plot_bs(
    ax,
    fontsize=fontsize,
    show_fermi=False,
    band_ids=np.array([[vb_i[0], None], [None, cb_i[1]]]),
    band_labels=['VB (up)', 'CB (down)'],
    band_colors=['r', 'orange'],
    linewidth=3.0,
)

# Add the legend
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('MnO_label_bands.png')
