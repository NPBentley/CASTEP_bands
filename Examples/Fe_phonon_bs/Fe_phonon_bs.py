from CASTEPbands.phonon import Phonon

import numpy as np
import matplotlib.pyplot as plt

# NB-.cell file is not necessary for phonon plots
phon = Phonon('Fe.phonon', verbose=True)

fig, ax = plt.subplots()
phon.plot_dispersion(ax, freq_lim=[0, 320])

plt.title('Fe (BCC): PBE DFPT', fontsize=18)
plt.savefig('Fe_phonon_bs.png')
# plt.show()
