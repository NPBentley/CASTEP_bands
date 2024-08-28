from CASTEPbands.Spectral import Spectral

import numpy as np
import matplotlib.pyplot as plt

spec = Spectral('MnO', zero_fermi=False)

fig, ax = plt.subplots()

spec.plot_bs(ax)

plt.show()
