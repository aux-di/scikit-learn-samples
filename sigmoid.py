# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.pyt0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)

phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.show()
