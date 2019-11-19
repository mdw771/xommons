import numpy as np
import xommons
import matplotlib.pyplot as plt

grid_size = 512
material = 'Au'
density = 19.32
energy_kev = 5
psize_cm = 1e-7
disk_radius = 64
thickness_cm = 100e-7
dist_cm = 1000e-7

# delta and beta follows n = 1 - delta - i*beta convention.
delta = xommons.ri_delta('Au', energy_kev, density)
beta = xommons.ri_beta('Au', energy_kev, density)

xx, yy = np.meshgrid(np.arange(grid_size) - (grid_size - 1) / 2, np.arange(grid_size) - (grid_size - 1) / 2)
img = disk_radius + 0.5 - np.sqrt(xx ** 2 + yy ** 2)
img = np.clip(img, 0, 1)
img = np.reshape(img, [1, *img.shape])

delta_grid = img * delta
beta_grid = img * beta

probe_real = np.ones([1].append([grid_size] * 2))
probe_imag = np.zeros([1].append([grid_size] * 2))
wavefield = probe_real + 1j * probe_imag

wavefield = xommons.modulate_wavefield(wavefield, delta_grid, beta_grid, energy_kev * 1e3, thickness_cm)
wavefield = xommons.fresnel_propagate(wavefield, energy_kev * 1e3, [psize_cm] * 2, dist_cm, fresnel_approx=True)

plt.imshow(abs(wavefield)[0])
plt.show()
