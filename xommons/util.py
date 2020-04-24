import numpy as np

def generate_disk(grid_size, disk_radius):
    xx, yy = np.meshgrid(np.arange(grid_size) - (grid_size - 1) / 2, np.arange(grid_size) - (grid_size - 1) / 2)
    img = disk_radius + 0.5 - np.sqrt(xx ** 2 + yy ** 2)
    img = np.clip(img, 0, 1)
    img = np.reshape(img, [1, *img.shape])
    return img