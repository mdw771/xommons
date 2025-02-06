import tifffile
import h5py
import scipy.io
import numpy as np


def complex_image_to_tiff(image, base_filename):
    tifffile.imwrite(base_filename + '_phase.tiff', np.angle(image))
    tifffile.imwrite(base_filename + '_mag.tiff', np.abs(image))


def load_mat(filename):
    f = scipy.io.loadmat(filename)
    return f


def load_h5(filename):
    f = h5py.File(filename, 'r')
    return f


def matlab_complex_to_array(a):
    return a['real'][...] + 1j * a['imag'][...]
