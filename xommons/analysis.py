import matplotlib
import matplotlib.pyplot as plt
try:
    from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftn, ifftn
    from pyfftw.interfaces.numpy_fft import fftshift as np_fftshift
    from pyfftw.interfaces.numpy_fft import ifftshift as np_ifftshift
except:
    from numpy.fft import fft2, ifft2, fftn, ifftn
    from numpy.fft import fftshift as np_fftshift
    from numpy.fft import ifftshift as np_ifftshift
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
import numpy as np
import dxchange
try:
    import tensorflow as tf
except:
    print('Cannot import tensorflow.')

import os, glob
import warnings
import sys


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

def generate_disk(shape, radius, anti_aliasing=5):
    shape = np.array(shape)
    radius = int(radius)
    x = np.linspace(-radius, radius, (radius * 2 + 1) * anti_aliasing)
    y = np.linspace(-radius, radius, (radius * 2 + 1) * anti_aliasing)
    xx, yy = np.meshgrid(x, y)
    a = (xx**2 + yy**2 <= radius**2).astype('float')
    res = np.zeros(shape * anti_aliasing)
    center_res = (np.array(res.shape) / 2).astype('int')
    res[center_res[0] - int(a.shape[0] / 2):center_res[0] + int(a.shape[0] / 2),
        center_res[1] - int(a.shape[0] / 2):center_res[1] + int(a.shape[0] / 2)] = a
    res = gaussian_filter(res, 0.5 * anti_aliasing)
    res = res[::anti_aliasing, ::anti_aliasing]
    return res


def generate_ring(shape, radius, anti_aliasing=5):

    disk1 = generate_disk(shape, radius + 0.5, anti_aliasing=anti_aliasing)
    disk2 = generate_disk(shape, radius - 0.5, anti_aliasing=anti_aliasing)
    return disk1 - disk2


def fourier_ring_correlation(obj, ref, step_size=1, save_path=None, save_mask=True, save_fname='fsc', threshold_curve=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    radius_max = int(min(obj.shape) / 2)
    f_obj = np_fftshift(fft2(obj))
    f_ref = np_fftshift(fft2(ref))
    f_prod = f_obj * np.conjugate(f_ref)
    f_obj_2 = np.real(f_obj * np.conjugate(f_obj))
    f_ref_2 = np.real(f_ref * np.conjugate(f_ref))
    radius_ls = np.arange(1, radius_max, step_size)
    fsc_ls = []
    if save_path is not None:
        np.save(os.path.join(save_path, 'radii.npy'), radius_ls)

    for rad in radius_ls:
        print(rad)
        if os.path.exists(os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad)))):
            mask = dxchange.read_tiff(os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad))))
        else:
            mask = generate_ring(obj.shape, rad, anti_aliasing=2)
            if save_mask:
                dxchange.write_tiff(mask, os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad))),
                                    dtype='float32', overwrite=True)
        fsc = abs(np.sum(f_prod * mask))
        fsc /= np.sqrt(np.sum(f_obj_2 * mask) * np.sum(f_ref_2 * mask))
        fsc_ls.append(fsc)
        if save_path is not None:
            np.save(os.path.join(save_path, '{}.npy'.format(save_fname)), fsc_ls)
    return np.array(fsc_ls)


def plot_frc(frc, radius_max='auto', step_size=1, output_fname='frc.pdf', threshold_curve=True):

    if radius_max == 'auto':
        radius_max = len(frc) + 1
    radius_ls = np.arange(1, radius_max, step_size)

    plt.plot(radius_ls.astype(float) / radius_ls[-1], frc, label=os.path.basename(output_fname)[:-4])
    plt.xlabel('Spatial frequency (1 / Nyquist)')
    plt.ylabel('FRC')

    if threshold_curve:
        n_ls = 2 * np.pi * radius_ls
        t = 0.2071 + 1.9102 / np.sqrt(n_ls) / (1.2071 + 0.9102 / np.sqrt(n_ls))
        plt.plot(radius_ls.astype(float) / radius_ls[-1], t, label='1/2-bit threshold')
    plt.legend()
    plt.savefig(output_fname, format='pdf')


def gaussian_fit_2d(img, n_iter=1000, verbose=False):

    # https://scipy-cookbook.readthedocs.io/items/FittingData.html

    sess = tf.Session()

    shape = np.array(img.shape)
    y = np.arange(shape[0]) - (shape[0] - 1.) / 2.
    x = np.arange(shape[1]) - (shape[1] - 1.) / 2.
    pts = [(iy, ix) for iy in y for ix in x]
    pts = np.array(pts)

    background = np.mean(img[:, 0])
    input_ac = img - background
    input_ac[input_ac < 0] = 0
    sigma_y = np.sqrt(np.sum(input_ac.flatten() * (pts[:, 0] - 0) ** 2) / (np.sum(input_ac)))
    sigma_x = np.sqrt(np.sum(input_ac.flatten() * (pts[:, 1] - 0) ** 2) / (np.sum(input_ac)))
    sigma_0 = np.mean([sigma_x, sigma_y])
    dat = tf.constant(img, dtype='float64')

    xx, yy = tf.meshgrid(x, y)
    sigma = tf.Variable(sigma_0, dtype='float64')
    b = tf.Variable(background, dtype='float64')
    a = tf.Variable(input_ac.max(), dtype='float64')

    predicted = a * tf.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) + b
    loss = tf.reduce_mean(tf.squared_difference(predicted, dat))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    fit = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())
    for i_iter in range(n_iter):
        _, this_loss, this_a, this_b, this_sigma = sess.run([fit, loss, a, b, sigma])
        if verbose:
            print('{}: {}, a = {}, b = {}, sigma = {}'.format(i_iter, this_loss, this_a, this_b, this_sigma))

    a_1, b_1, sigma_1, pred = sess.run([a, b, sigma, predicted])
    sess.close()

    return a_1, b_1, sigma_1, pred


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def find_frc_crossing(frc, radius_max='auto', step_size=1):

    if radius_max == 'auto':
        radius_max = len(frc) + 1
    radius_ls = np.arange(1, radius_max, step_size)
    radius = radius_ls.astype('float') / radius_ls[-1]
    r = 2 * np.pi * radius_ls
    t = 0.2071 + 1.9102 / np.sqrt(r) / (1.2071 + 0.9102 / np.sqrt(r))

    # Find all crossings
    cross = []
    for i in range(len(radius) - 2):
        if max(t[i:i + 2]) > min(frc[i:i + 2]) and min(t[i:i + 2]) < max(frc[i:i + 2]):
            cross.append(radius[i])
    if len(cross) == 0:
        cross.append(1)

    # Remove outliers
    while len(cross) > 1:
        if cross[-1] - cross[0] > 0.5:
            cross.pop()
        else:
            break

    # Take mean of crossings
    crossing = np.mean(cross)
    err = cross[-1] - cross[0]
    return crossing, err