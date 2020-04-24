#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift, ifftshift
from tqdm import trange

from xommons.phys import *
from xommons.constants import *


def gen_mesh(max, shape):
    """Generate mesh grid.
    """
    yy = np.linspace(-max[0], max[0], shape[0])
    xx = np.linspace(-max[1], max[1], shape[1])
    res = np.meshgrid(xx, yy)
    return res


def get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=True):
    """Get Fresnel propagation kernel for TF algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    k = 2 * PI / lmbda_nm
    u_max = 1. / (2. * voxel_nm[0])
    v_max = 1. / (2. * voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], grid_shape[0:2])
    if fresnel_approx:
        H = np.exp(-1j * PI * lmbda_nm * dist_nm * (u**2 + v**2))
    else:
        H = np.exp(-1j * 2 * PI * dist_nm / lmbda_nm * np.sqrt(1 - lmbda_nm ** 2 * (u**2 + v**2) + 0j))

    return H


def get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape):

    """
    Get Fresnel propagation kernel for IR algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    size_nm = np.array(voxel_nm) * np.array(grid_shape)
    k = 2 * PI / lmbda_nm
    ymin, xmin = np.array(size_nm)[:2] / -2.
    dy, dx = voxel_nm[0:2]
    x = np.arange(xmin, xmin + size_nm[1], dx)
    y = np.arange(ymin, ymin + size_nm[0], dy)
    x, y = np.meshgrid(x, y)
    h = np.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * np.exp(1j * k / (2 * dist_nm) * (x ** 2 + y ** 2))
    H = np.fft.fftshift(np.fft.fft2(h)) * voxel_nm[0] * voxel_nm[1]

    return H


def multislice_propagate(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm, free_prop_cm=None, return_intermediate=False):
    """
    Perform multislice propagation on a batch of 3D objects.
    :param grid_delta_batch: 4D array for object delta with shape [n_batches, n_y, n_x, n_z].
    :param grid_beta_batch: 4D array for object beta with shape [n_batches, n_y, n_x, n_z].
    :param probe_real: 2D array for the real part of the probe.
    :param probe_imag: 2D array for the imaginary part of the probe.
    :param energy_ev:
    :param psize_cm: size-3 vector with pixel size ([dy, dx, dz]).
    :param free_prop_cm:
    :return:
    """
    minibatch_size = grid_delta_batch.shape[0]
    grid_shape = grid_delta_batch.shape[1:]
    voxel_nm = np.array(psize_cm) * 1.e7
    wavefront = np.zeros([minibatch_size, grid_shape[0], grid_shape[1]], dtype='complex64')
    wavefront += (probe_real + 1j * probe_imag)

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm

    n_slice = grid_shape[-1]
    delta_nm = voxel_nm[-1]

    # h = get_kernel_ir(delta_nm, lmbda_nm, voxel_nm, grid_shape)
    h = get_kernel(delta_nm, lmbda_nm, voxel_nm, grid_shape)
    k = 2. * PI * delta_nm / lmbda_nm

    if return_intermediate:
        wavefront_ls = []
        wavefront_ls.append(abs(wavefront))

    for i in trange(n_slice):
        delta_slice = grid_delta_batch[:, :, :, i]
        beta_slice = grid_beta_batch[:, :, :, i]
        c = np.exp(1j * k * delta_slice) * np.exp(-k * beta_slice)
        wavefront = wavefront * c
        if i < n_slice - 1:
            wavefront = ifft2(ifftshift(fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
        if return_intermediate:
            wavefront_ls.append(abs(wavefront))

    if free_prop_cm not in [None, 0]:
        if free_prop_cm == 'inf':
            wavefront = fftshift(fft2(wavefront), axes=[1, 2])
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            algorithm = 'TF' if mean_voxel_nm > crit_samp else 'IR'
            if algorithm == 'TF':
                h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = ifft2(ifftshift(fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
            else:
                h = get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = ifft2(ifftshift(fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))

    if return_intermediate:
        if free_prop_cm not in [None, 0]:
            wavefront_ls.append(abs(wavefront))
        return wavefront, wavefront_ls
    else:
        return wavefront


def fresnel_propagate(probe_real, probe_imag, energy_ev, psize_cm, dist_cm):
    """
    Perform multislice propagation on a batch of 3D objects.
    :param grid_delta_batch: 4D array for object delta with shape [n_batches, n_y, n_x, n_z].
    :param grid_beta_batch: 4D array for object beta with shape [n_batches, n_y, n_x, n_z].
    :param probe_real: 2D array for the real part of the probe.
    :param probe_imag: 2D array for the imaginary part of the probe.
    :param energy_ev:
    :param psize_cm: size-3 vector with pixel size ([dy, dx, dz]).
    :param free_prop_cm:
    :return:
    """
    grid_shape = probe_real.shape[1:]
    voxel_nm = np.array(psize_cm) * 1.e7
    wavefront = probe_real + 1j * probe_imag

    lmbda_nm = 1240. / energy_ev
    dist_nm = dist_cm * 1e7

    h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape)

    wavefront = ifft2(ifftshift(fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))

    return wavefront
