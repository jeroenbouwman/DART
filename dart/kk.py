#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:42:14 2021

@author: bouwman
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as ft
import os
import astropy as ap
from astropy.io import ascii
import scipy.interpolate as intp
from scipy import integrate

# from scipy import arange, array, exp


def interp_imag_dielectric_func(
    wavelength: float, e_imag: float, wavelength_new: float) -> float:
    """
    Interpolate imagenary part of the dielectric function.

    Parameters
    ----------
    wavelength : float
        DESCRIPTION.
    e_imag : float
        DESCRIPTION.
    wavelength_new : float
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    Notes
    -----
    Using extrapolation formula of the Lorenz model.
    See Bohren & Huffman section 9.1.2
    """

    def extrap1d(interpolator):
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0]:
                return ys[0] * (x / xs[0])
            elif x > xs[-1]:
                return ys[-1] * (xs[-1] / x ** 3)
            else:
                return interpolator(x)

        def ufunclike(xs):
            return np.array(list(map(pointwise, np.array(xs))))

        return ufunclike

    f_interp = intp.interp1d(wavelength, e_imag)
    f_extrap = extrap1d(f_interp)
    e_imag_interp = f_extrap(wavelength_new)
    return e_imag_interp


# path = "/home/bouwman/MoDust/data/dbase2/oxides/C"
# file_name = "sio2_02.60_0000_042.lnk"

# path = "/home/bouwman/MoDust/data/dbase2/oxides/C"
# file_name = "sio2_02.60_0000_041.lnk"

path = "/home/bouwman/MoDust/data/dbase2/oxides/A"
file_name = "sio2_02.20_0300_001.lnk"

# path = "/home/bouwman/FIT/Qvalues/Optconst"
# file_name = "quartz_Epara_300Knew.txt"

#path = "/home/bouwman/FIT/Qvalues/Optconst"
#file_name = "quartz_Eperp_300Knew.txt"

file = ap.io.ascii.read(os.path.join(path, file_name))

ref_re = file["col2"].data
ref_im = file["col3"].data
wavelength = file["col1"].data


chir = ref_re ** 2 - ref_im ** 2
chii = 2 * ref_re * ref_im

# chir = ref_re
# chii = ref_im

wavelength_range = [1.0e-4, 2.0e4]
min_step = np.min(np.diff(wavelength))
nsteps = np.int((wavelength_range[1] - wavelength_range[0]) / min_step)
w_grid = np.linspace(wavelength_range[0], wavelength_range[1], nsteps)


chii_intp = interp_imag_dielectric_func(wavelength, chii, w_grid)

chir_trans_grid = ft.hilbert(chii_intp[::-1])[::-1]
chir_zero = integrate.simpson((chii_intp / (1 / w_grid))[::-1], (1 / w_grid)[::-1])
shift = chir_zero - chir_trans_grid[-1]
chir_trans_grid += shift

f = intp.interp1d(w_grid, chir_trans_grid)
chir_trans = f(wavelength)

plt.plot(wavelength, chir_trans, lw=2, label="${\\rm Re}[\chi(\omega)]$")
plt.plot(wavelength, chii, "r-", lw=1, label="${\\rm Im}[\chi(\omega)]$")
plt.plot(w_grid, chii_intp, "r-", lw=1, label="${\\rm Im}[\chi(\omega)]$")
plt.plot(wavelength, chir, "g-", lw=1, label="${\\rm Re}[\chi(\omega)]$")
plt.xlabel("$\lambda$", fontsize=18)
plt.ylabel("$\chi(\omega)$", fontsize=18)
plt.xlim([0.2, 100])
plt.ylim([-30, 60])
plt.xscale("log")
plt.legend()
plt.show()


chi = chir_trans + 1.0j * chii
ref_re_trans = np.sqrt((np.abs(chi) + chir_trans) / 2.0)
ref_im_trans = np.sqrt((np.abs(chi) - chir_trans) / 2.0)

plt.plot(wavelength, ref_re_trans, lw=2, label="${\\rm Re}[N(\omega)]$")
plt.plot(wavelength, ref_im_trans, "r-", lw=1, label="${\\rm Im}[N(\omega)]$")
plt.plot(wavelength, ref_im, "r-", lw=1, label="${\\rm Im}[N(\omega)]$")
plt.plot(wavelength, ref_re, "g-", lw=1, label="${\\rm Re}[N(\omega)]$")
plt.xlabel("$\lambda$", fontsize=18)
plt.ylabel("$\chi(\omega)$", fontsize=18)
plt.xlim([0.2, 100])
plt.ylim([-3, 10])
plt.xscale("log")
plt.legend()
plt.show()


Qabs = np.zeros_like(wavelength)
Qsca = np.zeros_like(wavelength)
Qabs2 = np.zeros_like(wavelength)
Qsca2 = np.zeros_like(wavelength)
Qabs3 = np.zeros_like(wavelength)
Qsca3 = np.zeros_like(wavelength)
g = np.zeros_like(wavelength)
g2 = np.zeros_like(wavelength)
g3 = np.zeros_like(wavelength)
condition1 = np.zeros_like(wavelength)
condition2 = np.zeros_like(wavelength)
grain_size = 0.1
for iw, (wav, n, k) in enumerate(zip(wavelength, ref_re, ref_im)):
   # _, _, Qabs[iw], _ = Q_MIE(n, k, wav, 0.01)
   size_parameter = 2*np.pi* grain_size/ wav
   # refractive_index = n - 1.0j*max(k,1.e-4)
   refractive_index = n - 1.0j*k
   condition1[iw] = size_parameter*np.abs(refractive_index.imag)
   # size_parameter
   # np.abs(1 - refractive_index.real)
   condition2[iw] = 2*size_parameter*np.abs(refractive_index.real-1)
   
   _,Qsca[iw], Qabs[iw], g[iw] = Mie(refractive_index, size_parameter)
   _,Qsca2[iw], Qabs2[iw], g2[iw] = calcMieEfficienciesGeometricOptics(refractive_index, size_parameter)
   _,Qsca3[iw], Qabs3[iw], g3[iw] = calcMieEfficienciesRayleigh(refractive_index, size_parameter) 

plt.plot(wavelength, Qabs);plt.plot(wavelength, Qabs2);plt.plot(wavelength, Qabs3)
plt.xlim([1, 800]);plt.xscale('log');plt.yscale('log')
plt.show()

plt.plot(wavelength, Qsca);plt.plot(wavelength, Qsca2);plt.plot(wavelength, Qsca3)
plt.xlim([1, 800]);plt.xscale('log');plt.yscale('log')
plt.show()

plt.plot(wavelength, g);plt.plot(wavelength, g2);plt.xlim([1, 800]);plt.xscale('log')
plt.show()


plt.plot(2*np.pi*grain_size/wavelength*np.abs(ref_im), np.abs(Qabs-Qabs2)/Qabs2)
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.plot(wavelength, condition1)
plt.yscale('log')
plt.xscale('log')
plt.show()
plt.plot(wavelength, condition2)
plt.yscale('log')
plt.xscale('log')
plt.show()
