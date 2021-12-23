#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 21:49:25 2021

This module provied the functionality to calculate opacities for spherical
grains. The routines for the MIE calculations are taken form the LX-MIE
Mie scattering code. Details can be found in Kitzmann&Heng (2017),
https://arxiv.org/abs/1710.04946 . When used reference to this paper should be
given. The routines listed below are a python implementation of the original
code was written in C++, with slight modifications to the in and output
functionality.

The Functionality valid in the large or small grain limts (geometric optics, 
respectively, Rayleigh) are taken fron Bohren & Huffman and Irvine 1963.

@author: bouwman
"""
import numpy as np
import numba as nb
# from numba import vectorize
from numba import float64, complex128, int64, void
from numba.types import UniTuple

# limits used for the MIE calculations
continued_fraction_max_terms = 10000001
continued_fraction_epsilon = 1e-10
# limits to ckeck if geometric optics are valid
geometric_optics_size_parameter_limit = 500.0
geometric_optics_absorbance_limit = 1.e-5
geometric_optics_reflectance_limit = 1.0
# limits to ckeck if Rayleigh limit is valid
rayleigh_size_parameter_limit = 0.5
rayleigh_size_refraction_limit = 0.5

NaN = float64('nan')


@nb.jit([float64(int64, float64, float64),
         complex128(int64, float64, complex128)], nopython=True, cache=True)
def an(n, nu, z):
    """
    Calculates single terms a_n used in the evaluation of the continued fractions by Lentz (1976)
    See Eq. 9 in Lentz (1976)
    This is the version for both complex as wel as real a_n

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    nu : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return (-1)**((n-1) % 2)*2.0*(nu+n-1)*1.0/z

@nb.jit([float64(int64, float64),
         complex128(int64, complex128)], nopython=True, cache=True)
def startingANcontinuedFractions(nb_mie_terms, mx):
    """
    Calculate the starting value of A_N via the method of continued fractions by Lentz (1976)
    Convergence is reached if two consecutive terms differ by less than "continued_fraction_epsilon"
    Returns A_N
    This is the version for both a complex and a real A_N

    Parameters
    ----------
    nb_mie_terms : TYPE
        DESCRIPTION.
    mx : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    nu = nb_mie_terms + 0.5
  
    #//starting values
    function_numerator = 1.0
    function_denominator = 1.0    
  
    #//n=1
    a_numerator = an(1, nu, mx)
    a_demoninator = 1.0
  
    function_numerator *= a_numerator
    function_denominator *= a_demoninator
  
    #//n=2
    a_numerator = an(2, nu, mx) + 1./a_numerator
    a_demoninator = an(2, nu, mx)
  
    function_numerator *= a_numerator
    function_denominator *= a_demoninator
  
    for i in range(3, continued_fraction_max_terms):
  
        a_i = an(i, nu, mx);
  
        a_numerator = a_i + 1./a_numerator
        a_demoninator = a_i + 1./a_demoninator
  
        function_numerator *= a_numerator
        function_denominator *= a_demoninator
  
        if ( np.abs( (np.abs(a_numerator) - np.abs(a_demoninator) ) / 
                  np.abs(a_numerator) ) < continued_fraction_epsilon):
            break
  
    return function_numerator / function_denominator - (nb_mie_terms*1.)/mx

@nb.jit(void(int64, complex128, float64, complex128[:], complex128[:]), nopython=True, cache=True)
def calcMieCoefficients(nb_mie_terms, refractive_index, size_parameter,
                        mie_coeff_a,  mie_coeff_b):
    """
    Calculates the Mie coefficients a and b that are used for the construction of the Mie series, see Eq. 17
    The required coefficients A_n are calculated via downward recursion, B_n and C_n by upward recursion
    a and b are evaluated up to a number of "nb_mie_terms"

    Parameters
    ----------
    nb_mie_terms : TYPE
        DESCRIPTION.
    refractive_index : TYPE
        DESCRIPTION.
    size_parameter : TYPE
        DESCRIPTION.
    mie_coeff_a : TYPE
        DESCRIPTION.
    mie_coeff_b : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    x = size_parameter
    m = refractive_index
    mx = m * x

    # First, calculate A_n via backward recursion
    # Note that we need A_N(mx), a complex number, and A_N(x), a real number (see Eq. 17)
    # We use the Mie a-coefficient to store the complex values and the Mie b-coefficients for the real numbers
    mie_coeff_a[-1] = startingANcontinuedFractions(nb_mie_terms, mx)
    mie_coeff_b[-1] = startingANcontinuedFractions(nb_mie_terms, x)

    # backward recursion
    # reversed function does not work in numba 0.54.
    # (should get implemented in later versions)
    # for n in reversed(range(1, nb_mie_terms)):
    for i in range(1, nb_mie_terms):
        n = nb_mie_terms+1-i
        mie_coeff_a[n-1] = n/mx - 1.0/(n/mx + mie_coeff_a[n])
        mie_coeff_b[n-1] = n/x - 1.0/(n/x + mie_coeff_b[n].real)

    # //Now we do a forward recursion to calculate B_n, C_n, and the Mie coefficients a_n and b_n
    C_n = 0.0 + 0.0j
    D_n = 0.0 - 1.0j

    # //n=1
    C_n = 1.0 + 1.0j*(np.cos(x) + x * np.sin(x)) / (np.sin(x) - x * np.cos(x))
    C_n = 1./C_n
    D_n = -1.0/x + 1.0/(1.0/x - D_n)

    A_n = mie_coeff_a[1]
    A_n_r = mie_coeff_b[1].real

    mie_coeff_a[1] = C_n * (A_n / m - A_n_r) / (A_n / m - D_n)
    mie_coeff_b[1] = C_n * (A_n * m - A_n_r) / (A_n * m - D_n)

    # //n > 1
    for n in range(2, mie_coeff_a.size):
        A_n = mie_coeff_a[n]
        A_n_r = mie_coeff_b[n].real

        D_n = -n/x + 1.0/(n/x - D_n)
        C_n = C_n * (D_n + n/x)/(A_n_r + n/x)

        mie_coeff_a[n] = C_n * (A_n / m - A_n_r) / (A_n / m - D_n)
        mie_coeff_b[n] = C_n * (A_n * m - A_n_r) / (A_n * m - D_n)


@nb.jit(UniTuple(float64, 4)(float64, complex128[:], complex128[:]),
        nopython=True, cache=True)
def calcMieEfficiencies(size_parameter, mie_coeff_a, mie_coeff_b):
    """
    Calculates the Mie efficiencies, see Eq. 1
    The absorption efficiency is calculated as the difference of the extinction and scattering efficiencies

    Parameters
    ----------
    size_parameter : TYPE
        DESCRIPTION.
    mie_coeff_a : TYPE
        DESCRIPTION.
    mie_coeff_b : TYPE
        DESCRIPTION.

    Returns
    -------
    q_ext : TYPE
        DESCRIPTION.
    q_sca : TYPE
        DESCRIPTION.
    q_abs : TYPE
        DESCRIPTION.

    """
    # q_ext = 0.0
    # q_sca = 0.0
    # #q_bac = 0.0+0.0j
    # for n in range(1, mie_coeff_a.size):
    #     q_sca += (2.0*n + 1.0) * (np.abs(mie_coeff_a[n])**2 +
    #                               np.abs(mie_coeff_b[n])**2)
    #     q_ext += (2.0*n + 1.0) * np.real(mie_coeff_a[n] + mie_coeff_b[n])
    #     #q_bac += (2.0*n + 1.0) * (-1**n) * (mie_coeff_a[n] - mie_coeff_b[n])


    n = np.arange(1, mie_coeff_a.size)
    cn = 2.0 * n + 1.0
    q_ext = 2 * np.sum(cn * (mie_coeff_a.real[1:] + mie_coeff_b.real[1:])) / size_parameter**2
    q_sca = 2 * np.sum(cn * (np.abs(mie_coeff_a[1:])**2 + np.abs(mie_coeff_b[1:])**2)) / size_parameter**2
    q_bac = np.abs(np.sum((-1)**n * cn * (mie_coeff_a[1:] - mie_coeff_b[1:])))**2 / size_parameter**2

    # #q_bac = (q_bac.real**2 + q_bac.imag**2) / (4.0 *np.pi*size_parameter**2)
    # q_sca *= 2.0 / size_parameter**2
    # q_ext *= 2.0 / size_parameter**2
    q_abs = q_ext - q_sca

    return  q_ext, q_sca, q_abs, q_bac

@nb.jit(float64(float64, float64, complex128[:], complex128[:]),
        nopython=True, cache=True)
def calcAsymmetryParameter(q_sca, size_parameter, mie_coeff_a, mie_coeff_b):
    """
    Calculate and return the asymmetry parameter
    See Bohren&Huffman, page 120, for details on the equation

    Parameters
    ----------
    q_sca : TYPE
        DESCRIPTION.
    size_parameter : TYPE
        DESCRIPTION.
    mie_coeff_a : TYPE
        DESCRIPTION.
    mie_coeff_b : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    g = 0
    
    for n in range(1, mie_coeff_a.size-1):
        cn = 2.0 * n + 1.0
        c1n = n * (n + 2.0) / (n + 1.0)
        c2n = cn / n / (n + 1.0)
        g += c1n * \
            np.real(mie_coeff_a[n] * np.conj(mie_coeff_a[n+1]) +
                    mie_coeff_b[n] * np.conj(mie_coeff_b[n+1])) + \
            c2n * np.real(mie_coeff_a[n] * np.conj(mie_coeff_b[n]))
  
    return g * 4./(size_parameter * size_parameter * q_sca)

# @nb.jit(void(float64, float64[:], float64[:]),
#         nopython=True, cache=True)
# def calcAngularFunctions(angle, pi_n, tau_n):
#     """
#     Calculate the angular eigenfunction pi and tau at a specified angle (Eq. 6)
#     Instead of evaluating the Legendre polynomials directly, we employ upward recurrence relations
#     We here follow the notation of Wiscombe (1979)

#     Parameters
#     ----------
#     angle : TYPE
#         DESCRIPTION.
#     pi_n : TYPE
#         DESCRIPTION.
#     tau_n : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     pi_n[0] = 0.0
#     pi_n[1] = 1.0
  
#     for n in range(1, pi_n.size-1):
#         s = angle * pi_n[n]
#         t = s - pi_n[n-1]
  
#         pi_n[n+1] = s + (n + 1.0)/(n*1.0) * t
  
#         tau_n[n] = n * t - pi_n[n-1]
  
#     n = pi_n.size-1
  
#     s = angle * pi_n[n]
#     t = s - pi_n[n-1]

#     tau_n[n] = n * t - pi_n[n-1]

# @nb.jit(UniTuple(complex128, 2)(float64, complex128[:], complex128[:]),
#         nopython=True, cache=True)
# def calcScatteringAmplitudes(angle, mie_coeff_a, mie_coeff_b):
#     """
#     Calculate the scattering amplitudes S_1 and S_2 at a specified angle
#     The Mie intensities in Eq. 5 are given by i_1 = (abs[S_1])^2, i_2 = (abs[S_2])^2
#     The intensities/amplitudes can be used to calculate the scattering phase function (see Eq. 7)
#     See Wiscombe (1979) for the recurrence relations used below

#     Parameters
#     ----------
#     angle : TYPE
#         DESCRIPTION.
#     mie_coeff_a : TYPE
#         DESCRIPTION.
#     mie_coeff_b : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     s_1 : TYPE
#         DESCRIPTION.
#     s_2 : TYPE
#         DESCRIPTION.

#     """
#     pi_n = np.zeros((mie_coeff_a.size))
#     tau_n = np.zeros((mie_coeff_a.size))
  
#     calcAngularFunctions(angle, pi_n, tau_n)
  
#     s_plus = 0.0 + 0.0j
#     s_minus = 0.0 + 0.0j
  
#     for n in range(1, pi_n.size):
#         s_plus  += (2.*n + 1.0)/(n * (n + 1.0)) * \
#             (mie_coeff_a[n] + mie_coeff_b[n]) * (pi_n[n] + tau_n[n])
#         s_minus += (2.*n + 1.0)/(n * (n + 1.0)) * \
#             (mie_coeff_a[n] - mie_coeff_b[n]) * (pi_n[n] - tau_n[n])
  
#     s_1 = 0.5 * (s_plus + s_minus) 
#     s_2 = 0.5 * (s_plus - s_minus) 
#     return s_1, s_2


@nb.jit(UniTuple(complex128, 2)(float64, complex128[:], complex128[:]),
        nopython=True, cache=True)
def calcScatteringAmplitudes(cos_angle, mie_coeff_a, mie_coeff_b):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.
    The units are weird, sr**(-0.5)

    Parameters
    ----------
    cos_angle : TYPE
        Drray of angles, cos(theta), to calculate scattering amplitudes
    mie_coeff_a : TYPE
        DESCRIPTION.
    mie_coeff_b : TYPE
        DESCRIPTION.

    Returns
    -------
    S1 : TYPE
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    S2 : TYPE
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    a = mie_coeff_a[1:]
    b = mie_coeff_b[1:]

    S1 = 0.0 + 0.0j
    S2 = 0.0 + 0.0j

    nstop = len(a)

    pi_nm2 = 0
    pi_nm1 = 1
    for n in range(1, nstop):
        tau_nm1 = n * cos_angle * pi_nm1 - (n + 1) * pi_nm2
        S1 += (2 * n + 1) * (pi_nm1 * a[n-1]
                                + tau_nm1 * b[n-1]) / (n + 1) / n
        S2 += (2 * n + 1) * (tau_nm1 * a[n-1]
                                + pi_nm1 * b[n-1]) / (n + 1) / n
        temp = pi_nm1
        pi_nm1 = ((2 * n + 1) * cos_angle * pi_nm1 - (n + 1) * pi_nm2) / n
        pi_nm2 = temp

    # calculate norm = sqrt(pi * Qext * x**2)
    n = np.arange(1, nstop+1)
    norm = np.sqrt(2 * np.pi * np.sum((2 * n + 1) * (a.real + b.real)))

    S1 /= norm
    S2 /= norm

    return S1, S2


@nb.jit(void(complex128[:], complex128[:], int64, complex128[:], complex128[:]),
        nopython=True, cache=True)
def calcMuellerArrays(mie_coeff_a, mie_coeff_b, nb_mie_terms, C, D):
    """
    Calculate the Mueller arrays D and C, required for the Legendre series of the phase function
    This follows the procedure and notation of Dave (1988)

    Parameters
    ----------
    mie_coeff_a : TYPE
        DESCRIPTION.
    mie_coeff_b : TYPE
        DESCRIPTION.
    nb_mie_terms : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    C[nb_mie_terms + 2] = 0.0 + 0.0j
    D[nb_mie_terms + 2] = 0.0 + 0.0j

    C[nb_mie_terms + 1] = (1. - 1./(nb_mie_terms+1)) * mie_coeff_b[nb_mie_terms]
    D[nb_mie_terms + 1] = (1. - 1./(nb_mie_terms+1)) * mie_coeff_a[nb_mie_terms]

    C[nb_mie_terms] = (1./nb_mie_terms + 1./(nb_mie_terms+1)) * \
        mie_coeff_a[nb_mie_terms] + (1. - 1./nb_mie_terms) * mie_coeff_b[nb_mie_terms-1]
    D[nb_mie_terms] = (1./nb_mie_terms + 1./(nb_mie_terms+1)) * \
        mie_coeff_b[nb_mie_terms] + (1. - 1./nb_mie_terms) * mie_coeff_a[nb_mie_terms-1]

    for i in range(2, nb_mie_terms):
        k = nb_mie_terms + 1 - i

        C[k] = C[k+2] - (1. + 1./(k+1)) * mie_coeff_b[k+1] + (1./k + 1./(k+1)) * \
            mie_coeff_a[k] + (1. - 1./k) * mie_coeff_b[k-1]
        D[k] = D[k+2] - (1. + 1./(k+1)) * mie_coeff_a[k+1] + (1./k + 1./(k+1)) * \
            mie_coeff_b[k] + (1. - 1./k) * mie_coeff_a[k-1]

    C[1] = C[3] + 1.5 * (mie_coeff_a[1] - mie_coeff_b[2])
    D[1] = D[3] + 1.5 * (mie_coeff_b[1] - mie_coeff_a[2])

    for k in range(1, nb_mie_terms+3):
        C[k] = (2.*k - 1) * C[k]
        D[k] = (2.*k - 1) * D[k]

@nb.jit(void(complex128[:], complex128[:], int64, int64, float64[:]),
        nopython=True, cache=True)
def calcLegendreMoments(mie_coeff_a, mie_coeff_b, nb_max_moments, nb_mie_terms,
                        legendre_moments):
    """
    Calculate the moments of the Legendre series of the phase function
    It follows the procedure of Dave (1988) 

    Parameters
    ----------
    mie_coeff_a : TYPE
        DESCRIPTION.
    mie_coeff_b : TYPE
        DESCRIPTION.
    nb_max_moments : TYPE
        DESCRIPTION.
    nb_mie_terms : TYPE
        DESCRIPTION.
    legendre_moments : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    a_m = np.zeros((nb_mie_terms+2))
    b_i = np.zeros((nb_mie_terms+2))
    b_i_delta = np.zeros((nb_mie_terms+2))

    C = np.zeros((nb_mie_terms+5), dtype=complex128)
    D = np.zeros((nb_mie_terms+5), dtype=complex128)

    calcMuellerArrays(mie_coeff_a, mie_coeff_b, nb_mie_terms, C, D);

    for l in range(0, nb_max_moments):
        even_moment = not (l & 1)
        ld2 = l//2
        legendre_moments[l] = 0;
    
        if (l == 0):
            small_delta = 1
            for m in range(0, nb_mie_terms+1):
                a_m[m] = 2.0 * 1./(2.*m + 1.)
            b_i[0] = 1.0
        elif even_moment:
            small_delta = 1
            for m in range(ld2, nb_mie_terms+1):
                a_m[m] = (1. + 1./(2.*m - l + 1)) * a_m[m]
            for i in range(0, ld2):
                b_i[i] = (1. - 1./(l - 2.*i)) * b_i[i]
            b_i[ld2] = (2. - 1./l) * b_i[ld2-1]
        else:
            small_delta = 2
            for m in range(ld2, nb_mie_terms+1):
                a_m[m] = (1. - 1./(2.*m + l + 2.)) * a_m[m]
            for i in range(0, ld2+1):
                b_i[i] = (1. - 1./(l + 2.*i + 1)) * b_i[i]

        mmax = nb_mie_terms - small_delta
        mmax += 1
        imax = min(ld2, mmax - ld2)

        for i in range(0, imax+1):
            b_i_delta[i] = b_i[i]

        if even_moment:
            b_i_delta[0] = 0.5 * b_i_delta[0]

        for i in range(0, imax+1):
            sum = 0
            for m in range(ld2, mmax-i+1):
                sum += a_m[m] * \
                    (np.real(C[m-i+1] * np.conj(C[m+i+small_delta])) +
                     np.real(D[m-i+1] * np.conj(D[m+i+small_delta]))
                     )
            legendre_moments[l] += b_i_delta[i] * sum

        legendre_moments[l] *= 0.5

@nb.jit(int64(float64), nopython=True, cache=True)
def numberOfMieTerms(size_parameter):
    """
    Calculate the maximum number of terms in the Mie series
    See Eq. 22    

    Parameters
    ----------
    size_parameter : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return int( size_parameter + 4.3 * np.power(size_parameter, 1./3.) + 2)

@nb.jit(UniTuple(float64, 5)(complex128, float64), nopython=True, cache=True)
def Mie(refractive_index, size_parameter):
    """
    Mie calculations.

    Parameters
    ----------
    refractive_index : TYPE
        DESCRIPTION.
    size_parameter : TYPE
        DESCRIPTION.

    Returns
    -------
    q_ext : TYPE
        DESCRIPTION.
    q_sca : TYPE
        DESCRIPTION.
    q_abs : TYPE
        DESCRIPTION.
    asymmetry_parameter : TYPE
        DESCRIPTION.

    """

    nb_mie_terms = numberOfMieTerms(size_parameter)

    mie_coeff_a = np.zeros((nb_mie_terms+1), dtype=complex128)
    mie_coeff_b = np.zeros((nb_mie_terms+1), dtype=complex128)
  
    calcMieCoefficients(nb_mie_terms, refractive_index, size_parameter,
                        mie_coeff_a, mie_coeff_b)
    q_ext, q_sca, q_abs, q_bac = \
        calcMieEfficiencies(size_parameter, mie_coeff_a, mie_coeff_b)
  
    asymmetry_parameter = \
        calcAsymmetryParameter(q_sca, size_parameter, mie_coeff_a, mie_coeff_b)

    return q_ext, q_sca, q_abs, q_bac, asymmetry_parameter

@nb.jit([float64(complex128, complex128, float64, float64),
         float64[:](complex128[:], complex128[:], float64, float64)],
        nopython=True, cache=True)
def phaseFunction(s1, s2, size_parameter, q_sca):
    """
    Returns the value of the scattering phase functions for given
    (angular-dependent) scattering amplitudes
    See Eq. 7  

    Parameters
    ----------
    s1 : TYPE
        DESCRIPTION.
    s2 : TYPE
        DESCRIPTION.
    size_parameter : TYPE
        DESCRIPTION.
    q_sca : TYPE
        DESCRIPTION.

    Returns
    -------
    phase_function : TYPE
        DESCRIPTION.

    """
    phase_function = 2.0/(size_parameter*size_parameter) * \
        (np.abs(s1)**2 + np.abs(s2)**2) / q_sca
    return phase_function

@nb.jit([float64(float64, complex128), float64[:](float64[:], complex128)],
        nopython=True, cache=True)
def reflection(theta_i, refractive_index):
    """
    Calculates the Reflectans for large grains.

    Parameters
    ----------
    m : 'complex'
        Complex refractive Index.
    theta_i: 'float'
        Scatering Angle

    Returns
    -------
    reflection: 'float'
        Reflectans for large grains
    
    Notes
    -----
    For details see Section 2.7 and Chapter 7 of Bohren and Huffman.
    Note that the use of this is only valid for a size parameter >> 1
    
    """     
    sin_theta_t = np.sin(theta_i)/refractive_index
    cos_theta_t = np.sqrt(1.0-sin_theta_t**2)
    cos_theta_i = np.cos(theta_i)
    #  r for E parallel to plane
    rpll = (cos_theta_t-refractive_index*cos_theta_i) / \
        (cos_theta_t+refractive_index*cos_theta_i)
    #  r for E perp. to plane
    rper = (cos_theta_i-refractive_index*cos_theta_t) / \
        (cos_theta_i+refractive_index*cos_theta_t)
    #  R = ½(|rpll|²+|rper|²)
    reflection = 0.5*(np.abs(rpll)**2 + np.abs(rper)**2)
    return reflection   

@nb.jit(float64(complex128), nopython=True, cache=True)
def QscaGeometricOptics(refractive_index):
    """
    Intgrates the reflection coefficient.

    Parameters
    ----------
    refractive_index : 'complex'
        Complex refractive index.

    Returns
    -------
    q_sca: 'float'
        Scattering efficiency.

    Note:
    -----
    Trapezium rule integration from 0 to pi/2 of R*sin(theta)*cons(theta)
    See also forula 7.5 and 7.6 in Bohren and Huffman.
    Note that this is only valid for large grains and that k, the imaginary
    part of the complex refractive index is non zero.
    """
    n_theta = 180
    theta_i = np.linspace(0.0, 0.5*np.pi, n_theta+1)
    integrand = reflection(theta_i, refractive_index) * \
        np.sin(theta_i)*np.cos(theta_i)
    d_theta_i = 0.5*np.pi/n_theta
    q_ref = 2.0*(d_theta_i/2)*np.sum(integrand[1:]+integrand[:-1])
    q_sca = 1.0+q_ref
    return q_sca

@nb.jit(UniTuple(float64[:], 3)(float64[:], float64[:], complex128),
        nopython=True, cache=True)
def integrandTermsAsymmetryFactor(sigma, r_term, refractive_index):
    """
    Helper function for calculation asymmetrey parameter

    Parameters
    ----------
    sigma : TYPE
        DESCRIPTION.
    r_term : TYPE
        DESCRIPTION.
    refractive_index : TYPE
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.

    Notes
    -----
    See section 3d of Irvine 1963, BAN 17, no3,176-184
    """
    A = 4.0*r_term**4 * (refractive_index.real**2 - sigma) * \
        (2.0*sigma - 1.0)/refractive_index.real**2
    B = (1.0 - r_term**2)**2 * ((2.0*sigma - 1.0) *
                                (2.0*sigma/refractive_index.real**2 - 1.0) +
                                4.0*sigma*np.sqrt(1.0 - sigma) *
                                np.sqrt(refractive_index.real**2 - sigma) /
                                refractive_index.real**2)
    C = 1.0 - 2.0*r_term**2 * (2.0*sigma/refractive_index.real**2 - 1.0) + \
        r_term**4
    return A, B, C

@nb.jit(float64(complex128), nopython=True, cache=True)
def asymmetryFactorGeometricOptics(refractive_index):
    """
    Calculate the asymmetry parameter.

    Parameters
    ----------
    refractive_index : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    
    Notes
    -----
    See W.E.irvine 1963

    """
    if np.abs(refractive_index.real - 1.0) < 1.e-6:
        return 1.0
    n_angles = 2000
    sigma = np.linspace(0.0, 1.0, n_angles+1)
    
    r_term1 = np.sqrt(1.0-sigma)
    r_term2 = np.sqrt(1.0-sigma/refractive_index.real**2)
    r_per = (r_term1-refractive_index.real*r_term2) / \
        (r_term1+refractive_index.real*r_term2)
    
    r_par = (refractive_index.real*r_term1-r_term2) / \
        (refractive_index.real*r_term1+r_term2)
    
    A,B,C = integrandTermsAsymmetryFactor(sigma, r_per, refractive_index)
    d_sigma = 1.0/n_angles
    integrand = (A + B) / C
    gamma_per = (d_sigma/2.0)*np.sum(integrand[1:] + integrand[:-1])
    A,B,C = integrandTermsAsymmetryFactor(sigma, r_par, refractive_index)
    integrand = (A + B) / C
    gamma_par = (d_sigma/2.0)*np.sum(integrand[1:] + integrand[:-1])
    
    gamma = 0.5*(gamma_per + gamma_par)
    
    g = 0.5*(1+gamma)
    return g

@nb.jit([UniTuple(float64, 5)(complex128, float64)],
        nopython=True, cache=True)
def calcMieEfficienciesGeometricOptics(refractive_index, size_parameter):
    """
    Return Q values in large grain limit

    Parameters
    ----------
    refractive_index : TYPE
        DESCRIPTION.

    Returns
    -------
    q_ext : TYPE
        DESCRIPTION.
    q_sca : TYPE
        DESCRIPTION.
    q_abs : TYPE
        DESCRIPTION.
    asymmetry_parameter : TYPE
        DESCRIPTION.

    Notes
    -----
    See chapter 7 Bohren & Huffman
    """

    if np.abs(refractive_index.imag) < geometric_optics_absorbance_limit:
        # k ~ 0
        q_ext = 2.0
        q_sca = 2.0
        q_abs = 0.0
    else:
        q_sca = QscaGeometricOptics(refractive_index)
        q_ext = 2.0
        q_abs = q_ext-q_sca
    q_bac = reflection(0.0, refractive_index)
    asymmetry_parameter = 1-(1-q_abs)/q_sca
    return q_ext, q_sca, q_abs, q_bac, asymmetry_parameter  


@nb.jit([UniTuple(float64, 4)(complex128, float64)],
        nopython=True, cache=True)
def calcMieEfficienciesRayleigh(refractive_index, size_parameter):
    """
    Calculate efficiencies using small particle limit. 

    Parameters
    ----------
    refractive_index : TYPE
        DESCRIPTION.
    size_parameter : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    Notes
    -----
    Best use for size_prameters smakker then 0.001
    
    The asymmetry parameter can be neglected for small particles
    
    See Bohren & Huffman Section 5.1
    """
    if size_parameter > rayleigh_size_parameter_limit:
        # check x << 1
        return NaN, NaN, NaN, NaN
    elif size_parameter*np.abs(refractive_index) >= \
        rayleigh_size_refraction_limit:
        # check |m|x << 1
        return NaN, NaN, NaN, NaN
    C=(refractive_index**2-1.0)/(refractive_index**2+2.0)
    q_sca = (8.0/3.0)*size_parameter**4 * np.abs(C)**2
    A = -4.0*C.imag
    B = (1.0 + (1.0/3.0)*size_parameter**3 * A)    
    q_abs = size_parameter*A*B
    q_ext = q_abs + q_sca
    # returns always zero array of same size as q_ext
    asymmetry_parameter = 0.0 * q_ext
    return q_ext, q_sca, q_abs, asymmetry_parameter  

@nb.jit([void(float64, nb.types.Omitted(None)),
         void(float64, float64)], nopython=True, cache=True)
def test_md(x, y=None):
    if y == None:
        print(x)
    else:
        print(x, y)
