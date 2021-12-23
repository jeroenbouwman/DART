#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:27:57 2021

@author: bouwman
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython
import dart

x = np.linspace(0.1, 1000, 300)

# mie() will automatically try to do the right thing

qext, qsca, qback, g = miepython.mie(1.5, x)
plt.plot(x, qext, color='red', label="1.5")

qext, qsca, qback, g = miepython.mie(1.5 - 0.1j, x)
plt.plot(x, qext, color='blue', label="1.5-0.1j")

qext = np.zeros_like(x)
qsca = np.zeros_like(x)
g = np.zeros_like(x)
qback = np.zeros_like(x)
for i, xi in enumerate(x):
   qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.0j, xi)
plt.plot(x, qext, color='green', label="1.5 LX-MIE")

for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] =dart.q_mie.Mie(1.5-0.1j, xi)
plt.plot(x, qext, color='orange', label="1.5-01 LX-MIE")

plt.title("Comparison of extinction for absorbing and non-absorbing spheres")
plt.xlabel("Size Parameter (-)")
plt.ylabel("Qext")
plt.xscale('log')
plt.legend()
plt.show()


qext, qsca, qback, g = miepython.mie(1.5, x)
plt.plot(x, qsca, color='red', label="1.5")

qext, qsca, qback, g = miepython.mie(1.5 - 0.1j, x)
plt.plot(x, qsca, color='blue', label="1.5-0.1j")

qext = np.zeros_like(x)
qsca = np.zeros_like(x)
g = np.zeros_like(x)
qback = np.zeros_like(x)
for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.0j, xi)
plt.plot(x, qsca, color='green', label="1.5 LX-MIE")

for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.1j, xi)
plt.plot(x, qsca, color='orange', label="1.5-0.1j LX-MIE")

plt.title("Comparison of extinction for absorbing and non-absorbing spheres")
plt.xlabel("Size Parameter (-)")
plt.ylabel("Qsca")
plt.xscale('log')
plt.legend()
plt.show()


qext, qsca, qback, g = miepython.mie(1.5, x)
plt.plot(x, g, color='red', label="1.5")
qext, qsca, qback, g = miepython.mie(1.5 - 0.1j, x)
plt.plot(x, g, color='blue', label="1.5-0.1j")

qext = np.zeros_like(x)
qsca = np.zeros_like(x)
g = np.zeros_like(x)
qback = np.zeros_like(x)
for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.0j, xi)
plt.plot(x, g, color='green', label="1.5 LX-MIE")

for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.1j, xi)
plt.plot(x, g, color='orange', label="1.5-0.1j LX-MIE")

plt.title("Comparison of extinction for absorbing and non-absorbing spheres")
plt.xlabel("Size Parameter (-)")
plt.ylabel("g")
plt.legend()
plt.show()

qext, qsca, qback, g = miepython.mie(1.5, x)
#plt.plot(x, qback, color='red', label="1.5")
qext, qsca, qback, g = miepython.mie(1.5 - 0.1j, x)
plt.plot(x, qback, color='blue', label="1.5-0.1j")

qext = np.zeros_like(x)
qsca = np.zeros_like(x)
g = np.zeros_like(x)
qback = np.zeros_like(x)
for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.0j, xi)
#plt.plot(x, qback*20, color='green', label="1.5 LX-MIE")

for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.1j, xi)
plt.plot(x, qback, color='orange', label="1.5-0.1j LX-MIE")

plt.title("Comparison of extinction for absorbing and non-absorbing spheres")
plt.xlabel("Size Parameter (-)")
plt.ylabel("Qback")
plt.legend()
plt.show()


qext, qsca, qback, g = miepython.mie(1.5, x)
plt.plot(x, qback, color='red', label="1.5")
qext, qsca, qback, g = miepython.mie(1.5 - 0.1j, x)
#plt.plot(x, qback, color='blue', label="1.5-0.1j")

qext = np.zeros_like(x)
qsca = np.zeros_like(x)
g = np.zeros_like(x)
qback = np.zeros_like(x)
for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.0j, xi)
plt.plot(x, qback, color='green', label="1.5 LX-MIE")

for i, xi in enumerate(x):
    qext[i], qsca[i], _, qback[i], g[i] = dart.q_mie.Mie(1.5-0.1j, xi)
#plt.plot(x, qback, color='orange', label="1.5-01 LX-MIE")

plt.title("Comparison of extinction for absorbing and non-absorbing spheres")
plt.xlabel("Size Parameter (-)")
plt.ylabel("Qback")
plt.legend()
plt.show()



##################################

x = np.linspace(0.1,4,50)

m = 3.41-1.94j
qext, qsca, qback, g = miepython.mie(m,x)
plt.plot(x,qback)
plt.text(0.6,0,"m=3.41-1.94j")

m = 10000
qext, qsca, qback, g = miepython.mie(m,x)
plt.plot(x,qback)
plt.text(1.2,3.0,"m=10,000")

plt.xlabel("Size Parameter")
plt.ylabel(r"$Q_{back}$")
plt.title("van de Hulst Figure 61")
plt.grid(True)
plt.show()


m = 10000
x = 1.0
qext, qsca, qabs,qback, g = dart.q_mie.Mie(m, x)
# qext, qsca, qback, g = miepython.mie(m,x)
nb_mie_terms = dart.q_mie.numberOfMieTerms(x)
mie_coeff_a = np.zeros((nb_mie_terms+1), dtype=complex)
mie_coeff_b = np.zeros((nb_mie_terms+1), dtype=complex)

dart.q_mie.calcMieCoefficients(nb_mie_terms, m,
                                x, mie_coeff_a, mie_coeff_b)
theta=-180.
mu = np.cos(theta/180*np.pi)
s1_mp, s2_mp = miepython.mie_S1_S2(m,x, mu)
s1, s2 = dart.q_mie.calcScatteringAmplitudes(mu, mie_coeff_a,
                                              mie_coeff_b)
phase = (abs(s1)**2+abs(s2)**2)/2
print(s1, s2)
print(4*np.pi*qsca*phase)
print(qback)


x_all = np.linspace(0.1,4,50)
Qback_alt = np.zeros_like(x_all)
m = 10000
for i, x in enumerate(x_all): 
    #x=2.0
    qext, qsca, qback, g = miepython.mie(m,x)
    theta = np.linspace(-180,180,180)
    mu = np.cos(theta/180*np.pi)
    s1,s2 = miepython.mie_S1_S2(m,x,mu)
    phase = (abs(s1[0])**2+abs(s2[0])**2)/2
    Qback_alt[i] = 4*np.pi*qsca*phase

x = x_all
qext, qsca, qback, g = miepython.mie(m,x)
plt.plot(x,qback)
plt.text(0.6,0,"m={}".format(m))

plt.plot(x, Qback_alt)
plt.xlabel("Size Parameter")
plt.ylabel(r"$Q_{back}$")
plt.title("van de Hulst Figure 61")
plt.grid(True)
plt.show()

##################################################################





lambda0 = 1             # microns
a = lambda0/10          # also microns
k = 2*np.pi/lambda0     # per micron

m = 1.5 
x = a * k
geometric_cross_section = np.pi * a**2

theta = np.linspace(-180,180,180)
mu = np.cos(theta/180*np.pi)
s1,s2 = miepython.mie_S1_S2(m,x,mu)
phase = (abs(s1[0])**2+abs(s2[0])**2)/2

print('     unpolarized =',phase)
print('   |s1[-180]|**2 =',abs(s1[0]**2))
print('   |s2[-180]|**2 =',abs(s2[0]**2))
print('   |s1[ 180]|**2 =',abs(s1[179]**2))
print('   |s2[ 180]|**2 =',abs(s2[179]**2))
print()

qext, qsca, qback, g = miepython.mie(m,x)

Cback = qback * geometric_cross_section
Csca  = qsca  * geometric_cross_section

print('            Csca =',Csca)
print('           Cback =',Cback)
print('4*pi*Csca*p(180) =',4*np.pi*Csca*phase)

print('           Qback =',qback)
print('4*pi*Qsca*p(180) =',4*np.pi*qsca*phase)

