


Qabs = np.zeros_like(wavelength)
Qbac = np.zeros_like(wavelength)
Qsca = np.zeros_like(wavelength)
Qabs2 = np.zeros_like(wavelength)
Qbac2 = np.zeros_like(wavelength)
Qsca2 = np.zeros_like(wavelength)
Qabs3 = np.zeros_like(wavelength)
Qsca3 = np.zeros_like(wavelength)
g = np.zeros_like(wavelength)
g2 = np.zeros_like(wavelength)
g3 = np.zeros_like(wavelength)
condition1 = np.zeros_like(wavelength)
condition2 = np.zeros_like(wavelength)
grain_size = 70000
for iw, (wav, n, k) in enumerate(zip(wavelength, ref_re, ref_im)):
   # _, _, Qabs[iw], _ = Q_MIE(n, k, wav, 0.01)
   size_parameter = 2*np.pi* grain_size/ wav
   # refractive_index = n - 1.0j*max(k,1.e-4)
   refractive_index = n - 1.0j*k
   condition1[iw] = size_parameter*np.abs(refractive_index.imag)
   # size_parameter
   # np.abs(1 - refractive_index.real)
   condition2[iw] = 2*size_parameter*np.abs(refractive_index.real-1)
   
   _,Qsca[iw], Qabs[iw], Qbac[iw],g[iw] = Mie(refractive_index, size_parameter)
   _,Qsca2[iw], Qabs2[iw], Qbac2[iw], g2[iw] = calcMieEfficienciesGeometricOptics(refractive_index, size_parameter)
   _,Qsca3[iw], Qabs3[iw], g3[iw] = calcMieEfficienciesRayleigh(refractive_index, size_parameter)
   # g2[iw] = 1 -0.6* reflection(0.0, refractive_index)/ reflection(0.5*np.pi, refractive_index)
   

plt.plot(wavelength, Qabs);plt.plot(wavelength, Qabs2);plt.plot(wavelength, Qabs3)
plt.xlim([1, 800]);plt.xscale('log');plt.yscale('log')
plt.show()

plt.plot(wavelength, Qsca);plt.plot(wavelength, Qsca2);plt.plot(wavelength, Qsca3)
plt.xlim([1, 800]);plt.xscale('log');plt.yscale('log')
plt.show()

plt.plot(wavelength, g);plt.plot(wavelength, g2);plt.xlim([1, 800]);plt.xscale('log')
plt.show()

plt.plot(wavelength, Qbac);plt.plot(wavelength, Qbac2)
plt.xlim([1, 800]);plt.xscale('log');plt.yscale('log')
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




####################################################################################

radius = 2                      # in microns
lambda0 = np.linspace(0.2, 1.2, 200)  # also in microns
x = 2 * np.pi * radius / lambda0

Qabs = np.zeros_like(lambda0)
Qbac = np.zeros_like(lambda0)
Qsca = np.zeros_like(lambda0)
g = np.zeros_like(lambda0)


# from https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
m2 = 1 + 1.03961212 / (1 - 0.00600069867 / lambda0**2)
m2 += 0.231792344 / (1 - 0.0200179144 / lambda0**2)
m2 += 1.01046945 / (1 - 103.560653 / lambda0**2)
m = np.sqrt(m2)

for iw in range(200):
   _, Qsca[iw], Qabs[iw], Qbac[iw], g[iw] = Mie(m[iw] - 0.0j, x[iw])

plt.plot(lambda0 * 1000, Qsca)
plt.title("BK7 glass spheres 4 micron diameter")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scattering Efficiency (-)")
# plt.show()







