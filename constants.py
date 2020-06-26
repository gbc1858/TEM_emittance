z0 = 0  # start of the transport matrix
ze = 29.455  # end of the transport matrix [cm]
zstep = 0.01  # step size of the sliced magnetic field
ek = 0.2  # kinetic energy of the beam (MeV)
e0 = 0.511  # electron rest energy (MeV)
e = ek + e0

# constants
c = 299792458
me = 9.10938291e-31
qe = 1.602176565e-19

# rest mass energy (eV)
rest_e = me * c ** 2 / qe
betagamma = (e ** 2 / e0 ** 2 - 1) ** 0.5
