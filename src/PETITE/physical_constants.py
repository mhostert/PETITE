try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
import numpy as np
from scipy import interpolate

"""For internal consistency, all physical units and constants are defined here.
"""

TeV = 1e3
GeV = 1
MeV = 1e-3
keV = 1e-6
eV = 1e-9
centimeter = 1
#  https://pdg.lbl.gov/2022/reviews/contents_sports.html
alpha_em = 1 / 137.035999
m_electron = 510.998950 * keV
m_proton = 938.272088 * MeV
m_proton_grams = 1.67262192369e-24
m_neutron = 939.56542052 * MeV
atomic_mass_unit = 931.49410241 * MeV
n_avogadro = 6.0221409e23
hbarc = 0.1973269804e-13 * GeV * centimeter
GeVsqcm2 = hbarc**2
cmtom = 0.01

m_muon = 105.6583755 * MeV
m_tau = 1776.93 * MeV
m_pi0, c_tau_pi0 = 134.9768 * MeV, 25.3 * (1e-9)  # meters
m_pi_pm, c_tau_pi_pm = 139.57039 * MeV, 7.8045  # meters

m_eta, Gamma_eta = 547.862 * MeV, 1.31 * keV
m_eta_prime, Gamma_eta_prime = 957.78 * MeV, 0.188 * MeV

m_omega = 782.65 * MeV

# Running alphaQED
_Q, _inv_alpha_em = np.genfromtxt(
    files("PETITE.include").joinpath("alpha_QED_running_posQ2.dat").open(),
    unpack=True,
)
running_alpha_em = interpolate.interp1d(
    _Q,
    1.0 / _inv_alpha_em,
    bounds_error=False,
    fill_value=(1.0 / _inv_alpha_em[0], 1.0 / _inv_alpha_em[-1]),
)
