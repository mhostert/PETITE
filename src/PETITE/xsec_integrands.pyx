import numpy as np

# use exp from C
from libc.math cimport exp, cos, sin, sqrt, log, pi

cdef double m_electron = 0.5109989461e-3  # GeV
cdef double alpha_em = 1.0 / 137.035999139  # fine structure constant

def c_dsigma_brem_dimensionless(double[:, ::1] x, double ep, double Egamma_min, int Z):
    """Standard Model Bremsstrahlung in the Small-Angle Approximation
    e (ep) + Z -> e (epp) + gamma (w) + Z
    Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)

    Input parameters needed:
        ep (incident electron energy)
        Z (Target Atomic Number)
    """

    cdef int i          # labels integration point
    cdef int dim = x.shape[1]
    cdef double[::1] ans = np.zeros(x.shape[0])
    cdef double w, d, dp, ph, epp, qsq, PF, jacobian_factor, FF, T1, T2, T3, T4

    for i in range(x.shape[0]):
        
        x1 = x[i,0] 
        x2 = x[i,1] 
        x3 = x[i,2] 
        x4 = x[i,3]
        
        w = Egamma_min + x1 * (ep - m_electron - Egamma_min)
        d = ep / (2 * m_electron) * (x2 + x3)
        dp = ep / (2 * m_electron) * (x2 - x3) 
        ph = (x4 - 1 / 2) * 2 * pi                


        epp = ep - w

        if ((Egamma_min < w)
            and (w < ep - m_electron)
            and (m_electron < epp)
            and (epp < ep)
            and (d > 0.0)
            and (dp > 0.0)):

            qsq = m_electron**2 * (
                (d**2 + dp**2 - 2 * d * dp * cos(ph))
                + m_electron**2 * ((1 + d**2) / (2 * ep) - (1 + dp**2) / (2 * epp)) ** 2
            )
            PF = (
                8.0
                / pi
                * alpha_em
                * (alpha_em / m_electron) ** 2
                * (epp * m_electron**4)
                / (w * ep * qsq**2)
                * d
                * dp
            )
            jacobian_factor = pi * ep**2 * (ep - m_electron - Egamma_min) / m_electron**2
            FF = g2_elastic_c(Z, qsq)
            T1 = d**2 / (1 + d**2) ** 2
            T2 = dp**2 / (1 + dp**2) ** 2
            T3 = w**2 / (2 * ep * epp) * (d**2 + dp**2) / ((1 + d**2) * (1 + dp**2))
            T4 = -(epp / ep + ep / epp) * (d * dp * cos(ph)) / ((1 + d**2) * (1 + dp**2))
            
            ans[i] = PF * (T1 + T2 + T3 + T4) * jacobian_factor * FF

    return ans



def c_dsigma_pairprod_dimensionless(double[:, ::1] x, double w, int Z):
    """Standard Model Pair Production in the Small-Angle Approximation
    gamma (w) + Z -> e+ (epp) + e- (epm) + Z
    Outgoing kinematics given by epp, dp (delta+), dm (delta-), and ph (phi)

    Input parameters needed:
        w (incident photon energy)
        Z (Target Atomic Number)
    """

    cdef int i          # labels integration point
    cdef int dim = x.shape[1]
    cdef double[::1] ans = np.zeros(x.shape[0])
    cdef double x1, x2, x3, x4, d, dp, dm, ph, epp, epm, qsq, PF, jacobian_factor, FF, T1, T2, T3, T4


    for i in range(x.shape[0]):
        x1 = x[i,0] 
        x2 = x[i,1] 
        x3 = x[i,2] 
        x4 = x[i,3]
            
        epp, dp, dm, ph = (
            m_electron + x1 * (w - 2 * m_electron),
            w / (2 * m_electron) * (x2 + x3),
            w / (2 * m_electron) * (x2 - x3),
            x4 * 2 * pi,
        )

        # Positron energy
        epm = w - epp

        if (m_electron < epm) and (m_electron < epp) and (epm < w) and (epp < w) and (dm > 0.0) and (dp > 0.0):
            qsq_over_m_electron_sq = (
                dp**2 + dm**2 + 2.0 * dp * dm * cos(ph)
            ) + m_electron**2 * (
                (1.0 + dp**2) / (2.0 * epp) + (1.0 + dm**2) / (2.0 * epm)
            ) ** 2

            PF = (
                8.0
                / pi
                * alpha_em
                * (alpha_em / m_electron) ** 2
                * epp
                * epm
                / (w**3 * qsq_over_m_electron_sq**2)
                * dp
                * dm
            )

            jacobian_factor = pi * w**2 * (w - 2 * m_electron) / m_electron**2
            FF = g2_elastic_c(Z, m_electron**2 * qsq_over_m_electron_sq)

            T1 = -1.0 * dp**2 / (1.0 + dp**2) ** 2
            T2 = -1.0 * dm**2 / (1.0 + dm**2) ** 2
            T3 = (
                w**2 / (2.0 * epp * epm) * (dp**2 + dm**2) / ((1.0 + dp**2) * (1.0 + dm**2))
            )
            T4 = (
                (epp / epm + epm / epp)
                * (dp * dm * cos(ph))
                / ((1.0 + dp**2) * (1.0 + dm**2))
            )
            ans[i] = PF * (T1 + T2 + T3 + T4) * jacobian_factor * FF
    
    return ans 


def aa_c(int Z):
    """Support function for atomic form factors"""
    return 184.15 * (2.718) ** -0.5 * Z ** (-1.0 / 3.0) / m_electron

def g2_elastic_c(int Z, double t):
    """Elastic form factor"""
    cdef double a0 = aa_c(Z)
    return Z**2 * a0**4 * t**2 / (1 + a0**2 * t) ** 2
