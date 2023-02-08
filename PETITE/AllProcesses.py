import numpy as np
import vegas as vg
import functools
import random as rnd

#--------------------------------------------------------------------
#Differential Cross Sections for Incident Electrons/Positrons/Photons
#--------------------------------------------------------------------

def Unity(Z, me, t):
    return(1.0)

def dummy(x,y,z):
    return(0)

def PPQSq(xx, me, w):
    """Computes momentum transfer squared for photon-scattering pair production
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables 
            epsilon_plus, delta_plus, delta_minus, phi (see ... for definitions) 
        me: electron mass 
        w:  frequency of emitted virtual photon, equal to sum of energies of outgoing e+- pair
    Returns:
        nuclear momentum transfer squared
    """
    epp, dp, dm, ph = xx
    epm = w - epp
    return me**2*((dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + me**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2)
 

def BremQSq(xx, me, ep):
    """Momentum Transfer Squared for electron/positron bremsstrahlung
    Args:
        w: frequency of radiated photon 
        d: rescaled emission angle, equal to theta * epsilon/m, 
           where epsilon energy of incoming lepton
        dp: rescaled emission angle, equal to theta * epsilon'/m, 
           where epsilon' energy of outgoing lepton
        ph: angle phi
        me: electron mass
        ep: epsilon', e+/e- energy after radiation
    Returns:
        nuclear momentum transfer squared
    """
    w, d, dp, ph = xx
    epp = ep - w
    return me**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + me**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)


def dSBrem_dP(EvtInfo, varthV):
    """Standard Model Bremsstrahlung
       e (ep) + Z -> e (epp) + gamma (w) + Z
       Outgoing kinematics given by w, ct (cos(theta)), ctp (cos(theta')), and ph (phi)

       Input parameters needed:
            ep (incident electron energy)
            me (electron mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    ep=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=0

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])

    dSigs = []
    for varth in varthV:
        w, ct, ctp, ph = varth

        epp = ep - w
        p, pp = np.sqrt(ep**2 - me**2), np.sqrt(epp**2 - me**2)
        k, kp = ep - p*ct, epp - pp*ctp

        qsq = p**2 + pp**2 + w**2 - 2*p*w*ct + 2*pp*w*ctp - 2*p*pp*(ct*ctp + np.sqrt((1-ct**2)*(1-ctp**2))*np.cos(ph))

        PF = Z**2*al*(al/me)**2/(2.0*np.pi)*(pp*me**2/(w*p*qsq**2))
        T1 = (pp/kp)**2*(4*ep**2 - qsq)*(1-ctp**2)
        T2 = 2.0*w**2/(k*kp)*(p**2*(1-ct**2) + pp**2*(1-ctp**2))
        T3 = (p/k)**2*(4*epp**2 - qsq)*(1-ct**2)
        T4 = -2*p*pp/(k*kp)*(2*ep**2 + 2*epp**2 - qsq)*np.sqrt((1-ct**2)*(1-ctp**2))*np.cos(ph)

        dSig0 = PF*(T1+T2+T3+T4)
        
        if dSig0 < 0.0 or np.isnan(dSig0):
            print([dSig0, PF, T1, T2, T3, T4, qsq])

        dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSBrem_dP_T(EvtInfo, varthTildeV):
    """Standard Model Bremsstrahlung in the Small-Angle Approximation
       e (ep) + Z -> e (epp) + gamma (w) + Z
       Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)

       Input parameters needed:
            ep (incident electron energy)
            me (electron mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    ep=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=0


    if len(np.shape(varthTildeV)) == 1:
        varthTildeV = np.array([varthTildeV])
    dSigs = []
    for varthTilde in varthTildeV:
        w, d, dp, ph = varthTilde

        epp = ep - w

        qsqT = me**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + me**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)
        PF = 8.0/np.pi*Z**2*al*(al/me)**2*(epp*me**4)/(w*ep*qsqT**2)*d*dp

        T1 = d**2/(1 + d**2)**2
        T2 = dp**2/(1 + dp**2)**2
        T3 = w**2/(2*ep*epp)*(d**2 + dp**2)/((1 + d**2)*(1 + dp**2))
        T4 = -(epp/ep + ep/epp)*(d*dp*np.cos(ph))/((1 + d**2)*(1 + dp**2))
        dSig0 = PF*(T1+T2+T3+T4)

        if dSig0 < 0.0 or np.isnan(dSig0):
            print([dSig0, PF, T1, T2, T3, T4, qsqT])
        dSigs.append(dSig0)

    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSDBrem_dP(EvtInfo, varthV):
    """Dark Vector Bremsstrahlung
       e (ep) + Z -> e (epp) + V (w) + Z
       Outgoing kinematics given by w, ct (cos(theta)), ctp (cos(theta')), and ph (phi)

       Input parameters needed:
            ep (incident electron energy)
            me (electron mass)
            MV (Dark Vector Mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    ep=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=EvtInfo['m_V']

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])

    dSigs = []
    for varth in varthV:
        w, ct, ctp, ph = varth

        epp = ep - w
        p, pp = np.sqrt(ep**2 - me**2), np.sqrt(epp**2 - me**2)
        k = np.sqrt(w**2 - MV**2)
        K = ep*w - p*k*ct - 0.5*MV**2
        KP = epp*w - pp*k*ctp + 0.5*MV**2

        qsq = k**2 + p**2 + pp**2 + 2*k*pp*ctp - 2*k*p*ct - 2*p*pp*(ct*ctp + np.sqrt((1-ct**2)*(1-ctp**2))*np.cos(ph))

        dSig0 = (al*k*me**2*pp*(al/me)**2*(2*me**2*((K - KP)**2*qsq - 4*(ep*K + KP*(-ep + w))*(-(epp*KP) + K*(epp + w))) + MV**2*((K + KP)**2*qsq - 4*(ep*K - epp*KP)*(KP*(-ep + w) + K*(epp + w))) - 2*K*KP*(qsq**2 - 2*qsq*(2*ep*epp + K - KP + ep*w - epp*w) + 2*(K**2 + KP*(KP + 2*ep*(ep - epp - w)) - 2*epp*K*(-ep + epp + w))))*Z**2)/(8*K**2*KP**2*p*np.pi**2*qsq**2)

        if dSig0 < 0.0 or np.isnan(dSig0):
            print(dSig0, varth, epp, K, KP, qsq)
        dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSDBrem_dP_T(EvtInfo, varthV):
    """Dark Vector Bremsstrahlung in the Small-Angle Approximation
       e (ep) + Z -> e (epp) + V (w) + Z
       Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)

       Input parameters needed:
            ep (incident electron energy)
            me (electron mass)
            MV (Dark Vector Mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    ep=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=EvtInfo['m_V']

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])
    dSigs = []
    for varth in varthV:
        w, d, dp, ph = varth

        epp = ep - w

        xsq = (d**2 + dp**2 - 2.0*d*dp*np.cos(ph)) + mV**2/(4.0*ep*epp)*(1 + 0.5*(d**2 + dp**2))**2
        PF = 4.0*al**3*Z**2/(np.pi*mV**2)*w**3/ep**3*1/(xsq**2*epp)*(d*dp)/((1+d**2)*(1+dp**2))
        T1 = (ep**2+epp**2)/w**2*xsq
        T2 = -(d**2 - dp**2)**2/((1+d**2)*(1+dp**2))
        T3 = -mV**2/(4.0*w**2)*(ep/epp*(1+dp**2) + epp/ep*(1+d**2))
        dSig0 = PF*(T1+T2+T3)
        if dSig0 < 0.0 or np.isnan(dSig0):
            print(dSig0, varth, ep, epp, PF, T1, T2, T3)
        dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dAnn_dCT(EvtInfo, varthV):
    """Annihilation of a Positron and Electron into a Photon and a (Dark) Photon
       e+ (Ee) + e- (me) -> gamma + gamma/V

       Input parameters needed:
            Ee (incident positron energy)
            me (electron mass)
            mV (Dark Vector Mass -- can be set to zero for SM Case)
            al (electro-weak fine-structure constant)
    """
    Ee=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=EvtInfo['m_V']

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])
    dSigs = []
    for varth in varthV:
        ct = varth[0]

        if Ee < (mV**2-2*me**2)/(2*me):
            return 0.0

        s = 2.0*me*(Ee + me)
        b = np.sqrt(1.0 - 4.0*me**2/s)
        dSigs.append(4.0*np.pi*al**2/(s*(1 - b**2*ct**2))*((s-mV**2)/(2*s)*(1+ct**2) + 2.0*mV**2/(s-mV**2)))
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSPairProd_dP(EvtInfo, varthTildeV):
    """Standard Model Pair Production
       gamma (w) + Z -> e+ (epp) + e- (epm) + Z
       Outgoing kinematics given by epp, ctp (cos(theta_+)), ctm (cos(theta_-)), and ph (phi)

       Input parameters needed:
            w (incident photon energy)
            me (electron mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """

    w=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=0 # This function does not handle dark photon s
    
    if len(np.shape(varthTildeV)) == 1:
        varthTildeV = np.array([varthTildeV])
    dSigs = []
    for varthTilde in varthTildeV:
        epp, ctp, ctm, ph = varthTilde

        epm = w - epp
        pp, pm = np.sqrt(epp**2 - me**2), np.sqrt(epm**2 - me**2)
        kp, km = epp - pp*ctp, epm - pm*ctm

        qsq = (w**2 + pm**2 + pp**2 - 2*w*pm*ctm - 2*w*pp*ctp + 2*pm*pp*(ctp*ctm + np.sqrt((1-ctm**2)*(1-ctp**2))*np.cos(ph)))
        PF = Z**2*al*(al/me)**2/(2.0*np.pi)*(pp*pm*me**2)/(w**3*qsq**2)
        T1 = (pp/kp)**2*(4*epm**2-qsq)*(1-ctp**2)
        T2 = -2.0*w**2/(kp*km)*(pp**2*(1-ctp**2) + pm**2*(1-ctm**2))
        T3 = (pm/km)**2*(4*epp**2-qsq)*(1-ctm**2)
        T4 = -2.0*pp*pm/(kp*km)*(2*epp**2 + 2*epm**2 - qsq)*np.sqrt((1-ctp**2)*(1-ctm**2))*np.cos(ph)

        dSig0 = PF*(T1+T2+T3+T4)

        if dSig0 < 0.0 or np.isnan(dSig0):
            print([dSig0, PF, T1, T2, T3, T4, qsq])
        dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def aa(Z, me):
    return 184.15*(2.718)**-0.5*Z**(-1./3.)/me
def aap(Z, me):
    return 1194*(2.718)**-0.5*Z**(-2./3.)/me

def G2el(Z, me, t):
    a0 = aa(Z, me)
    return Z**2*a0**4*t**2/(1 + a0**2*t)**2

def G2inel(Z, me, t):
    ap0 = aap(Z, me)
    return Z*ap0**4*t**2/(1 + ap0**2*t)**2

def dSPairProd_dP_T(EvtInfo, varthTildeV):
    """Standard Model Pair Production in the Small-Angle Approximation
       gamma (w) + Z -> e+ (epp) + e- (epm) + Z
       Outgoing kinematics given by epp, dp (delta+), dm (delta-), and ph (phi)

       Input parameters needed:
            w (incident photon energy)
            me (electron mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    w=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=0
    
    if len(np.shape(varthTildeV)) == 1:
        varthTildeV = np.array([varthTildeV])
    dSigs = []
    for varthTilde in varthTildeV:
        epp, dp, dm, ph = varthTilde

        epm = w - epp

        #qsqT = me**2*(dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph) + me**2*((1.0 + dp**2)/(2.0*epp) + (1.0 + dm**2)/(2.0*epm))**2)
        #PF = 8.0/np.pi*Z**2*al*(al/me)**2*me**4*epp*epm/(w**3*qsqT**2)*dp*dm
        qsqT = (dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + me**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2

        #FF0 = G2el(Z, me, me**2*qsqT)# + G2inel(Z, me, me**2*qsqT)
        FF0 = Z**2

        #PF = 8.0/np.pi*Z**2*al*(al/me)**2*epp*epm/(w**3*qsqT**2)*dp*dm
        PF = 8.0/np.pi*FF0*al*(al/me)**2*epp*epm/(w**3*qsqT**2)*dp*dm

        T1 = -1.0*dp**2/(1.0 + dp**2)**2
        T2 = -1.0*dm**2/(1.0 + dm**2)**2
        T3 = w**2/(2.0*epp*epm)*(dp**2 + dm**2)/((1.0 + dp**2)*(1.0 + dm**2))
        T4 = (epp/epm + epm/epp)*(dp*dm*np.cos(ph))/((1.0 + dp**2)*(1.0+dm**2))

        dSig0 = PF*(T1+T2+T3+T4)

        if dSig0 < 0.0 or np.isnan(dSig0):
            print([dSig0, PF, T1, T2, T3, T4, qsqT])
        dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSCompton_dCT(EvtInfo, varthV):
    """Compton Scattering of a Photon off an at-rest Electron, producing either a photon or a Dark Vector
        gamma (Eg) + e- (me) -> e- + gamma/V

       Input parameters needed:
            Eg (incident photon energy)
            me (electron mass)
            MV (Dark Vector Mass -- can be set to zero for SM Case)
            al (electro-weak fine-structure constant)
    """
    Eg=EvtInfo['E_inc']
    me=EvtInfo['m_e']
    Z =EvtInfo['Z_T']
    al=EvtInfo['alpha_FS']
    mV=EvtInfo['m_V']

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])
    dSigs = []
    for varth in varthV:
        ct = varth[0]

        s = me**2 + 2*Eg*me
        dSigs.append((al**2*np.pi*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))*((1 + 3*ct**2)*me**10 - me**8*((3 + 9*ct**2)*mV**2 + (7 + 17*ct**2)*s + ct*(3 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + 
   me**6*((3 + 9*ct**2)*mV**4 + 2*mV**2*(2*(4 + 7*ct**2)*s + ct*(3 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + 2*s*((-31 + 3*ct**2)*s + 2*ct*(4 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s)))) + 
   s**2*((-5 + ct**2)*mV**6 + mV**4*((11 + ct**2)*s - ct*(3 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + mV**2*s*(-((11 + 5*ct**2)*s) + 2*ct*(3 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + 
     s**2*((5 + 3*ct**2)*s - ct*(7 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s)))) + me**2*s*((2 + 6*ct**2)*mV**6 + s**2*((13 - 25*ct**2)*s + 4*ct*(2 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + 
     mV**4*((5 - 33*ct**2)*s + 2*ct*(3 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + 2*mV**2*s*(2*(-16 + 9*ct**2)*s - ct*(11 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s)))) - 
   me**4*((1 + 3*ct**2)*mV**6 + mV**4*((11 + 17*ct**2)*s + ct*(3 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + 
     2*mV**2*s*(-((31 + 23*ct**2)*s) + ct*(11 + ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))) + 2*s**2*((71 - 15*ct**2)*s + ct*(-41 + 3*ct**2)*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s))))))/(2*s**2*(-me**2 + s)**3*(me**2 - mV**2 + s - ct*np.sqrt(me**4 + (mV**2 - s)**2 - 2*me**2*(mV**2 + s)))**2))

    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

#Function for drawing unweighted events from a weighted distribution
def GetPts(Dist, npts):
    """If weights are too cumbersome, this function returns a properly-weighted sample from Dist"""
    ret = []
    MW = np.max(np.transpose(Dist)[-1])

    tochoosefrom = [pis for pis in range(len(Dist))]
    choicesgetter = rnd.choices(tochoosefrom, np.transpose(Dist)[-1], k=npts)
    for cg in choicesgetter:
        ret.append(Dist[cg][0:-1])

    return ret


#------------------------------------------------------------------------------
#Total Cross Sections and Sample Draws for Incident Electrons/Positrons/Photons
#------------------------------------------------------------------------------
NPts = 10000 #Default number of points to draw for unweighted samples

def Brem_S(EI, Egmin, VB=False, mode='XSec'):
    ep, me, Z, al = EI
    igrange = [[Egmin, ep-me], [-0.99999, 0.99999], [-0.99999, 0.99999], [0, 2*np.pi]]
    integrand = vg.Integrator(igrange)

    if mode == 'XSec':
        resu = integrand(functools.partial(dSBrem_dP, EI), nitn=100, nstrat=[10,15,15,6])
        if VB:
            print(resu.summary())
        tr = resu.mean

    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dSBrem_dP, EI), nitn=30, nstrat=[6,30,30,6])
        result = integrand(functools.partial(dSBrem_dP, EI), nitn=30, nstrat=[6,30,30,6])

        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt*dSBrem_dP(EI, x)
        NSamp = 10
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                M0 = wgt*integral*dSBrem_dP(EI, x)
                pts.append(np.concatenate([x, [M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts, NPts)], dtype=object)

    return tr

def Brem_S_T(EI, Egmin, VB=False, mode='XSec'):
    ep, me, Z, al = EI
    igrange = [[Egmin, ep-me], [0.0, np.sqrt(ep/me)], [0.0, np.sqrt(ep/me)], [0.0, 2.0*np.pi]]
    integrand = vg.Integrator(igrange)

    if mode == 'Pickle':
        print("Integrator set up")
        print(EI)
        integrand(functools.partial(dSBrem_dP_T, EI), nitn=500, nstrat=[15, 25, 25, 15])
        print("Burn-in complete")
        print(EI)
        result = integrand(functools.partial(dSBrem_dP_T, EI), nitn=500, nstrat=[15, 25, 25, 15])
        print("Fully Integrated")
        print(EI, result.mean)
        return integrand
    elif mode == 'XSec':
        resu = integrand(functools.partial(dSBrem_dP_T, EI), nitn=100, nstrat=[10, 30, 30, 10])
        if VB:
            print(resu.summary())
        tr = resu.mean
    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dSBrem_dP_T, EI), nitn=50, nstrat=[12, 16, 16, 12])
        result = integrand(functools.partial(dSBrem_dP_T, EI), nitn=50, nstrat=[12, 16, 16, 12])

        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(dSBrem_dP_T(EI, x))
        if VB:
            print(integral)
        NSamp = 1
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                #M0 = wgt*integral*dSBrem_dP_T(EI, x)
                M0 = wgt*dSBrem_dP_T(EI, x)
                pts.append(np.concatenate([x, [M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts, NPts)], dtype=object)

    return tr

def DBrem_S(EI, VB=False, mode='XSec'):
    ep, me, mV, Z, al = EI
    igrange = [[mV, ep-me], [-1, 1], [-1, 1], [0, 2*np.pi]]
    integrand = vg.Integrator(igrange)

    if mode == 'XSec':
        resu = integrand(functools.partial(dSDBrem_dP, EI), nitn=30, nstrat=[6,30,30,6])
        if VB:
            print(resu.summary())
        tr = resu.mean
    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dSDBrem_dP, EI), nitn=30, nstrat=[6, 30, 30, 6])
        result = integrand(functools.partial(dSDBrem_dP, EI), nitn=30, nstrat=[6,30,30,6])
        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(dSDBrem_dP(EI, x))
        NSamp = 10
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                M0 = wgt*integral*dSDBrem_dP(EI, x)
                pts.append(np.concatenate([x, [M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts, NPts)], dtype=object)            
    return tr

def DBrem_S_T(EI, VB=False, mode='XSec'):
    ep, me, mV, Z, al = EI
    igrange = [[mV, ep-me], [0, np.sqrt(ep/mV)], [0, np.sqrt(ep/mV)], [0, 2*np.pi]]
    integrand = vg.Integrator(igrange)

    if mode == 'Pickle':
        print("Integrator set up")
        print(EI)
        integrand(functools.partial(dSDBrem_dP_T, EI), nitn=100, nstrat=[10, 15, 15, 10])
        print("Burn-in complete")
        print(EI)
        result = integrand(functools.partial(dSDBrem_dP_T, EI), nitn=100, nstrat=[10, 15, 15, 10])
        print("Fully Integrated")
        print(EI, result.mean)

        return integrand
    if mode == 'XSec':
        resu = integrand(functools.partial(dSDBrem_dP_T, EI), nitn=50, nstrat=[4,20,20,4])
        if VB:
            print(resu.summary())
        tr = resu.mean

    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dSDBrem_dP_T, EI), nitn=50, nstrat=[12, 16, 16, 12])
        result = integrand(functools.partial(dSDBrem_dP_T, EI), nitn=50, nstrat=[12, 16, 16, 12])
        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(dSDBrem_dP_T(EI,x))
        NSamp = 1
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                #M0 = wgt*integral*dSDBrem_dP_T(EI,x)
                M0 = wgt*dSDBrem_dP_T(EI,x)
                pts.append(np.concatenate([x,[M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts, NPts)], dtype=object)
    return tr

def Ann_S(EI, Egmin, VB=False, mode='XSec'):
    Ee, me, mV, al = EI
    EVMin = Egmin + mV

    #Determine ranges of cos(theta) that give photons/dark photons with sufficient energy
    ctmaxV = np.sqrt(2.0)*(2.0*me*(Ee-EVMin)*(Ee+me)+Ee*mV**2)/(np.sqrt((Ee-me)*(2.0*Ee+me))*(2*me*(Ee+me)-mV**2))
    ctMaxmV = np.sqrt(2.0)*(Ee*mV**2 - 2.0*me*(Ee - Egmin)*(Ee+me))/(np.sqrt((Ee-me)*(2*Ee+me))*(2*me*(Ee+me)-mV**2))
    if ctmaxV < 0.0 or ctMaxmV > 0.0:
        ctMax = 0.0
    else:
        ctMax = np.min([np.abs(ctmaxV), np.abs(ctMaxmV)])
    igrange = [[-ctMax, ctMax]]
    if ctMax == 0.0:
        if mode == 'XSec':
            return 0.0
        elif mode == 'Pickle':
            return None
        else:
            return np.array([0.0, [0.0 for k in range(NPts)]], dtype=object)
    integrand = vg.Integrator(igrange)

    if mode == 'Pickle':
        integrand(functools.partial(dAnn_dCT, EI), nitn=5000, nstrat=[300])
        result = integrand(functools.partial(dAnn_dCT, EI), nitn=5000, nstrat=[300])
        return integrand

    if mode == 'XSec':
        resu = integrand(functools.partial(dAnn_dCT, EI), nitn=100)
        if VB:
            print(resu.summary())
        tr = resu.mean
    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dAnn_dCT, EI), nitn=100)
        result = integrand(functools.partial(dAnn_dCT, EI), nitn=100)
        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(dAnn_dCT(EI,x))
        NSamp = 10
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                M0 = wgt*integral*dAnn_dCT(EI,x)
                pts.append(np.concatenate([x,[M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts,NPts)], dtype=object)
    return tr

def PairProd_S(EI, VB=False, mode='XSec'):
    w, me, Z, al = EI
    igrange = [[me, w-me], [-1, 1], [-1, 1], [0.0, 2.0*np.pi]]
    integrand = vg.Integrator(igrange)

    if mode == 'XSec':
        resu = integrand(functools.partial(dSPairProd_dP, EI), nitn=100, nstrat=[6, 24, 24, 6])
        if VB:
            print(resu.summary())
        tr = resu.mean

    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dSPairProd_dP, EI), nitn=30, nstrat=[6, 24, 24, 6])    
        result = integrand(functools.partial(dSPairProd_dP, EI), nitn=30, nstrat=[6, 24, 24, 6])    

        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(dSPairProd_dP(EI,x))
        NSamp = 10
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                M0 = wgt*integral*dSPairProd_dP(EI,x)
                pts.append(np.concatenate([x,[M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts,NPts)], dtype=object)

    return tr

def PairProd_S_T(EI, VB=False, mode='XSec'):
    w, me, Z, al = EI
    igrange = [[me, w-me], [0.0, np.sqrt(w/me)], [0.0, np.sqrt(w/me)], [0.0, 2.0*np.pi]]
    integrand = vg.Integrator(igrange)

    if mode == 'Pickle':
        integrand(functools.partial(dSPairProd_dP_T, EI), nitn=300, nstrat=[15, 25, 25, 15])
        result = integrand(functools.partial(dSPairProd_dP_T, EI), nitn=300, nstrat=[15, 25, 25, 15])
        return integrand
    if mode == 'XSec':
        resu = integrand(functools.partial(dSPairProd_dP_T, EI), nitn=100, nstrat=[6, 24, 24, 6])
        if VB:
            print(resu.summary())
        tr = resu.mean

    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dSPairProd_dP_T, EI), nitn=30, nstrat=[12, 16, 16, 12])    
        result = integrand(functools.partial(dSPairProd_dP_T, EI), nitn=100, nstrat=[12, 16, 16, 12])    

        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(dSPairProd_dP_T(EI,x))
        NSamp = 1
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                #M0 = wgt*integral*dSPairProd_dP_T(EI,x)
                M0 = wgt*dSPairProd_dP_T(EI,x)
                pts.append(np.concatenate([x,[M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts,NPts)], dtype=object)

    return tr

def Compton_S(EI, VB=False, mode='XSec'):
    Eg, me, mV, al = EI
    igrange = [[-1.0, 1.0]]
    integrand = vg.Integrator(igrange)

    if Eg <= mV*(1.0 + mV/(2*me)):
        if mode == 'Pickle':
            return None
        elif mode == 'XSec':
            return 0.0
        else:
            return np.array([0.0, [0.0 for k in range(NPts)]], dtype=object)

    if mode == 'Pickle':
        integrand(functools.partial(dSCompton_dCT, EI), nitn=5000, nstrat=[300])
        result = integrand(functools.partial(dSCompton_dCT, EI), nitn=5000, nstrat=[300])
        return integrand
    elif mode == 'XSec':
        resu = integrand(functools.partial(dSCompton_dCT, EI), nitn=100)
        if VB:
            print(resu.summary())
        tr = resu.mean
    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(dSCompton_dCT, EI), nitn=30)
        result = integrand(functools.partial(dSCompton_dCT, EI), nitn=30)
        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(dSCompton_dCT(EI,x))
        NSamp = 10
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                M0 = wgt*integral*dSCompton_dCT(EI,x)
                pts.append(np.concatenate([x,[M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts,NPts)], dtype=object)

    return tr
