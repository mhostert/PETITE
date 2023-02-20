import numpy as np

from scipy.interpolate import interp1d

from .moliere import get_scattered_momentum 
from .particle import Particle
from .kinematics import eegFourVecs, eeVFourVecs, gepemFourVecs, Compton_FVs, Ann_FVs
from .shower import Shower

import sys

me = 0.000511
        

Z = {'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'graphite':12.0, 'lead':207.2} #atomic mass of different targets

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

MVLib = {'10MeV':0.010,'100MeV':0.100}

class DarkShower(Shower):
    def __init__(self, PickDir, TargetMaterial, MinEnergy, MVStr):
        super().__init__(PickDir, TargetMaterial, MinEnergy)

        self.set_MV(MVStr)
        self.set_DarkPickDir(PickDir)
        self.set_DarkSampDir(PickDir + TargetMaterial + "/DarkV/")
        self.set_DarkSampDirE(PickDir + "electrons/DarkV/")

        self.set_Darksamples()
        self.set_DarkCrossSections()
        self.set_DarkNSigmas()

    def set_DarkPickDir(self, value):
        self._DarkPickDir = value
    def get_DarkPickDir(self):
        return self._DarkPickDir
    def set_DarkSampDir(self, value):
        self._DarkSampDir = value
    def get_DarkSampDir(self):
        return self._DarkSampDir
    def set_DarkSampDirE(self, value):
        self._DarkSampDirE = value
    def get_DarkSampDirE(self):
        return self._DarkSampDirE
    def set_MV(self, value):
        self._MVStr = value
        self._MV = MVLib[value]
    def get_MVStr(self):
        return self._MVStr
    def get_MV(self):
        return self._MV

    def set_Darksamples(self):
        self._DarkBremSamples = np.load(self.get_DarkSampDir()+"DarkBremEvts_" + self.get_MVStr() + ".npy", allow_pickle=True)
        self._DarkAnnSamples = np.load(self.get_DarkSampDirE()+"AnnihilationEvts_" + self.get_MVStr() + ".npy", allow_pickle=True)
        self._DarkCompSamples = np.load(self.get_DarkSampDirE()+"ComptonEvts_" + self.get_MVStr() + ".npy", allow_pickle=True)
    def get_DarkBremSamples(self, ind):
        return self._DarkBremSamples[ind]
    def get_DarkAnnSamples(self, ind):
        return self._DarkAnnSamples[ind]
    def get_DarkCompSamples(self, ind):
        return self._DarkCompSamples[ind]
    
    def set_DarkCrossSections(self):
        self._DarkBremXSec = np.load(self.get_DarkSampDir()+"DarkBremXSec_"+self.get_MVStr()+".npy",allow_pickle=True)
        self._DarkAnnXSec = np.load(self.get_DarkSampDirE()+"AnnihilationXSec_"+self.get_MVStr()+".npy",allow_pickle=True)
        self._DarkCompXSec = np.load(self.get_DarkSampDirE()+"ComptonXSec_"+self.get_MVStr()+".npy",allow_pickle=True)

        self._EeVecDarkBrem = np.transpose(self._DarkBremXSec)[0]
        self._EeVecDarkAnn = np.transpose(self._DarkAnnXSec)[0]
        self._EgVecDarkComp = np.transpose(self._DarkCompXSec)[0]

        self._logEeMinDarkBrem, self._logEeSSDarkBrem = np.log10(self._EeVecDarkBrem[0]), np.log10(self._EeVecDarkBrem[1]) - np.log10(self._EeVecDarkBrem[0])
        self._logEeMinDarkAnn, self._logEeSSDarkAnn = np.log10(self._EeVecDarkAnn[0]), np.log10(self._EeVecDarkAnn[1]) - np.log10(self._EeVecDarkAnn[0])
        self._logEgMinDarkComp, self._logEgSSDarkComp = np.log10(self._EgVecDarkComp[0]), np.log10(self._EgVecDarkComp[1]) - np.log10(self._EgVecDarkComp[0])

    def get_DarkBremXSec(self):
        return self._DarkBremXSec
    def get_DarkAnnXSec(self):
        return self._DarkAnnXSec
    def get_DarkCompXSec(self):
        return self._DarkCompXSec

    def set_DarkNSigmas(self):
        DBS, DAnnS, DCS = self.get_DarkBremXSec(), self.get_DarkAnnXSec(), self.get_DarkCompXSec()
        nZ, ne = self.get_nTargets()
        self._NSigmaDarkBrem = interp1d(np.transpose(DBS)[0], nZ*GeVsqcm2*np.transpose(DBS)[1])
        self._NSigmaDarkAnn = interp1d(np.transpose(DAnnS)[0], ne*GeVsqcm2*np.transpose(DAnnS)[1])
        self._NSigmaDarkComp = interp1d(np.transpose(DCS)[0], ne*GeVsqcm2*np.transpose(DCS)[1])

    def GetBSMWeights(self, PID, Energy):
        if PID == 22:
            if np.log10(Energy) < self._logEgMinDarkComp:
                return 0.0
            else:
                return self._NSigmaDarkComp(Energy)/(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))
        elif PID == 11:
            if np.log10(Energy) < self._logEeMinDarkBrem:
                return 0.0
            else:
                return self._NSigmaDarkBrem(Energy)/self._NSigmaBrem(Energy)
        elif PID == -11:
            if np.log10(Energy) < self._logEeMinDarkBrem:
                BremPiece = 0.0
            else:
                BremPiece = self._NSigmaDarkBrem(Energy)
            if np.log10(Energy) < self._logEeMinDarkAnn:
                AnnPiece = 0.0
            else:
                AnnPiece = self._NSigmaDarkAnn(Energy)
            return (BremPiece + AnnPiece)/(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy))

    def GetPositronDarkBF(self, Energy):
        if np.log10(Energy) < self._logEeMinDarkAnn:
            return 1.0
        else:
            return self._NSigmaDarkBrem(Energy)/(self._NSigmaDarkBrem(Energy) + self._NSigmaDarkAnn(Energy))

    def DarkElecBremSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinDarkBrem)/self._logEeSSDarkBrem)
        ts = self.get_DarkBremSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecDarkBrem[LUKey]

        ct = np.cos(self.get_MV()/(SampEvt[0]*Ee0/EeMod)*np.sqrt((EeMod-SampEvt[0])/EeMod)*SampEvt[1])
        ctp =np.cos(self.get_MV()/(SampEvt[0]*Ee0/EeMod)*np.sqrt(EeMod/(EeMod-SampEvt[0]))*SampEvt[2])
        NFVs = eeVFourVecs(Ee0, me, SampEvt[0]*Ee0/EeMod, self.get_MV(), ct, ctp, SampEvt[3])

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs[2]
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)

        if EVf > Ee0:
            print("---------------------------------------------")
            print("High Energy V Found from Electron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EeMod)
            print("---------------------------------------------")

        wg = self.GetBSMWeights(11, Ee0)

        GenType = 0

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def DarkAnnihilationSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinDarkAnn)/self._logEeSSDarkAnn)
        EeMod = self._EeVecDarkAnn[LUKey]
        ts = self.get_DarkAnnSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        #NFVs = Ann_FVs(EeMod, meT, MVT, SampEvt[0])[1]
        NFVs = Ann_FVs(Ee0, me, self.get_MV(), SampEvt[0])[1]
        GenType = 2

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)
        wg = self.GetBSMWeights(-11, Ee0)

        if EVf > Ee0:
            print("---------------------------------------------")
            print("High Energy V Found from Positron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EeMod)
            print(wg)
            print("---------------------------------------------")

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def DarkComptonSample(self, Phot0):
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinDarkComp)/self._logEgSSDarkComp)
        EgMod = self._EgVecDarkComp[LUKey]
        ts = self.get_DarkCompSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        #NFVs = Compton_FVs(EgMod, meT, MVT, SampEvt[0])[1]
        NFVs = Compton_FVs(Eg0, me, self.get_MV(), SampEvt[0])[1]

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)

        wg = self.GetBSMWeights(22, Eg0)
        GenType = 3
        if EVf > Eg0:
            print("---------------------------------------------")
            print("High Energy V Found from Photon Samples:")
            print(Phot0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EgMod)
            print(wg)
            print("---------------------------------------------")

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*(Phot0.get_IDs()[1])+0, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def GenDarkShower(self, ExDir=None, SParams=None):
        if ExDir is None and SParams is None:
            print("Need an existing SM shower-file directory or SM shower parameters to run dark shower")
            return None
        
        if ExDir is not None:
            ShowerToSamp = np.load(ExDir, allow_pickle=True)
        else:
            PID0, p40, ParPID = SParams
            ShowerToSamp = self.GenShower(PID0, p40, ParPID)
        
        NewShower = []
        for ap in ShowerToSamp:
            if ap.get_IDs()[0] == 11:
                if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem:
                    continue
                npart = self.DarkElecBremSample(ap)
            elif ap.get_IDs()[0] == -11:
                if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem:
                    continue
                DarkBFEpBrem = self.GetPositronDarkBF(ap.get_pf()[0])
                ch = np.random.uniform(low=0., high=1.0)
                if ch < DarkBFEpBrem:
                    npart = self.DarkElecBremSample(ap)
                else:
                    npart = self.DarkAnnihilationSample(ap)
            elif ap.get_IDs()[0] == 22:
                if np.log10(ap.get_pf()[0]) < self._logEgMinDarkComp:
                    continue
                npart = self.DarkComptonSample(ap)
            NewShower.append(npart)

        return ShowerToSamp, NewShower