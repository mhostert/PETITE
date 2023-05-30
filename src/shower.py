import numpy as np
import pickle 

from scipy.interpolate import interp1d

from .moliere import get_scattered_momentum 
from .particle import Particle
from .kinematics import e_to_egamma_fourvecs, e_to_eV_fourvecs, gamma_to_epem_fourvecs, compton_fourvecs, annihilation_fourvecs, ee_to_ee_fourvecs
from .all_processes import *
from datetime import datetime

np.random.seed(int(datetime.now().timestamp()))

import sys
from numpy.random import random as draw_U


Z = {'hydrogen':1.0, 'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'hydrogen':1.0, 'graphite':12.0, 'lead':207.2} #atomic mass of different targets
rho = {'hydrogen':1.0, 'graphite':2.210, 'lead':11.35} #g/cm^3
dEdx = {'hydrogen':2.0*rho['hydrogen'], 'graphite':2.0*rho['graphite'], 'lead':2.0*rho['lead']} #MeV per cm

# GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
GeVsqcm2 = hbarc**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = m_proton_grams # proton mass in grams

process_code = {'Brem':0, 'Ann': 1, 'PairProd': 2, 'Comp': 3, "Moller":4, "Bhabha":5}
#Egamma_min = 0.001
#FIXME incorporate the minimum brem-photon energy used in training into stored, dictionary information somewhere

class Shower:
    """ Representation of a shower

    """
    def __init__(self, dict_dir, target_material, min_energy, maxF_fudge_global=1,max_n_integrators=int(1e4)):
        """Initializes the shower object.
        Args:
            dict_dir: directory containing the pre-computed VEGAS integrators and auxillary info.
            target_material: string label of the homogeneous material through which 
            particles propagate (available materials are the dict keys of 
            Z, A, rho, etc)
            min_Energy: minimum particle energy in GeV at which the particle 
            finishes its propagation through the target
        """

        
        ## Need to swap this out for integrator objects
        self.set_dict_dir(dict_dir)
        self.set_target_material(target_material)
        self.min_energy = min_energy

        self.set_material_properties()
        self.set_n_targets()
        self.set_cross_sections()
        self.set_NSigmas()
        self.set_samples()

        self._maxF_fudge_global=maxF_fudge_global
        self._max_n_integrators=max_n_integrators

                
        
    def load_sample(self, dict_dir, process):
        sample_file=open(dict_dir + "samp_Dicts.pkl", 'rb')
        sample_dict=pickle.load(sample_file)
        sample_file.close()

        if process in sample_dict.keys():
            return(sample_dict[process])
        else:
            print(process)
            raise Exception("Process String does not match library")
    

    def load_cross_section(self, dict_dir, process, target_material):
        cross_section_file=open( dict_dir + "xSec_Dicts.pkl", 'rb')
        cross_section_dict=pickle.load(cross_section_file)
        cross_section_file.close()

        if process not in cross_section_dict:
            raise Exception("Process String does not match library")
        
        if target_material in cross_section_dict[process]:
            return(cross_section_dict[process][target_material])
        else:
            raise Exception("Target Material is not in library")



        
    def set_dict_dir(self, value):
        """Set the top level directory containing pre-computed MC pickles to value"""
        self._dict_dir = value
    def get_dict_dir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._dict_dir
    def set_target_material(self, value):
        """Set the string representing the target material to value"""
        self._target_material = value
    def get_target_material(self):
        """Get the string representing the target material"""
        return self._target_material
    def set_material_properties(self):
        """Defines material properties (Z, A, rho, etc) based on the target 
        material label
        """
        self._ZTarget, self._ATarget, self._rhoTarget, self._dEdx = Z[self.get_target_material()], A[self.get_target_material()], rho[self.get_target_material()], dEdx[self.get_target_material()]

    def get_material_properties(self):
        """Returns target material properties: Z, A, rho, dE/dx"""
        return self._ZTarget, self._ATarget, self._rhoTarget, self._dEdx

    def set_n_targets(self):
        """Determines nuclear and electron target densities for the 
           target material
        """
        ZT, AT, rhoT, dEdxT = self.get_material_properties()
        self._nTarget = rhoT/mp0/AT
        self._nElecs = self._nTarget*ZT

    def set_samples(self):
        self._loaded_samples={}
        for Process in process_code.keys():
            self._loaded_samples[Process]= \
                self.load_sample(self._dict_dir, Process)
        self._Egamma_min = self._loaded_samples['Brem'][0][1]['Eg_min']
        
    def get_n_targets(self):
        """Returns nuclear and electron target densities for the 
           target material in 1/cm^3
        """

        return self._nTarget, self._nElecs

    def set_cross_sections(self):
        """Loads the pre-computed cross-sections for various shower processes 
        and extracts the minimum/maximum values of initial energies
        """

        # These contain only the cross sections for the chosen target material
        self._brem_cross_section = self.load_cross_section(self._dict_dir, 'Brem', self._target_material)
        self._pair_production_cross_section   = self.load_cross_section(self._dict_dir, 'PairProd', self._target_material)
        self._annihilation_cross_section  = self.load_cross_section(self._dict_dir, 'Ann', self._target_material) 
        self._compton_cross_section = self.load_cross_section(self._dict_dir, 'Comp', self._target_material) 
        self._moller_cross_section = self.load_cross_section(self._dict_dir, 'Moller', self._target_material) 
        self._bhabha_cross_section = self.load_cross_section(self._dict_dir, 'Bhabha', self._target_material) 

        self._EeVecBrem = np.transpose(self._brem_cross_section)[0] #FIXME: not sure what these are
        self._EgVecPP = np.transpose(self._pair_production_cross_section)[0]
        self._EeVecAnn = np.transpose(self._annihilation_cross_section)[0]
        self._EgVecComp = np.transpose(self._compton_cross_section)[0]
        self._EeVecMoller = np.transpose(self._moller_cross_section)[0]
        self._EeVecBhabha = np.transpose(self._bhabha_cross_section)[0]

        # log10s of minimum energes, energy spacing for the cross-section tables  
        self._logEeMinBrem, self._logEeSSBrem = np.log10(self._EeVecBrem[0]), np.log10(self._EeVecBrem[1]) - np.log10(self._EeVecBrem[0])
        self._logEeMinAnn, self._logEeSSAnn = np.log10(self._EeVecAnn[0]), np.log10(self._EeVecAnn[1]) - np.log10(self._EeVecAnn[0])
        self._logEgMinPP, self._logEgSSPP = np.log10(self._EgVecPP[0]), np.log10(self._EgVecPP[1]) - np.log10(self._EgVecPP[0])
        self._logEgMinComp, self._logEgSSComp= np.log10(self._EgVecComp[0]), np.log10(self._EgVecComp[1]) - np.log10(self._EgVecComp[0])
        self._logEeMinMoller, self._logEeSSMoller= np.log10(self._EeVecMoller[0]), np.log10(self._EeVecMoller[1]) - np.log10(self._EeVecMoller[0])
        self._logEeMinBhabha, self._logEeSSBhabha= np.log10(self._EeVecBhabha[0]), np.log10(self._EeVecBhabha[1]) - np.log10(self._EeVecBhabha[0])

    def get_brem_cross_section(self):
        """ Returns array of [energy,cross-section] values for brem """ 
        return self._brem_cross_section
    def get_pairprod_cross_section(self):
        """ Returns array of [energy,cross-section] values for pair production """ 
        return self._pair_production_cross_section
    def get_annihilation_cross_section(self):
        """ Returns array of [energy,cross-section] values for e+e- annihilation """ 
        return self._annihilation_cross_section
    def get_compton_cross_section(self):
        """ Returns array of [energy,cross-section] values for Compton """ 
        return self._compton_cross_section
    def get_moller_cross_section(self):
        """ Returns array of [energy,cross-section] values for Moller """ 
        return self._moller_cross_section
    def get_bhabha_cross_section(self):
        """ Returns array of [energy,cross-section] values for Bhabha """ 
        return self._bhabha_cross_section

    def set_NSigmas(self):
        """Constructs interpolations of n_T sigma (in 1/cm) as a functon of 
        incoming particle energy for each process
        """
        BS, PPS, AnnS, CS, MS, BhS = self.get_brem_cross_section(), self.get_pairprod_cross_section(), self.get_annihilation_cross_section(), self.get_compton_cross_section(), self.get_moller_cross_section(), self.get_bhabha_cross_section()
        nZ, ne = self.get_n_targets()
        self._NSigmaBrem = interp1d(np.transpose(BS)[0], nZ*GeVsqcm2*np.transpose(BS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaPP = interp1d(np.transpose(PPS)[0], nZ*GeVsqcm2*np.transpose(PPS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaAnn = interp1d(np.transpose(AnnS)[0], ne*GeVsqcm2*np.transpose(AnnS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaComp = interp1d(np.transpose(CS)[0], ne*GeVsqcm2*np.transpose(CS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaMoller = interp1d(np.transpose(MS)[0], ne*GeVsqcm2*np.transpose(MS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaBhabha = interp1d(np.transpose(BhS)[0], ne*GeVsqcm2*np.transpose(BhS)[1], fill_value=0.0, bounds_error=False)

    def get_mfp(self, PID, Energy): #FIXME: variable PID is not defined
        """Returns particle mean free path in meters for PID=22 (photons), 
        11 (electrons) or -11 (positrons) as a function of energy in GeV"""
        if PID == 22:
            return cmtom*(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))**-1
        elif PID == 11:
            return cmtom*(self._NSigmaBrem(Energy) + self._NSigmaMoller(Energy))**-1
        elif PID == -11:
            return cmtom*(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy) + self._NSigmaBhabha(Energy))**-1
        
    def BF_positron_brem(self, Energy):
        """Branching fraction for a positron to undergo brem vs annihilation"""
        b0, b1 = self._NSigmaBrem(Energy), self._NSigmaAnn(Energy)
        return b0/(b0+b1)
    def BF_photon_pairprod(self, Energy):
        """Branching fraction for a photon to undergo pair production vs compton"""
        b0, b1 = self._NSigmaPP(Energy), self._NSigmaComp(Energy)
        return b0/(b0+b1)


    def draw_sample(self,Einc,LU_Key,process,VB=False):

        sample_list=self._loaded_samples 

        # this grabs the dictionary part rather than the energy. 
        sample_dict=sample_list[process][LU_Key][1]

        adaptive_map = sample_dict["adaptive_map"]
        max_F      = sample_dict["max_F"]*self._maxF_fudge_global
        neval_vegas= sample_dict["neval"]
        integrand=vg.Integrator(map=adaptive_map, max_nhcube=1, neval=neval_vegas)

        event_info={'E_inc': Einc, 'm_e': m_electron, 'Z_T': self._ZTarget, 'A_T':self._ATarget, 'mT':self._ATarget, 'alpha_FS': alpha_em, 'mV': 0, 'Eg_min':self._Egamma_min}
        event_info_H={'E_inc': Einc, 'm_e': m_electron, 'Z_T': 1.0, 'A_T':1.0, 'mT':1.0, 'alpha_FS': alpha_em, 'mV': 0, 'Eg_min':self._Egamma_min}
        diff_xsection_options={"PairProd" : dsigma_pairprod_dimensionless,
                               "Comp"     : dsigma_compton_dCT,
                               "Brem"     : dsigma_brem_dimensionless,
                               "Ann"      : dsigma_annihilation_dCT,
                               "Moller"   : dsigma_moller_dCT,
                               "Bhabha"   : dsigma_bhabha_dCT }
        
        formfactor_dict      ={"PairProd" : g2_elastic,
                               "Comp"     : unity,
                               "Brem"     : g2_elastic,
                               "Ann"      : unity,
                               "Moller"   : unity,
                               "Bhabha"   : unity }

        QSq_dict             ={"PairProd" : pair_production_q_sq_dimensionless, "Brem"     : brem_q_sq_dimensionless, "Comp": dummy, "Ann": dummy, "Moller":dummy, "Bhabha":dummy }

        
        if process in diff_xsection_options:
            diff_xsec_func = diff_xsection_options[process]
            FF_func        = formfactor_dict[process]
            QSq_func       = QSq_dict[process]
        else:
            raise Exception("Your process is not in the list")

        if VB:
            sampcount = 0
        n_integrators_used = 0
        sample_found = False
        while sample_found is False and n_integrators_used < self._max_n_integrators:
            n_integrators_used += 1
            for x,wgt in integrand.random():
                FF_eval=FF_func(event_info, QSq_func(x, event_info ) )/event_info['Z_T']**2
                FF_H = FF_func(event_info_H, QSq_func(x, event_info_H) ) 
                if VB:
                    sampcount += 1  
                if  max_F*draw_U()<wgt*diff_xsec_func(event_info,x)*FF_eval/FF_H:
                    sample_found = True
                    break
        if sample_found is False:
            print("No Sample Found")
            #FIXME What to do when we end up here?
        if VB:
            return np.concatenate([list(x), [sampcount]])
        else:
            return(x)



    def electron_brem_sample(self, Elec0, VB=False):
        """Generate a brem event from an initial electron/positron
            Args:
                Elec0: incoming electron/positron (instance of) Particle 
                in lab frame
            Returns:
                [NewE, NewG] where
                NewE: outgoing electron/positron (instance of) Particle 
                in lab frame
                NewG: outgoing photon (instance of) Particle 
                in lab frame
        """
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Ee0) - self._logEeMinBrem)/self._logEeSSBrem)
        LUKey = LUKey + 1
        
        sample_event = self.draw_sample(Ee0, LUKey, 'Brem', VB=VB)
        x1, x2, x3, x4 = sample_event[:4]
        w = self._Egamma_min + x1*(Ee0 - m_electron - self._Egamma_min)
        ct = np.cos((x2+x3)/2)
        ctp = np.cos((x2-x3)*Ee0/(2*(Ee0-w)))
        ph = (x4-1/2)*2.0*np.pi
                
        # reconstruct final electron and photon 4-momenta from the MC-sampled variables
        NFVs = e_to_egamma_fourvecs(Ee0, m_electron, w, ct, ctp, ph)

        Eef, pexfZF, peyfZF, pezfZF = NFVs[1]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[2]

        pe3ZF = [pexfZF, peyfZF, pezfZF]
        pg3ZF = [pgxfZF, pgyfZF, pgzfZF]
        
        # Rotate back to lab frame
        pe3LF = np.dot(RM, pe3ZF)
        pg3LF = np.dot(RM, pg3ZF)
        
        pos = Elec0.get_rf()
        init_IDs = Elec0.get_ids()

        if VB:
            newparticlewgt = sample_event[-1]
        else:
            newparticlewgt = 1.0
        NewE = Particle(init_IDs[0], Eef, pe3LF[0], pe3LF[1], pe3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1,process_code['Brem'], newparticlewgt)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Brem'], newparticlewgt)

        return [NewE, NewG]

    def AnnihilationSample(self, Elec0, VB=False):
        """Generate an annihilation event from an initial positron
            Args:
                Elec0: incoming positron (instance of) Particle in lab frame
            Returns:
                [NewG1, NewG2]: outgoing photons (instances of) Particle 
                in lab frame
        """

        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Ee0) - self._logEeMinAnn)/self._logEeSSAnn)
        LUKey = LUKey + 1
        
        SampEvt = self.draw_sample(Ee0, LUKey, 'Ann', VB=VB)

        # reconstruct final photon 4-momenta from the MC-sampled variables
        NFVs = annihilation_fourvecs(Ee0, m_electron, 0.0, SampEvt[0])

        Eg1f, pg1xfZF, pg1yfZF, pg1zfZF = NFVs[0]
        Eg2f, pg2xfZF, pg2yfZF, pg2zfZF = NFVs[1]

        pg3ZF1 = [pg1xfZF, pg1yfZF, pg1zfZF]
        pg3ZF2 = [pg2xfZF, pg2yfZF, pg2zfZF]
    
        pg3LF1 = np.dot(RM, pg3ZF1)
        pg3LF2 = np.dot(RM, pg3ZF2)   

        pos = Elec0.get_rf()
        init_IDs = Elec0.get_ids()

        if VB:
            newparticlewgt = SampEvt[-1]
        else:
            newparticlewgt = 1.0

        NewG1 = Particle(22, Eg1f, pg3LF1[0], pg3LF1[1], pg3LF1[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Ann'], newparticlewgt)
        NewG2 = Particle(22, Eg2f, pg3LF2[0], pg3LF2[1], pg3LF2[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Ann'], newparticlewgt)

        return [NewG1, NewG2]

    def pairprod_sample(self, Phot0, VB=False):
        """Generate a photon splitting event from an initial photon
            Args:
                Phot0: incoming positron (instance of) Particle in lab frame
            Returns:
                [NewEp, NewEm]: outgoing positron and electron (instances of) Particle 
                in lab frame
        """
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Eg0) - self._logEgMinPP)/self._logEgSSPP)
        LUKey = LUKey + 1
        
        sample_event = self.draw_sample(Eg0, LUKey, 'PairProd', VB=VB)
        x1, x2, x3, x4 = sample_event[:4]
        epp = m_electron + x1*(Eg0-2*m_electron)
        ctp = np.cos(Eg0*(x2+x3)/(2*epp))
        ctm = np.cos(Eg0*(x2-x3)/(2*(Eg0-epp)))
        ph = x4*2*np.pi
        
        # reconstruct final electron and positron 4-momenta from the MC-sampled variables
        NFVs = gamma_to_epem_fourvecs(Eg0, m_electron, epp, ctp, ctm, ph)
        Eepf, pepxfZF, pepyfZF, pepzfZF = NFVs[1]
        Eemf, pemxfZF, pemyfZF, pemzfZF = NFVs[2]

        pep3ZF = [pepxfZF, pepyfZF, pepzfZF]
        pem3ZF = [pemxfZF, pemyfZF, pemzfZF]

        pep3LF = np.dot(RM, pep3ZF)
        pem3LF = np.dot(RM, pem3ZF)

        pos = Phot0.get_rf()
        init_IDs = Phot0.get_ids()

        if VB:
            newparticlewgt = sample_event[-1]
        else:
            newparticlewgt = 1.0

        NewEp = Particle(-11,Eepf, pep3LF[0], pep3LF[1], pep3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['PairProd'], newparticlewgt)
        NewEm = Particle(11, Eemf, pem3LF[0], pem3LF[1], pem3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['PairProd'], newparticlewgt)

        return [NewEp, NewEm]

    def compton_sample(self, Phot0, VB=False):
        """Generate a Compton event from an initial photon
            Args:
                Phot0: incoming photon (instance of) Particle in lab frame
            Returns:
                [NewE, NewG]: electron and photon (instances of) Particle 
                in lab frame
        """

        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Eg0) - self._logEgMinComp)/self._logEgSSComp)        
        LUKey = LUKey + 1
        
        SampEvt = self.draw_sample(Eg0, LUKey, 'Comp', VB=VB)

        
        # reconstruct final electron and photon 4-momenta from the MC-sampled variables
        NFVs = compton_fourvecs(Eg0, m_electron, 0.0, SampEvt[0])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[0]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[1]


        pe3LF = np.dot(RM, [pexfZF, peyfZF, pezfZF])
        pg3LF = np.dot(RM, [pgxfZF, pgyfZF, pgzfZF])

        pos = Phot0.get_rf()
        init_IDs = Phot0.get_ids()

        if VB:
            newparticlewgt = SampEvt[-1]
        else:
            newparticlewgt = 1.0

        NewE = Particle(11, Eef, pe3LF[0], pe3LF[1], pe3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Comp'], newparticlewgt)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Comp'], newparticlewgt)

        return [NewE, NewG]
    
    def moller_bhabha_sample(self, ElecPos0, Process="Moller", VB=False):
        """Generate a Moller or Bhabha scattering event from an initial electron/positron
            Args:
                ElecPos0: incoming electron/positron (instance of) Particle in lab frame
            Returns:
                [NewEP, NewE]: electron/positron and electron (instances of) Particle 
                in lab frame
        """

        Ee0, pex0, pey0, pez0 = ElecPos0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        if Process == "Moller":
            LUKey = int((np.log10(Ee0) - self._logEeMinMoller)/self._logEeSSMoller)
        elif Process == "Bhabha":
            LUKey = int((np.log10(Ee0) - self._logEeMinBhabha)/self._logEeSSBhabha)
        else:
            raise Exception("Wrong Process specified for Moller/Bhabha sampling")

        LUKey = LUKey + 1
        SampEvt = self.draw_sample(Ee0, LUKey, Process, VB=VB)


        # reconstruct final electron and photon 4-momenta from the MC-sampled variables
        NFVs = ee_to_ee_fourvecs(Ee0, m_electron, SampEvt[0])

        Eepf, pepxfZF, pepyfZF, pepzfZF = NFVs[0]
        Eef, pexfZF, peyfZF, pezfZF = NFVs[1]

        pep3LF = np.dot(RM, [pepxfZF, pepyfZF, pepzfZF])
        pe3LF = np.dot(RM, [pexfZF, peyfZF, pezfZF])

        pos = ElecPos0.get_rf()
        init_IDs = ElecPos0.get_ids()

        if VB:
            newparticlewgt = SampEvt[-1]
        else:
            newparticlewgt = 1.0

        NewEP = Particle(init_IDs[0], Eepf, pep3LF[0], pep3LF[1], pep3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code[Process], newparticlewgt)
        NewE = Particle(11, Eef, pe3LF[0], pe3LF[1], pe3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code[Process], newparticlewgt)

        return [NewEP, NewE]

    def propagate_particle(self, Part0, Losses=False, MS=False):
        """Propagates a particle through material between hard scattering events, 
        possibly including multiple scattering and dE/dx losses
            Args:
                Part0: initial Particle object
                Losses: bool that indicates whether to include dE/dx losses
                MS: bool that indicates whether to include multiple scattering
            Returns:
                Part0: updated Particle object with new position and 
                (potentially) energy/momentum
        """
        if Part0.get_ended() is True:
            Part0.set_rf(Part0.get_rf())
            return Part0
        else:
            if Part0.get_ids()[0] == 11 and (np.log10(Part0.get_p0()[0]) < np.max([self._logEeMinBhabha, self._logEeMinBrem])):
                Part0.set_ended(True)
                return Part0
            elif Part0.get_ids()[0] == -11 and (np.log10(Part0.get_p0()[0]) < np.max([self._logEeMinBhabha, self._logEeMinBrem, self._logEeMinAnn])):
                Part0.set_ended(True)
                return Part0
            elif Part0.get_ids()[0] == 22 and (np.log10(Part0.get_p0()[0]) < np.max([self._logEgMinComp, self._logEgMinPP])):
                Part0.set_ended(True)
                return Part0
            mfp = self.get_mfp(Part0.get_ids()[0], Part0.get_p0()[0])
            distC = np.random.uniform(0.0, 1.0)
            dist = mfp*np.log(1.0/(1.0-distC))
            if np.abs(Part0.get_ids()[0]) == 11:
                M0 = m_electron
            elif Part0.get_ids()[0] == 22:
                M0 = 0.0

            E0, px0, py0, pz0 = Part0.get_p0()
            if MS:
                ZT, AT, rhoT, dEdxT = self.get_material_properties()
                EF0, PxF0, PyF0, PzF0 = get_scattered_momentum(Part0.get_p0(), rhoT*(dist/cmtom), AT, ZT)
                PHatDenom = np.sqrt((PxF0 + px0)**2 + (PyF0 + py0)**2 + (PzF0 + pz0)**2)
                PHat = [(PxF0 + px0)/PHatDenom, (PyF0 + py0)/PHatDenom, (PzF0 + pz0)/PHatDenom]
            else:
                PHatDenom = np.sqrt(px0**2 + py0**2 + pz0**2)
                PHat = [(px0)/PHatDenom, (py0)/PHatDenom, (pz0)/PHatDenom]

            p30 = np.sqrt(px0**2 + py0**2 + pz0**2)

            x0, y0, z0 = Part0.get_r0()
            Part0.set_rf([x0 + PHat[0]*dist, y0 + PHat[1]*dist, z0 + PHat[2]*dist])

            if Losses is False:
                if MS:
                    Part0.set_pf(np.array([E0, PxF0, PyF0, PzF0]))
                else:
                    Part0.set_pf(Part0.get_p0())
            else:
                Ef = E0 - Losses*dist
                if Ef <= M0 or Ef < self.min_energy:
                    #print("Particle lost too much energy along path of propagation!")
                    Part0.set_ended(True)
                    return Part0
                Part0.set_pf(np.array([Ef, px0/p30*np.sqrt(Ef**2-M0**2), py0/p30*np.sqrt(Ef**2-M0**2), pz0/p30*np.sqrt(Ef**2-M0**2)]))

            Part0.set_ended(True)
            return Part0

    def generate_shower(self, PID0, p40, ParPID, VB=False, GlobalMS=True):
        """
        Generates particle shower from an initial particle
        Args:
            PID0: PDG ID of the initial particle
            p40: four-momentum of the initial particle
            ParID: PDG ID of the parent of the initial particle
            VB: bool to turn on/off verbose output

        Returns:
            AllParticles: a list of all particles generated in the shower
        """
        p0 = Particle(PID0, p40[0], p40[1], p40[2], p40[3], 0.0, 0.0, 0.0, 1, 0, ParPID, 0, -1, 1.0)
        if VB:
            print("Starting shower, initial particle with ID Info")
            print(p0.get_ids())
            print("Initial four-momenta:")
            print(p0.get_p0())

        all_particles = [p0]

        if GlobalMS==True:
            MS_e=True
            MS_g=False
        else:
            MS_e=False
            MS_g=False

        if p0.get_p0()[0] < self.min_energy:
            p0.set_ended(True)
            return all_particles

        while all([ap.get_ended() == True for ap in all_particles]) is False:
            for apI, ap in enumerate(all_particles):
                if ap.get_ended() is True:
                    continue
                else:
                    # Propagate particle until next hard interaction
                    if ap.get_ids()[0] == 22:
                        ap = self.propagate_particle(ap,MS=MS_g)
                    elif np.abs(ap.get_ids()[0]) == 11:
                        dEdxT = self.get_material_properties()[3]*(0.1) #Converting MeV/cm to GeV/m
                        ap = self.propagate_particle(ap, MS=MS_e, Losses=dEdxT)
                    
                    all_particles[apI] = ap

                    if (all([ap.get_ended() == True for ap in all_particles]) is True and ap.get_pf()[0] < self.min_energy):
                        break

                    # Generate secondaries for the hard interaction
                    # Note: secondaries include the scattered parent particle 
                    # (i.e. the original the parent is not modified)
                    if ap.get_ids()[0] == 11:
                        choices0 = self._NSigmaBrem(ap.get_pf()[0]), self._NSigmaMoller(ap.get_pf()[0])
                        SC = np.sum(choices0)
                        if SC == 0.0:
                            continue
                        choices0 = choices0/SC
                        draw = np.random.choice([0,1], p=choices0)
                        if draw == 0:
                            npart = self.electron_brem_sample(ap, VB=VB)
                        else:
                            npart = self.moller_bhabha_sample(ap, Process="Moller", VB=VB)
                    elif ap.get_ids()[0] == -11:
                        choices0 = self._NSigmaBrem(ap.get_pf()[0]), self._NSigmaAnn(ap.get_pf()[0]), self._NSigmaBhabha(ap.get_pf()[0])
                        SC = np.sum(choices0)
                        if SC == 0.0:
                            continue
                        choices0 = choices0/SC
                        draw = np.random.choice([0,1,2], p=choices0)
                        if draw == 0:
                            npart = self.electron_brem_sample(ap, VB=VB)
                        elif draw == 1:
                            npart = self.AnnihilationSample(ap, VB=VB)
                        else:
                            npart = self.moller_bhabha_sample(ap, Process="Bhabha", VB=VB)
                    elif ap.get_ids()[0] == 22:
                        BFPhPP = self.BF_photon_pairprod(ap.get_pf()[0])
                        ch = np.random.uniform(low=0., high=1.)
                        if ch < BFPhPP:
                            npart = self.pairprod_sample(ap, VB=VB)
                        else:
                            npart = self.compton_sample(ap, VB=VB)
                    if (npart[0]).get_p0()[0] > self.min_energy:
                        all_particles.append(npart[0])
                    if (npart[1]).get_p0()[0] > self.min_energy:
                        all_particles.append(npart[1])

        return all_particles
