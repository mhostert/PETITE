import numpy as np
import pickle 

from scipy.interpolate import interp1d

from .moliere import get_scattered_momentum 
from .particle import Particle
from .kinematics import e_to_egamma_fourvecs, e_to_eV_fourvecs, gamma_to_epem_fourvecs, compton_fourvecs, annihilation_fourvecs
from .shower import Shower
from .all_processes import *

import sys
from numpy.random import random as draw_U
        

Z = {'hydrogen':1.0, 'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'hydrogen':1.0, 'graphite':12.0, 'lead':207.2} #atomic mass of different targets

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

dark_process_code = {'ExactBrem':0, 'Ann': 1, 'Comp': 3}
dark_kinematic_function = {"ExactBrem" : e_to_eV_fourvecs,
                           "Ann"       : annihilation_fourvecs,
                           "Comp"      : compton_fourvecs}

class DarkShower(Shower):
    """ A class to reprocess an existing EM shower to generate dark photons
    """

    def __init__(self, dict_dir, target_material, min_energy, mV_in_GeV , \
                          mode="exact", maxF_fudge_global=1,max_n_integrators=int(1e4)):
        super().__init__(dict_dir, target_material, min_energy)
        """Initializes the shower object.
        Args:
            PickDir: directory containing the pre-computed MC samples of various shower processes
            TargetMaterial: string label of the homogeneous material through which 
            particles propagate (available materials are the dict keys of 
            Z, A, rho, etc)
            MinEnergy: minimum particle energy in GeV at which the particle 
            finishes its propagation through the target
            mV_string: str which determines the pre-computed MC sample of massive
            vector events to use (see MVLib variable for available choices)
        """


        self.set_dark_dict_dir(dict_dir)
        self.set_target_material(target_material)
        self.min_energy = min_energy

        self.set_material_properties()
        self.set_n_targets()
        self.set_mV_list(dict_dir)
        self.set_mV(mV_in_GeV, mode)
        
        self.set_dark_cross_sections()
        self.set_dark_NSigmas()
        self.set_dark_samples()


        self._maxF_fudge_global=maxF_fudge_global
        self._max_n_integrators=max_n_integrators
  
    
    def set_dark_dict_dir(self, value):
        """Set the directory containing pre-simulated MC events for processes involing target nuclei"""
        self._dark_dict_dir = value

    def get_dark_dict_dir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._dark_dict_dir
   

    def set_mV_list(self,dict_dir):
        sample_file=open(dict_dir + "dark_maps.pkl", 'rb')
        outer_dict=pickle.load(sample_file)
        sample_file.close()

        mass_list=list(outer_dict.keys())

        self._mV_list=mass_list


    def closest_lesser_value(self, input_list, input_value):
        arr = np.asarray(input_list)
        index = (np.abs(arr - input_value)).argmin()

        if arr[index]<=input_value:
            return(arr[index])
        else:
            return(arr[index-1])

    def set_mV(self, value, mode):
        """Set MVStr to value and extract the corresponding numerical mass of the dark photon"""
        if mode=='exact':
            self._mV= self.closest_lesser_value(self._mV_list, value)
            self._mV_estimator = self._mV
        elif mode=='approx':
            self._mV= value
            self._mV_estimator = self.closest_lesser_value(self._mV_list, value)
        else:
            raise Exception("Mode not valid. Chose exact or approx.")

    
    def get_mV(self):
        """Get the numerical value of the dark vector mass"""
        return self._mV
    
    def load_dark_sample(self, dict_dir, process): 
        sample_file=open(dict_dir + "dark_maps.pkl", 'rb')
        outer_dict=pickle.load(sample_file)
        sample_file.close()

        sample_dict=outer_dict[self._mV_estimator]
        if process in sample_dict.keys():
            return(sample_dict[process])
        else:
            print(process)
            raise Exception("Process String does not match library")
        

    def set_dark_samples(self):
        self._loaded_dark_samples={}
        for process in dark_process_code.keys():
            self._loaded_dark_samples[process]= \
                self.load_dark_sample(self._dict_dir, process)
            


    def load_dark_cross_section(self, dict_dir, process, target_material):
        dark_cross_section_file=open( dict_dir + "dark_xsecs.pkl", 'rb')
        outer_dict=pickle.load(dark_cross_section_file)
        dark_cross_section_file.close()

        dark_cross_section_dict=outer_dict[self._mV_estimator]

        if process not in dark_cross_section_dict:
            raise Exception("Process String does not match library")
        
        if target_material in dark_cross_section_dict[process]:
            return(dark_cross_section_dict[process][target_material])

        else:
            raise Exception("Target Material is not in library")

    
    def set_dark_cross_sections(self):
        """Loads the pre-computed cross-sections for various shower processes 
        and extracts the minimum/maximum values of initial energies
        """

        # These contain only the cross sections for the chosen target material
        self._dark_brem_cross_section = self.load_dark_cross_section(self._dict_dir, 'ExactBrem', self._target_material)
        self._dark_annihilation_cross_section  = self.load_dark_cross_section(self._dict_dir, 'Ann', self._target_material) 
        self._dark_compton_cross_section = self.load_dark_cross_section(self._dict_dir, 'Comp', self._target_material) 

        self._EeVecDarkBrem = np.transpose(self._dark_brem_cross_section)[0] #FIXME: not sure what these are
        self._EeVecDarkAnn = np.transpose(self._dark_annihilation_cross_section)[0]
        self._EgVecDarkComp = np.transpose(self._dark_compton_cross_section)[0]

        # log10s of minimum energes, energy spacing for the cross-section tables  
        self._logEeMinDarkBrem, self._logEeSSDarkBrem = np.log10(self._EeVecDarkBrem[0]), np.log10(self._EeVecDarkBrem[1]) - np.log10(self._EeVecDarkBrem[0])
        self._logEeMinDarkAnn, self._logEeSSDarkAnn = np.log10(self._EeVecDarkAnn[0]), np.log10(self._EeVecDarkAnn[1]) - np.log10(self._EeVecDarkAnn[0])
        self._logEgMinDarkComp, self._logEgSSDarkComp= np.log10(self._EgVecDarkComp[0]), np.log10(self._EgVecDarkComp[1]) - np.log10(self._EgVecDarkComp[0])


    def get_DarkBremXSec(self):
        """ Returns array of [energy,cross-section] values for brem """ 
        return self._dark_brem_cross_section 
    def get_DarkAnnXSec(self):
        """ Returns array of [energy,cross-section] values for e+e- annihilation """ 
        return self._dark_annihilation_cross_section
    def get_DarkCompXSec(self):
        """ Returns array of [energy,cross-section] values for Compton """ 
        return self._dark_compton_cross_section

    def set_dark_NSigmas(self):
        """Constructs interpolations of n_T sigma (in 1/cm) as a functon of 
        incoming particle energy for each process
        """
        DBS, DAnnS, DCS = self.get_DarkBremXSec(), self.get_DarkAnnXSec(), self.get_DarkCompXSec()
        nZ, ne = self.get_n_targets()
        self._NSigmaDarkBrem = interp1d(np.transpose(DBS)[0], nZ*GeVsqcm2*np.transpose(DBS)[1])
        self._NSigmaDarkAnn = interp1d(np.transpose(DAnnS)[0], ne*GeVsqcm2*np.transpose(DAnnS)[1])
        self._NSigmaDarkComp = interp1d(np.transpose(DCS)[0], ne*GeVsqcm2*np.transpose(DCS)[1])

    def GetBSMWeights(self, PID, Energy, process):
        """Compute relative weight of dark photon emission to the available SM processes
        Args: 
            PID: incoming PDG ID of the particle 
            Energy: its energy
        Returns:
            float, representing probability of V emission (for a fixed kinetic mixing) 
            divided by the probabilities of available SM processes

        """
        if PID == 22:
            if (Energy < self._mV*(1 + self._mV/(2*m_electron))) or (process != "Comp"):
                return 0.0
            else:
                return self._NSigmaDarkComp(Energy)/(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))
        elif PID == 11:
            if (np.log10(Energy) < self._logEeMinDarkBrem) or (process != "ExactBrem"):
                return 0.0
            else:
                return self._NSigmaDarkBrem(Energy)/self._NSigmaBrem(Energy)
        elif PID == -11:
            if process == "ExactBrem":
                if np.log10(Energy) < self._logEeMinDarkBrem:
                    BremPiece = 0.0
                else:
                    BremPiece = self._NSigmaDarkBrem(Energy)
                return BremPiece/(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy) + self._NSigmaBhabha(Energy))
            elif process == "Ann":
                if Energy < (self._mV**2 - m_electron**2)/(2*m_electron) + 2*self._Egamma_min:
                    AnnPiece = 0.0
                else:
                    AnnPiece = self._NSigmaDarkAnn(Energy)
                return AnnPiece/(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy) + self._NSigmaBhabha(Energy))
            else:
                return 0.0



    def draw_dark_sample(self,Einc,LU_Key=-1,process="ExactBrem",VB=False):

        dark_sample_list=self._loaded_dark_samples 
        if LU_Key<0 or LU_Key > len(dark_sample_list[process]):
            # Get the LU_Key corresponding to the closest incoming energy
            energies = dark_sample_list[process][0:]
            energies = np.array([x[0] for x in energies])
            # Get index of nearest (higher) energy
            LU_Key = np.argmin(np.abs(energies - Einc)) + 1

        # this grabs the dictionary part rather than the energy. 
        dark_sample_dict=dark_sample_list[process][LU_Key][1]

        integrand = dark_sample_dict["integrator"]
        max_F      = dark_sample_dict["max_F"][self._target_material]*self._maxF_fudge_global
        neval_vegas= dark_sample_dict["neval"]
        integrand=vg.Integrator(map=integrand, max_nhcube=1, neval=neval_vegas)

        event_info={'E_inc': Einc, 'm_e': m_electron, 'Z_T': self._ZTarget, 'A_T':self._ATarget, 'mT':self._ATarget, 'alpha_FS': alpha_em, 'mV': self._mV, 'Eg_min':self._Egamma_min}
        #event_info_H={'E_inc': Einc, 'm_e': m_electron, 'Z_T': 1.0, 'A_T':1.0, 'mT':1.0, 'alpha_FS': alpha_em, 'mV': self._mV, 'Eg_min':self._Egamma_min}

        diff_xsection_options={"Comp"     : dsigma_compton_dCT,
                               "ExactBrem": dsig_etl_helper,
                               "Ann"      : dsigma_annihilation_dCT }
        
        formfactor_dict      ={"Comp"     : unity,
                               "ExactBrem": Gelastic_inelastic,
                               "Ann"      : unity }

        QSq_dict             ={"ExactBrem": darkbrem_qsq,
                                "Comp"    : dummy, 
                                "Ann"     : dummy }

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
                #FF_eval=FF_func(event_info, QSq_func(x, event_info ) )/event_info['Z_T']**2
                #FF_H = FF_func(event_info_H, QSq_func(x, event_info_H ) ) 
                if VB:
                    sampcount += 1  
                if  max_F*draw_U()<wgt*diff_xsec_func(event_info,x):#*FF_eval/FF_H:
                    sample_found = True
                    break
        if sample_found is False:
            print("No Sample Found")
            print(process, Einc, LU_Key)
            #FIXME What to do when we end up here?
            #Coordinate with SM solution
        if VB:
            return np.concatenate([list(x), [sampcount]])
        else:
            return(x)

    def GetPositronDarkBF(self, Energy):
        """Branching fraction for a positron to undergo dark brem vs dark 
        annihilation"""
        if Energy < (self._mV**2 - m_electron**2)/(2*m_electron) + 2*self._Egamma_min:
            return 1.0
        else:
            return self._NSigmaDarkBrem(Energy)/(self._NSigmaDarkBrem(Energy) + self._NSigmaDarkAnn(Energy))

    def produce_bsm_particle(self, p0, process, VB=False):
        E0 = p0.get_pf()[0]
        RM = p0.rotation_matrix()
        sample_event = self.draw_dark_sample(E0, process=process, VB=VB)

        #dark-production is estabilished such that the last particle returned corresponds to the dark vector
        EVf, pVxfZF, pVyfZF, pVzfZF = dark_kinematic_function[process](p0, sample_event, mV=self._mV_estimator)[-1]
        pV4LF = np.concatenate([[EVf], np.dot(RM, [pVxfZF, pVyfZF, pVzfZF])])

        wg = self.GetBSMWeights(p0.get_ids()["PID"], E0, process)

        init_IDs = p0.get_ids()
        V_dict = {}
        V_dict["PID"] = 4900022
        V_dict["parent_PID"] = init_IDs["PID"]
        V_dict["ID"] = 2*(init_IDs["ID"]) + 0
        V_dict["parent_ID"] = init_IDs["ID"]
        V_dict["generation_number"] = init_IDs["generation_number"] + 1
        V_dict["generation_process"] = process
        V_dict["weight"] = wg

        return Particle(pV4LF, p0.get_rf(), V_dict)


    def DarkElecBremSample(self, Elec0, VB=False, relative_weight=1.0):
        """Generate a brem event from an initial electron/positron
            Args:
                Elec0: incoming electron/positron (instance of) Particle 
                in lab frame
                relative_weight (optional): a reweighting factor relative
                to the overall BSM weight
            Returns:
                NewV: outgoing dark photon (instance of) Particle 
                in lab frame
        """
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        try:
            LUKey = int((np.log10(Ee0) - self._logEeMinDarkBrem)/self._logEeSSDarkBrem)
        except:
            print(Ee0, self._logEeMinDarkBrem, self._logEeSSDarkBrem, (np.log10(Ee0) - self._logEeMinDarkBrem)/self._logEeSSDarkBrem)
        LUKey = LUKey + 1
        
        ## FIXME Need right key name
        sample_event = self.draw_dark_sample(Ee0, LUKey, 'ExactBrem', VB=VB)
        EV = sample_event[0]*Ee0
        ct = (1 - 10**sample_event[1])
        EVf, pVxfZF, pVyfZF, pVzfZF = e_to_eV_fourvecs(Ee0, m_electron, EV, self.get_mV(), ct, 0, 0)[2]
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV4LF = np.concatenate([[EVf], np.dot(RM, pV3ZF)])

        if EVf > Ee0:
            print("---------------------------------------------")
            print("High Energy V Found from Electron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(sample_event)
            print(LUKey)
            print(Ee0)
            print("---------------------------------------------")

        wg = self.GetBSMWeights(11, Ee0)*relative_weight
        
        init_IDs = Elec0.get_ids()
        VDict = {}
        VDict["PID"] = 4900022
        VDict["parent_PID"] = init_IDs["PID"]
        VDict["ID"] = 2*(init_IDs["ID"]) + 1
        VDict["parent_ID"] = init_IDs["ID"]
        VDict["generation_number"] = init_IDs["generation_number"] + 1
        VDict["generation_process"] = "ExactBrem"
        VDict["weight"] = wg
        NewV = Particle(pV4LF, Elec0.get_rf(), VDict)

        return NewV

    def DarkAnnihilationSample(self, Elec0, VB=False, relative_weight=1.0):
        """Generate an annihilation event from an initial positron
            Args:
                Elec0: incoming positron (instance of) Particle in lab frame
                relative_weight (optional): a reweighting factor relative 
                to the overall BSM weight
        Returns:
                NewV: outgoing dark photon (instances of) Particle 
                in lab frame
        """

        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinDarkAnn)/self._logEeSSDarkAnn)
        LUKey = LUKey + 1
        
        # FIXME  need right key name 
        sample_event = self.draw_dark_sample(Ee0, LUKey, 'Ann', VB=VB)
        #NFVs = Ann_FVs(EeMod, meT, MVT, SampEvt[0])[1]
        #NFVs = annihilation_fourvecs(Ee0, m_electron, self.get_mV(), sample_event[0])[1]
        NFVs = annihilation_fourvecs(Elec0, sample_event, mV=self._mV_estimator)[1]

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV4LF = np.concatenate([[EVf], np.dot(RM, pV3ZF)])
        wg = self.GetBSMWeights(-11, Ee0)*relative_weight

        if EVf > Ee0+m_electron/2+(2*Ee0-m_electron)*self.get_mV()**2/(8*Ee0**2):
            print("---------------------------------------------")
            print("High Energy V Found from Positron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(sample_event)
            print(LUKey)
            print(wg)
            print("---------------------------------------------")

        init_IDs = Elec0.get_ids()
        VDict = {}
        VDict["PID"] = 4900022
        VDict["parent_PID"] = init_IDs["PID"]
        VDict["ID"] = 2*(init_IDs["ID"]) + 1
        VDict["parent_ID"] = init_IDs["ID"]
        VDict["generation_number"] = init_IDs["generation_number"] + 1
        VDict["generation_process"] = "Ann"
        VDict["weight"] = wg
        NewV = Particle(pV4LF, Elec0.get_rf(), VDict)

        return NewV

    def DarkComptonSample(self, Phot0, VB=False):
        """Generate a dark Compton event from an initial photon
            Args:
                Phot0: incoming photon (instance of) Particle in lab frame
            Returns:
                NewV: outgoing dark photon (instances of) Particle 
                in lab frame
        """
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinDarkComp)/self._logEgSSDarkComp)
        LUKey = LUKey + 1
        
        # FIXME Need right key name
        sample_event = self.draw_dark_sample(Eg0, LUKey, 'Comp', VB=VB)

        #NFVs = Compton_FVs(EgMod, meanniT, MVT, SampEvt[0])[1]
        EVf, pVxfZF, pVyfZF, pVzfZF = compton_fourvecs(Phot0, sample_event, mV=self._mV_estimator)[1]
        #EVf, pVxfZF, pVyfZF, pVzfZF = compton_fourvecs(Eg0, m_electron, self.get_mV(), sample_event[0])[1]
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV4LF = np.concatenate([[EVf], np.dot(RM, pV3ZF)])

        wg = self.GetBSMWeights(22, Eg0)
        GenType = dark_process_code['Comp']
        if EVf > Eg0:
            print("---------------------------------------------")
            print("High Energy V Found from Photon Samples:")
            print(Phot0.get_pf())
            print(EVf)
            print(sample_event)
            print(LUKey)
            print(wg)
            print("---------------------------------------------")

        init_IDs = Phot0.get_ids()
        VDict = {}
        VDict["PID"] = 4900022
        VDict["parent_PID"] = init_IDs["PID"]
        VDict["ID"] = 2*(init_IDs["ID"]) + 1
        VDict["parent_ID"] = init_IDs["ID"]
        VDict["generation_number"] = init_IDs["generation_number"] + 1
        VDict["generation_process"] = "Ann"
        VDict["weight"] = wg
        NewV = Particle(pV4LF, Phot0.get_rf(), VDict)

        return NewV

    def generate_dark_shower(self, ExDir=None, SParams=None):
        """ Process an existing SM shower (or produce a new one) by interating 
        through its particles and generating possible dark photon emissions using 
        all available processes.
        Args:
            ExDir: path to file containing existing SM shower OR an actual shower (list of Particle objects)
            SParamas: if no path provided, incident particle of a new SM shower to generate, 
            consisting of a "Particle" object
        Returns:
            [ShowerToSamp, NewShower]: where ShowerToSamp is the initial SM shower and NewShower 
            is the list of possible dark photon emissions generated from it
        """
        if ExDir is None and SParams is None:
            print("Need an existing SM shower-file directory or SM incident particle to run dark shower")
            return None
        
        if ExDir is not None and type(ExDir)==str:
            ShowerToSamp = np.load(ExDir, allow_pickle=True)
        elif ExDir is not None and type(ExDir)==list:
            ShowerToSamp = ExDir
        elif type(SParams)==Particle:
            ShowerToSamp = self.generate_shower(SParams)
        else:
            raise ValueError("Provided SParams must be a `Particle' class object")

        NewShower = []
        for ap in ShowerToSamp:
            for process_code in dark_process_code.keys():
                if self.GetBSMWeights(ap.get_ids()["PID"], ap.get_pf()[0], process=process_code) > 0.0:
                    npart = self.produce_bsm_particle(ap, process=process_code)
                    NewShower.append(npart)
            return ShowerToSamp, NewShower
            '''
            #if self.GetBSMWeights(ap.get_ids()["PID"], ap.get_pf()[0]) == 0.0:
            #    continue
            if ap.get_ids()["PID"] == 11:
                #if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem or np.isnan(ap.get_pf()[0]):
                #    continue
                #npart = self.DarkElecBremSample(ap)
                if self.GetBSMWeights(ap.get_ids()["PID"], ap.get_pf()[0], process="ExactBrem") == 0.0:
                    continue
                npart = self.produce_bsm_particle(ap, process="ExactBrem")
                NewShower.append(npart)
            elif ap.get_ids()["PID"] == -11:
                #if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem or np.isnan(ap.get_pf()[0]):
                #    continue
                #DarkBFEpBrem = self.GetPositronDarkBF(ap.get_pf()[0])
                #ch = np.random.uniform(low=0., high=1.0)
                #if ch < DarkBFEpBrem:
                #    npart = self.DarkElecBremSample(ap)
                #else:
                #    npart = self.DarkAnnihilationSample(ap)
                if self.GetBSMWeights(ap.get_ids()["PID"], ap.get_pf()[0], process="ExactBrem") > 0.0:
                    #npart = self.DarkElecBremSample(ap, relative_weight=DarkBFEpBrem)
                    npart = self.produce_bsm_particle(ap, process="ExactBrem")
                    NewShower.append(npart)
                if self.GetBSMWeights(ap.get_ids()["PID"], ap.get_pf()[0], process="Ann") > 0.0:
                    #npart2 = self.DarkAnnihilationSample(ap, relative_weight=(1.0-DarkBFEpBrem))
                    npart = self.produce_bsm_particle(ap, process="Ann")
                    NewShower.append(npart)
            elif ap.get_ids()["PID"] == 22:
                #if ap.get_pf()[0] < self._mV*(1.0 + self._mV/(2*m_electron)) or np.isnan(ap.get_pf()[0]):
                #    continue
                if self.GetBSMWeights(ap.get_ids()["PID"], ap.get_pf()[0], process="Comp") > 0.0:
                    npart = self.produce_bsm_particle(ap, process="Comp")
                    NewShower.append(npart)

        return ShowerToSamp, NewShower
        '''