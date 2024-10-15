import numpy as np

from PETITE.physical_constants import m_proton_grams


"""
target_information is a dictionary that is loaded in throughout the code
if additional targets are desired, please add them here
Required information:
    Z_T: atomic number of target
    A_T: atomic mass of target
    mT: mass of target in GeV (needed for dark shower production)
    rho: density of target in g/cm^3
"""
target_information = {
    "graphite": {"Z_T": 6, "A_T": 12, "mT": 11.178, "rho": 2.210},
    "lead": {"Z_T": 82, "A_T": 207, "mT": 207.2, "rho": 11.35},
    "iron": {"Z_T": 26, "A_T": 56, "mT": 55.845, "rho": 8.00},
    "hydrogen": {"Z_T": 1, "A_T": 1, "mT": 1.0, "rho": 1.0},
    "aluminum": {"Z_T": 13, "A_T": 27, "mT": 26.9815385, "rho": 2.699},
    "tungsten": {"Z_T": 74, "A_T": 183.84, "mT": 183.84, "rho": 19.3},
    "molybdenum": {"Z_T": 42, "A_T": 95.95, "mT": 95.95, "rho": 10.2},
}

for tm in target_information:
    target_information[tm]["dEdx"] = 2.0 * target_information[tm]["rho"]  # MeV/cm


class Target:
    """Representation of a nuclear target"""

    def __init__(self, name):
        """
        Initializes the target object.

        Args:
            Z: atomic number of the target
            A: atomic mass of the target
            rho: density of the target in g/cm^3
            dEdx: energy loss of the target in MeV/cm
        """
        self.name = name

        try:
            self._target_material = target_information[self.name.lower()]
        except KeyError:
            raise ValueError(
                f"Target material {self.name} not implemented. Please choose from {list(target_information.keys())}"
            )

        self.Z = self._target_material["Z_T"]
        self.A = self._target_material["A_T"]
        self.rho = self._target_material["rho"]
        self.dEdx = self._target_material["dEdx"]

        self.nTarget = self.rho / m_proton_grams / self.A
        self.nElecs = self.nTarget * self.Z

    def get_material_properties(self):
        """Returns target material properties: Z, A, rho, dE/dx"""
        return self.Z, self.A, self.rho, self.dEdx

    def get_n_targets(self):
        """Returns nuclear and electron target densities for the
        target material in 1/cm^3
        """

        return self.nTarget, self.nElecs

    def get_n_target_for_process(self, process):
        """Returns nuclear and electron target densities for the
        target material in 1/cm^3
        """
        if process in ["Ann", "Comp", "Moller", "Bhabha"]:
            return self.nElecs
        elif process in ["Brem", "PairProd"]:
            return self.nTarget
        else:
            raise ValueError(
                f"process={process} not implemented for target {self.name}. "
            )
