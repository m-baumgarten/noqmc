import numpy as np

from noqmc.utils.utilities import Parser

from noqmc.nomrccmc.system import System
from noqmc.nomrciqmc.propagator import Propagator
from noqmc.nomrccmc.postprocessor import Postprocessor
from noqmc.nomrccmc.statistics import Statistics

from pyscf.gto import Mole
from pyscf import gto
from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE

DEFAULT_CIQMC_ARGS = {
    'mode': 'noci',
    'verbosity': 1,
    'seed': 69420,
    'dt': 0.01,
    'nr_w': 3000,
    'A': 10,
    'c': 0.01,
    'it_nr': 50000,
    'delay': 20000,
    'theory_level': 1,
    'benchmark': 1,
}

THRESHOLDS = {
    'ov_zero_th':       5e-06,
    'rounding':         int(-np.log10(ZERO_TOLERANCE))-4,
}

class NOCIQMC(Propagator):
        """Object that wraps initialization of the system, running the 
        population dynamics, processing the results and performing a 
        blocking analysis on it."""
        def __init__(self, mol: Mole, params = None):
                if params is not None:
                        if isinstance(params, dict):
                                params = params
                        elif isinstance(params, str):
                                params = Parser().parse(params)
                else: params = DEFAULT_CIQMC_ARGS 
                self.params = params
                self.mol = mol
                self.system = System(mol = mol, params = params)
                self.initialized = False
                
        def run(self) -> Propagator:
                r"""Executes the population dynamics algorithm."""
                if not self.initialized: self.initialize_references() 
                self.prop = Propagator(self.system)
                self.prop.run()
                self.__dict__.update(self.prop.__dict__)
                return self.prop

        def initialize_references(self, guess_rhf: np.ndarray = None, 
                guess_uhf: np.ndarray = None
                ) -> None:
                r"""Generates the SCF solutions required to run the population
                dynamics. Currently, 3 SCF solutions are generated: 1 RHF and 
                2 UHF solutions. However, to generalize the code, just change
                this function correspondingly.
                
                :param guess_rhf: 
                :param guess_uhf:"""
                self.system.get_reference(
                    guess_rhf = guess_rhf, guess_uhf = guess_uhf
                )
                self.system.initialize()
                self.initialized = True

        def get_data(self):
                r"""After running the population dynamics, get_data() will be
                able to extract the Shift, coefficients, projected energy,
                matrix elements and an error analysis."""
                self.postpr = Postprocessor(self.prop)
                self.postpr.postprocessing(benchmark = self.params['benchmark'])
                
                self.stat = Statistics(self.prop.Ss, self.params)
                self.stat.analyse() 

if __name__ == '__main__':
        mol = gto.M(atom = [['H', 0, 0, 0], ['H', 0, 0, 1.8]], 
            basis = 'sto-3g', unit = 'Angstrom')
        
        my_nociqmc = NOCIQMC(mol)
        my_nociqmc.run()
