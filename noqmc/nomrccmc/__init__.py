import numpy as np

from noqmc.utils.utilities import Parser

from noqmc.nomrccmc.system import System
from noqmc.nomrccmc.propagator import Propagator
from noqmc.nomrccmc.postprocessor import Postprocessor
from noqmc.nomrccmc.statistics import Statistics

from pyscf.gto import Mole
from pyscf import gto

DEFAULT_CCMC_ARGS = {
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

class NOCCMC(Propagator):
        __doc__ = """Initialises the system and sets up the propagator.\n""" + Propagator.__doc__
        def __init__(self, mol: Mole, params = None):
                if params is not None:
                        if isinstance(params, dict):
                                params = params
                        elif isinstance(params, str):
                                params = Parser().parse(params)
                else: params = DEFAULT_CCMC_ARGS 
                
                self.mol = mol
                self.system = System(mol = mol, params = params)
                self.initialized = False

#                return self.system
                
        def run(self):
                if not self.initialized: self.initialise_references() 
                self.prop = PropagatorCC(self.system)
                self.prop.run()
                return self.prop

        def initialise_references(self, guess_rhf: np.ndarray = None, 
                guess_uhf: np.ndarray = None
                ):
                r""""""
                self.system.get_reference(
                    guess_rhf = guess_rhf, guess_uhf = guess_uhf
                )
                self.system.initialize()
                self.initialized = True

        def get_data(self):
                pass

        def plot(self):
                pass


if __name__ == '__main__':
        mol = gto.M(atom = [['H', 0, 0, 0], ['H', 0, 0, 1.3]], 
            basis = 'sto-3g', unit = 'Angstrom')
        
        my_noccmc = NOCCMC(mol)
        my_noccmc.run()
