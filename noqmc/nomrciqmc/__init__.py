import logging
import os
import numpy as np

from noqmc.utils.utilities import Parameters

from noqmc.nomrccmc import NOCCMC
from noqmc.nomrciqmc.propagator import Propagator

from pyscf.gto import Mole
from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE

DEFAULT_CIQMC_ARGS = Parameters(
    mode='noci',
    verbosity=1,
    seed=69420,
    dt=0.01,
    nr_w=3000,
    A=10,
    c=0.01,
    it_nr=50000,
    delay=20000,
    theory_level=1,
    benchmark=1,
)

THRESHOLDS = {
    'ov_zero_th':       5e-06,
    'rounding':         int(-np.log10(ZERO_TOLERANCE))-4,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NOCIQMC(NOCCMC):
        """Object that wraps initialization of the system, running the 
        population dynamics, processing the results and performing a 
        blocking analysis on it."""
        def __init__(self, mol: Mole, params: Parameters=None):
                super().__init__(mol, params)

        def initialize_log(self) -> None:
                r""""""
                filename = os.path.join(self.params.workdir, 
                                        f'nociqmc_{os.getpid()}.log')
                logging.basicConfig(
                        filename=filename, #format='%(levelname)s: %(message)s', 
                        ) #level=logging.INFO

        def run(self) -> Propagator:
                r"""Executes the population dynamics algorithm."""
                if not self.initialized: #self.initialize_references()
                        self.system.initialize()
                self.prop = Propagator(self.system)
                self.prop.run()
                self.__dict__.update(self.prop.__dict__)
                return self.prop


if __name__ == '__main__':
        from pyscf import gto
        mol = gto.M(atom=[['H', 0, 0, 0], ['H', 0, 0, 1.8]], 
                    basis='sto-3g', unit='Angstrom')
        
        my_nociqmc = NOCIQMC(mol)
        my_nociqmc.run()

