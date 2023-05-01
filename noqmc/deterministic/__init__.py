import logging
import os
import numpy as np

from noqmc.utils.utilities import (
        Parser,
        Parameters,
)

from noqmc.utils.plot import Plot 
from noqmc.utils.glob import DEFAULT_DETERMINISTIC_ARGS
from noqmc.nomrccmc import NOCCMC
from noqmc.nomrccmc.system import System
from noqmc.deterministic.propagator import Propagator
from noqmc.nomrccmc.postprocessor import Postprocessor

from pyscf.gto import Mole

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Deterministic(NOCCMC):
        r""""""
        def __init__(self, mol: Mole, params: Parameters=None):
                if params is None:
                        params = DEFAULT_DETERMINISTIC_ARGS
                super().__init__(mol, params)
        

        def initialize_log(self) -> None:
                r""""""
                filename = os.path.join(self.params.workdir, 
                                        f'deterministic_{os.getpid()}.log')
                logging.basicConfig(
                        filename=filename, #format='%(levelname)s: %(message)s', 
                ) #level=logging.INFO

               
        def run(self) -> Propagator:
                if not self.initialized: 
                        self.system.initialize() 
                
                self.prop = Propagator(self.system)
                self.prop.run()
                self.__dict__.update(self.prop.__dict__)
                return self.prop

        def get_data(self) -> None:
                r"""After running the population dynamics, get_data() will be
                able to extract the Shift, coefficients, projected energy,
                matrix elements and an error analysis."""
                self.postpr = Postprocessor(self.prop)
                self.postpr.postprocessing(benchmark = self.params.benchmark)
                                
        def plot(self) -> None:
                import matplotlib.pyplot as plt
                plot = Plot()
                data = plot.add_data(self.postpr)
                plot.setup_figure(data)
                #plot.plot_data()
                
                plot.plot_energy(plot.ax[0,1])
                plot.plot_coeffs(plot.ax[0,0], plot.ax[1,0])
                #plot.plot_walkers(plot.ax[0,2])
                #self.plot_stderr(plot.ax[1,1])
                plot.plot_nullspace(plot.ax[1,2])
                plt.savefig('summary.png')
                plt.close()

                plot.plot_l1() 


if __name__ == '__main__':
        from pyscf import gto
#        mol = gto.M(atom=[['H', 0, 0, 0], ['H', 0, 0, 1.8]], 
#                    basis='sto-3g', unit='Bohr')

        

        basis = '6-31g*'
        mol = gto.M(
                atom=[["He", 0., 0., 0.],
                      ["He", 0., 0., 2.5 ],
                      ["He", 0., 0., 5.]],
                basis=basis, verbose=0,
                unit='Angstrom'
        )


        my_det = Deterministic(mol)
        my_det.initialize_references()
        my_det.run()
        my_det.get_data()
        my_det.plot()
        
