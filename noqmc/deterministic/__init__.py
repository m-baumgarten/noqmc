import numpy as np

from noqmc.utils.utilities import Parser
from noqmc.utils.plot import Plot 

from noqmc.nomrccmc.system import System
from noqmc.deterministic.propagator import Propagator
from noqmc.nomrccmc.postprocessor import Postprocessor
from noqmc.nomrccmc.statistics import Statistics

from pyscf.gto import Mole
from pyscf import gto
from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE

DEFAULT_DETERMINISTIC_ARGS = {
        'mode': 'ref',
        'verbosity': 1,
        'seed': 69420,
        'dt': 0.1,
        'nr_w': 3000,
        'A': 10,
        'c': 0.01,
        'it_nr': 21,
        'delay': 20,
        'theory_level': 2,
        'benchmark': 1,
        'localization': 0,
        'nr_scf': 2,
}

THRESHOLDS = {
        'ov_zero_th':       5e-06,
        'rounding':         int(-np.log10(ZERO_TOLERANCE))-4,
}

class Deterministic(Propagator):
        __doc__ = """Executes the deterministic QMC power-method-like propagation.\n""" + Propagator.__doc__
        def __init__(self, mol: Mole, params = None):
                if params is not None:
                        if isinstance(params, dict):
                                params = params
                        elif isinstance(params, str):
                                params = Parser().parse(params)
                else: params = DEFAULT_DETERMINISTIC_ARGS 
 
                self.params = params
                self.mol = mol
                self.system = System(mol=mol, params=params)
                self.initialized = False

                
        def run(self) -> Propagator:
                if not self.initialized: self.initialize_references() 
                self.prop = Propagator(self.system)
                self.prop.run()
                self.__dict__.update(self.prop.__dict__)
                return self.prop

        def initialize_references(self, guess_rhf: np.ndarray = None, 
                                  guess_uhf: np.ndarray = None) -> None:
                r""""""
                self.system.get_reference(
                    guess_rhf=guess_rhf, guess_uhf=guess_uhf
                )
                self.system.initialize()
                self.initialized = True

        def get_data(self) -> None:
                r"""After running the population dynamics, get_data() will be
                able to extract the Shift, coefficients, projected energy,
                matrix elements and an error analysis."""
                self.postpr = Postprocessor(self.prop)
                self.postpr.postprocessing(benchmark = self.params['benchmark'])
                                
                #self.stat = Statistics(self.prop.Ss, self.params)
                #self.stat.analyse() 
                #self.postpr.data_summary = self.stat.data_summary

        def plot(self) -> None:

                plot = Plot()
                data = plot.add_data(self.postpr)
                plot.setup_figure(data)
                #plot.plot_data()
                
                plot.plot_energy(plot.ax[0,1])
                plot.plot_coeffs(plot.ax[0,0], plot.ax[1,0])
                plot.plot_walkers(plot.ax[0,2])
                #self.plot_stderr(plot.ax[1,1])
                plot.plot_nullspace(plot.ax[1,2])
                plt.savefig('summary.png')
                plt.close()

                plot.plot_l1() 


if __name__ == '__main__':
        mol = gto.M(atom=[['H', 0, 0, 0], ['H', 0, 0, 1.8]], 
                    basis='sto-3g', unit='Bohr')

        basis = '6-31g*'
        mol = gto.M(
                atom=[["He", 0., 0., 0.],
                      ["He", 0., 0., 2.5 ],
                      ["He", 0., 0., 5.],
                      ["He", 0., 0., 7.5]],
                basis = basis, verbose = 0,
                unit = 'Angstrom'
        )


        my_nociqmc = Deterministic(mol)
        my_nociqmc.run()
        my_nociqmc.get_data()
        my_nociqmc.plot()
        
