from noqmc.utils.utilities import Parser

from noqmc.nomrccmc.modules.system import System
from noqmc.nomrciqmc.modules.propagator import Propagator
from noqmc.nomrccmc.modules.postprocessor import Postprocessor
from noqmc.nomrccmc.modules.statistics import Statistics

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

class NOCIQMC(Propagator):
        __doc__ = """Initialises the system and sets up the propagator.\n""" + Propagator.__doc__
        #define system here
        def __init__(self, mol: Mole, args: dict = None):
                #parse in args, have a bunch of standard settings
                #define system
                #initializse propagator
                if args is not None:
                        if isinstance(args, dict):
                                system = System(args)
                        elif isinstance(args, str):
                                args = Parser().parse(args)
                else: args = DEFAULT_CIQMC_ARGS 

                self.system = System(args)

                return self.system
                #return Propagator()

        def get_references(self):
                #actuall, absorb this in the System class -> no HF calculations upon initialization of NOCIQMC
                pass

        def get_data(self):
                pass

        def plot(self):
                pass


if __name__ == '__main__':

        my_nociqmc = NOCIQMC()
        my_nociqmc.run()
