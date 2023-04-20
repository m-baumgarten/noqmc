import numpy as np

from pyscf import gto
import noqmc.nomrciqmc as no


def get_init_guess_H2(r: float):
        r""""""
        mol_data = np.load(
                '../density_matrices/H2_dms.npy', allow_pickle=True
        )[()]
        index = (np.isclose(mol_data['x'], r)).nonzero()[0][0]
        guess_rhf = mol_data['rhf_dm'][index]
        guess_uhf = mol_data['uhf_dm'][index]
        return guess_rhf, guess_uhf

if __name__ == '__main__':
        
        r = 1.8
        mol = gto.M(atom=[["H", 0., 0., 0.],["H", 0., 0., r ]], 
                    basis='sto-3g', verbose=0, unit='Angstrom'
        )   
        guess_rhf, guess_uhf = get_init_guess_H2(r)

        params = { 
                'mode': 'noci',
                'verbosity': 1,
                'seed': 69420,
                'dt': 0.01,
                'nr_w': 2000,
                'A': 10, 
                'c': 0.05,
                'it_nr': 5000,
                'delay': 500,
                'theory_level': 2,
                'benchmark': 1,
                'localization': 0,
                'scf_sols': [1,1,0],
                'sampling': 'uniform',
                'binning': 1,
                'baseS': 'hf'
        }


        my_nociqmc = no.NOCIQMC(mol=mol, params=params)
        my_nociqmc.initialize_references(
                guess_rhf=guess_rhf, guess_uhf=guess_uhf
        )
        my_prop = my_nociqmc.run()
 #       print(my_prop.E_NOCI)
        my_nociqmc.get_data()
#        print(my_nociqmc.__dict__)

        my_nociqmc.plot()

