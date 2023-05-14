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
        
        r = 3.8
        mol = gto.M(atom=[["H", 0., 0., 0.],["H", 0., 0., r ]], 
                    basis='sto-3g', verbose=0, unit='Angstrom'
        )   
        guess_rhf, guess_uhf = get_init_guess_H2(r)
        print(guess_rhf, guess_uhf)
        exit()

        params = { 
                'mode': 'ref',
                'verbosity': 1,
                'seed': 69420,
                'dt': 0.03,
                'nr_w': 4000,
                'A': 10, 
                'c': 0.05,
                'it_nr': 2000,
                'delay': 200,
                'theory_level': 2,
                'benchmark': 1,
                'localization': 0,
                'scf_sols': [1,1,1],
                'sampling': 'heatbath',
                'binning': 1,
                'baseS': 'noci'
        }


        my_nociqmc = no.NOCIQMC(mol=mol, params=params)
        my_nociqmc.initialize_references(
                guess_rhf=guess_rhf, guess_uhf=guess_uhf
        )
        my_prop = my_nociqmc.run()
        my_nociqmc.get_data()

        my_nociqmc.plot()

