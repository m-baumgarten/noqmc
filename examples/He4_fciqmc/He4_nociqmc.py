import numpy as np

from pyscf import gto
import noqmc.nomrciqmc as no


def get_init_guess_He4(r: float):
        r""""""
        mol_data = np.load(
                '../density_matrices/He4_dms.npy', allow_pickle=True
        )[()]
        index = (np.isclose(mol_data['x'], r)).nonzero()[0][0]
        guess_rhf = mol_data['rhf_dm'][index]
        guess_uhf = mol_data['uhf_dm'][index]
        return guess_rhf, guess_uhf

def setup_mol(r: float):
        basis = '6-31g*'

        mol = gto.M(
                atom=[["He", 0., 0., 0.],
                      ["He", 0., 0., r ],
                      ["He", 0., 0., 5.],
                      ["He", 0., 0., 7.5]],
                basis = basis, verbose = 0, 
                unit = 'Angstrom'
        )
        return mol

if __name__ == '__main__':

        r = 2.5
        mol = setup_mol(r = r)

        guess_rhf, guess_uhf = get_init_guess_He4(r = r)

        params = { 
                'mode': 'ref',
                'verbosity': 1,
                'seed': 69420,
                'dt': 0.001,
                'nr_w': 8000,
                'A': 12, 
                'c': 0.01,
                'it_nr': 8000,
                'delay': 4000,
                'theory_level': 2,
                'benchmark': 1,
                'localization': 1,
                'nr_scf': 2
        }


        my_nociqmc = no.NOCIQMC(mol=mol, params=params)
        my_nociqmc.initialize_references(
                guess_rhf=guess_rhf, guess_uhf=guess_uhf
        )
        my_prop = my_nociqmc.run()
        my_nociqmc.get_data()
 
        my_nociqmc.plot()




