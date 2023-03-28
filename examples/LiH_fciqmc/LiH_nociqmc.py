import numpy as np

from pyscf import gto
import noqmc.nomrciqmc as no


def get_init_guess_LiH(r: float):
        r""""""
        mol_data = np.load(
                '../density_matrices/LiH_dms.npy', allow_pickle=True
        )[()]
        index = (np.isclose(mol_data['x'], r)).nonzero()[0][0]
        guess_rhf = mol_data['rhf_dm'][index]
        guess_uhf = mol_data['uhf_dm'][index]
        return guess_rhf, guess_uhf

def setup_mol(r: float):
        basis = {'H': gto.basis.parse('''
                H       S
                        13.01000        0.01968500
                        1.962000        0.1379770
                        0.444600        0.4781480
                        0.122000        0.5012400'''),
                        'Li': gto.basis.parse('''
                Li      S
                        1469.000        0.0007660       -0.0001200
                        220.5000        0.0058920       -0.0009239
                        50.26000        0.0296710       -0.0046890
                        14.24000        0.1091800       -0.0176820
                        4.581000        0.2827890       -0.0489020
                        1.580000        0.4531230       -0.0960090
                        0.564000        0.2747740       -0.1363800
                        0.073450        0.0097510       0.57510200
                        0.028050        -0.003180       0.51766100
                Li      P
                        1.534000        0.0227840       0.00000000
                        0.274900        0.1391070       0.00000000
                        0.073620        0.5003750       0.00000000
                        0.024030        0.5084740       1.00000000'''
                )
        }

        mol = gto.M(atom=[["Li", 0., 0., 0.],["H", 0., 0., r ]],
                    basis = basis, verbose = 0, unit = 'Angstrom'
        )
        return mol

if __name__ == '__main__':

        #unsmoothness: mode 'noci', dt 0.002, nr_w 4000, A 12, c 0.03, it_nr 40000, delay 10000, theory_level 1
        r = 3
        mol = setup_mol(r=r)

        guess_rhf, guess_uhf = get_init_guess_LiH(r=r)

        params = { 
                'mode': 'noci',
                'verbosity': 1,
                'seed': 69420,
                'dt': 0.005,
                'nr_w': 4000,
                'A': 12, 
                'c': 0.005,
                'it_nr': 8000,
                'delay': 2000,
                'theory_level': 1,
                'benchmark': 1,
                'localization': 0,
                'scf_sols': [1,1,1]
        }


        my_nociqmc = no.NOCIQMC(mol=mol, params=params)
        my_nociqmc.initialize_references(
                guess_rhf=guess_rhf, guess_uhf=guess_uhf
        )
        my_prop = my_nociqmc.run()
        my_nociqmc.get_data()
 
        my_nociqmc.plot()




