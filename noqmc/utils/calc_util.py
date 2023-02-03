#!/usr/bin/env python
# Author: James D Whitfield <jdwhitfield@gmail.com>
'''
Scan H2 molecule dissociation curve comparing UHF and RHF solutions per the 
example of Szabo and Ostlund section 3.8.7

The initial guess is obtained by mixing the HOMO and LUMO and is implemented
as a function that can be used in other applications.

See also 16-h2_scan.py, 30-scan_pes.py, 32-break_spin_symm.py
'''

import numpy as np
from typing import Sequence

from pyscf import fci, scf, gto, cc
from pyscf.tools.molden import from_scf
import os
from qcmagic.core.sspace.single_determinant import SingleDeterminant

from qcmagic.interfaces.converters.pyscf import scf_to_state
from excips import flatten

from noqmc.utils.utilities import setup_workdir

def init_guess_mixed(mol,rhf,uhf,mixing_parameter=np.pi/4):
    ''' Generate density matrix with broken spatial and spin symmetry by mixing
    HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.
    
    psi_1a = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
    psi_1b = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
        
    psi_2a = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
    psi_2b =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo

    Returns: 
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    # opt: q, mixing parameter 0 < q < 2 pi
    
    #based on init_guess_by_1e
    h1e = scf.hf.get_hcore(mol)
    s1e = scf.hf.get_ovlp(mol)
    mo_energy, mo_coeff = rhf.eig(h1e, s1e)
    mo_occ = rhf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx=0
    lumo_idx=1

    for i in range(len(mo_occ)-1):
        if mo_occ[i]>0 and mo_occ[i+1]<0:
            homo_idx=i
            lumo_idx=i+1

    psi_homo=mo_coeff[:, homo_idx]
    psi_lumo=mo_coeff[:, lumo_idx]
    
    Ca=np.zeros_like(mo_coeff)
    Cb=np.zeros_like(mo_coeff)


    #mix homo and lumo of alpha and beta coefficients
    q=mixing_parameter

    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:,k] = np.cos(q)*psi_homo + np.sin(q)*psi_lumo
            Cb[:,k] = np.cos(q)*psi_homo - np.sin(q)*psi_lumo
            continue
        if k==lumo_idx:
            Ca[:,k] = -np.sin(q)*psi_homo + np.cos(q)*psi_lumo
            Cb[:,k] =  np.sin(q)*psi_homo + np.cos(q)*psi_lumo
            continue
        Ca[:,k]=mo_coeff[:,k]
        Cb[:,k]=mo_coeff[:,k]

    dm =scf.UHF(mol).make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
    return dm 


def generate_scf(mol, init_guess_rhf = None, init_guess_uhf = None, workdir = 'output'):
    
    setup_workdir(workdir)

    rhf = scf.RHF(mol)
    uhf = scf.UHF(mol)
    if init_guess_rhf is None:
        erhf = rhf.kernel()
    else:
        erhf = rhf.kernel(init_guess_rhf)
    
    if init_guess_uhf is None:
        euhf = uhf.kernel(init_guess_mixed(mol, rhf, uhf))
    else: 
        euhf = uhf.kernel(init_guess_uhf) 

    from_scf(rhf, os.path.join(workdir, 'rhf.molden'))
    from_scf(uhf, os.path.join(workdir, 'uhf.molden'))

    sd_rhf = scf_to_state(rhf)
    rhf_occ = [sd_rhf.occupied_coefficients[0]] * 2
    rhf_coeffs = [sd_rhf.coefficients[0]] * 2
    sd_new_rhf = SingleDeterminant(n_electrons = sd_rhf.n_electrons * 2, 
                                   config = sd_rhf.configuration, holo = False,
                                   full_coeffs = rhf_coeffs)
    
    sd_uhf1 = scf_to_state(uhf)
    sd_uhf2 = sd_uhf1.copy_from(sd_uhf1, dtype=np.float64)
    sd_uhf2.coefficients[1], sd_uhf2.coefficients[0] = sd_uhf2.coefficients[0], sd_uhf2.coefficients[1]

    cisolver = fci.FCI(rhf)
    efci = cisolver.kernel()[0]
    mycc = cc.CCSD(rhf).run()
    with open(os.path.join(workdir, 'fci.txt'), 'w') as f:
        f.write('E(FCI)  = %.12f' % efci)
        f.write('\nE(CCSD) = %.12f' % mycc.e_tot)

    return sd_new_rhf, sd_uhf1, sd_uhf2

def tensor2number(indices: np.ndarray, shape: np.ndarray) -> int:
        r"""E.g. indices = (3,4,1) of tensor with shape (5,5,5)"""
        number = 0
        for i in range(len(shape)):
                number += indices[i] * np.prod(shape[i+1:])
        return number


#def number2tensor(number: int, shape: np.ndarray) -> np.ndarray:
#        r"""Reverses tensor2number and yields the indices corresp.
#        to the number provided"""
#        dim = len(shape)
#        indices = np.zeros(dim)
#        for i in range(dim):
#                indices[i], number = divmod(number, np.prod(shape[i+1:]))
#        return indices

def number2tensor(number: int, shape: np.ndarray) -> np.ndarray:
        r"""Reverses tensor2number and yields the indices corresp.
        to the number provided"""
        dim = len(shape)
        indices = np.zeros(dim, dtype = np.int32)
        prod = np.prod(shape[1:])
        for i in range(dim-1):
                indices[i], number = divmod(number, prod)
                prod //= shape[i+1]
        indices[-1], number = divmod(number, prod)
        return indices


def exstr2tensor(exstr, excitation_level: int) -> np.ndarray:
        r""""""
        scf_sol = np.array([exstr[0]])
        ex_a = exstr[1][0]
        ex_b = exstr[1][1]

        spin = np.array(len(ex_a) * [0] + len(ex_b) * [1])
        flat_ex = flatten(exstr[1])
        flat_dex = flatten(exstr[2])
        fill_zeros = np.zeros( excitation_level - len(flat_ex) )
        return np.concatenate((scf_sol, flat_ex, fill_zeros, flat_dex, 
                                fill_zeros, spin, fill_zeros))

def exstr2number(exstr, shape, ex_lvl) -> int:
        r""""""
        tensor = exstr2tensor(exstr, ex_lvl)
        return tensor2number(indices = tensor, shape = shape)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    l = []
    euhf = []
    erhf = []
    x = np.arange(5,5.8,0.1)
    for r in x:
        mol = gto.M(atom=[["Li", 0., 0., 0.],
                    ["H", 0., 0., r]], unit='Angstrom', basis='sto-3g', verbose=0)

        _,_, e, eu, er = generate_scf(mol)
        l.append(e)
        euhf.append(eu)
        erhf.append(er)
        
    plt.plot(x, np.array(l))
    plt.plot(x, np.array(euhf))
    plt.plot(x, np.array(erhf))
    plt.savefig('b.png')
    plt.show()
