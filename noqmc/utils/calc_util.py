#!/usr/bin/env python
# Author: James D Whitfield <jdwhitfield@gmail.com>
'''
Scan H2 molecule dissociation curve comparing UHF and RHF solutions per the 
example of Szabo and Ostlund section 3.8.7

The initial guess is obtained by mixing the HOMO and LUMO and is implemented
as a function that can be used in other applications.

See also 16-h2_scan.py, 30-scan_pes.py, 32-break_spin_symm.py
'''

import logging
import numpy as np
from typing import Sequence

from pyscf import fci, scf, gto, cc
from pyscf.tools.molden import from_scf
from pyscf.lo import Boys 
import os
from qcmagic.core.sspace.single_determinant import SingleDeterminant

from qcmagic.interfaces.converters.pyscf import scf_to_state
from noqmc.utils.excips import flatten

from noqmc.utils.utilities import setup_workdir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

#write generate_rhf, generate_uhf, and switch_uhf 

def generate_scf(mol, scf_sols, init_guess_rhf=None, init_guess_uhf=None, 
                 workdir='output', localization=True) -> Sequence:
        r""""""

        scf_solutions = []

        #Generate RHF solutions
        for sol in range(scf_sols[0]):
                rhf = scf.RHF(mol)
                if init_guess_rhf is None:
                        erhf = rhf.kernel()
                else:
                        erhf = rhf.kernel(init_guess_rhf)
                from_scf(rhf, os.path.join(workdir, f'rhf_{sol}.molden'))
                rhf = scf.addons.convert_to_uhf(rhf)
                rhf.mo_coeff = np.array(rhf.mo_coeff)
                scf_solutions.append(rhf)                

        #Generate UHF solutions
        for sol in range(scf_sols[1] + scf_sols[2]):
                uhf = scf.UHF(mol)
                if init_guess_uhf is None:
                        euhf = uhf.kernel(init_guess_mixed(mol, rhf, uhf))
                else:
                        euhf = uhf.kernel(init_guess_uhf)
                
                if sol >= scf_sols[1]:
                        uhf.mo_coeff[1], uhf.mo_coeff[0] = uhf.mo_coeff[0].copy(), uhf.mo_coeff[1].copy()

                from_scf(uhf, os.path.join(workdir, f'uhf_{sol}.molden'))
                scf_solutions.append(uhf)

        if localization:
                for sol in scf_solutions:
                        localize(sol)
                        #use mulliken charges later for assignment of 
                        #electrons to certain local areas, allowing 
                        #for excitation generation between locally 
                        #adjacent areas.
                        a = sol.mulliken_pop()
                        print('Mulliken Charges:        ', a)
        
        #TODO exchange rhf.mo_coeff with concatenated version of all MO's/Mulliken charges?
        MO_AO_MAP = {i: np.where(np.abs(mo) == np.max(np.abs(mo)))[0][0] 
                     for i, mo in enumerate(rhf.mo_coeff.T)}

        MO_AO_MAP = {}
        dim = len(scf_solutions[0].mo_coeff.T)
        test = []
        for i_sol, sol in enumerate(scf_solutions):
                for i_spinspace, spinspace in enumerate(sol.mo_coeff):
                        MO_AO_MAP.update(
                                {2*dim*i_sol + dim*i_spinspace + i_mo: 
                                 2*dim*i_sol + dim*i_spinspace + np.where(np.abs(mo) == np.max(np.abs(mo)))[0][0]
                                 for i_mo, mo in enumerate(spinspace.T)}
                        )
        print(MO_AO_MAP)
        exit()

        dump_fci_ccsd(rhf, workdir=workdir)

        scf_solutions = [scf_to_state(sol) for sol in scf_solutions]
        return scf_solutions

def dump_fci_ccsd(mf, workdir='output') -> None:
        r""""""
        cisolver = fci.FCI(mf)
        efci = cisolver.kernel()[0]
        mycc = cc.CCSD(mf).run()
        with open(os.path.join(workdir, 'fci.txt'), 'w') as f:
                f.write('E(FCI)  = %.12f' % efci)
                f.write('\nE(CCSD) = %.12f' % mycc.e_tot)

def localize(mf) -> np.ndarray:
        
        #get occupations of converged MOs
        occs = mf.mo_occ
        if len(occs.shape) == 1: occs = occs[np.newaxis,:]
        mol = mf.mol
        c = mf.mo_coeff
        if len(c.shape) == 2: c = c[np.newaxis, :, :] 

        occ, virt = [], []
        
        for occ_spinspace in occs:
                
                occ_tmp, virt_tmp = [], []
                
                for j,o in enumerate(occ_spinspace):
                        target = virt_tmp if np.isclose(o, 0.) else occ_tmp
                        target.append(j)
                
                occ.append(occ_tmp)
                virt.append(virt_tmp)
        
        #Perform localization
        for i, (o, v) in enumerate(zip(occ, virt)):        
                c_i = c[i, :, :]
                c_i[:, o] = Boys(mol, c_i[:, o]).kernel(verbose=4)
                c_i[:, v] = Boys(mol, c_i[:, v]).kernel(verbose=4)

        print('localization done')

        return c


def tensor2number(indices: np.ndarray, shape: np.ndarray) -> int:
        r"""E.g. indices = (3,4,1) of tensor with shape (5,5,5)"""
        number = 0
        for i in range(len(shape)):
                number += indices[i] * np.prod(shape[i+1:])
        return number


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
